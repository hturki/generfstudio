from typing import Tuple, Optional, Dict

import math
import torch
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components import MLP
from nerfstudio.field_components.encodings import NeRFEncoding
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.fields.base_field import Field
from rich.console import Console
from torch import Tensor
from torch import nn
from torch.nn import functional as F

from generfstudio.generfstudio_constants import NEIGHBORING_VIEW_CAMERAS, IMAGE_FEATURES, FEATURE_SCALING
from generfstudio.generfstudio_utils import repeat_interleave, get_pixel_aligned_features
from generfstudio.pixelnerf_utils.scaled_nerf_encoding import ScaledNeRFEncoding

CONSOLE = Console(width=120)


class IBRNetField(Field):

    def __init__(
            self,
            in_feature_dim: int = 128,
            out_feature_dim: int = 128,
            positional_encoding_scale_factor: float = math.pi,
    ) -> None:
        super().__init__()
        self.inner = IBRNetInnerField(in_feature_dim, out_feature_dim, positional_encoding_scale_factor)
        self.out_feature_dim = out_feature_dim

    def get_density(self, ray_samples: RaySamples) -> Tuple[Tensor, Tensor]:
        neighboring_cameras = ray_samples.metadata[NEIGHBORING_VIEW_CAMERAS]

        neighboring_c2w = neighboring_cameras.camera_to_worlds.view(-1, 3, 4)
        rot = neighboring_c2w[:, :3, :3].transpose(1, 2)  # (B, 3, 3)
        trans = -torch.bmm(rot, neighboring_c2w[:, :3, 3:])  # (B, 3, 1)
        neighboring_w2c = torch.cat((rot, trans), dim=-1)

        image_count = neighboring_cameras.shape[0]
        positions = ray_samples.frustums.get_positions().view(image_count, -1, 3)
        density, rgb, features = self.inner(ray_samples, positions, ray_samples.metadata[IMAGE_FEATURES],
                                            neighboring_w2c, neighboring_cameras, ray_samples.metadata[FEATURE_SCALING])
        return density, (torch.cat([rgb, features], -1) if self.out_feature_dim > 0 else rgb)

    def get_outputs(
            self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None
    ) -> Dict[FieldHeadNames, Tensor]:
        outputs = {FieldHeadNames.RGB: density_embedding[..., :3]}
        if self.out_feature_dim > 0:
            outputs[FieldHeadNames.SEMANTICS] = density_embedding[..., 3:]

        return outputs


class IBRNetInnerField(nn.Module):

    def __init__(
            self,
            in_feature_dim: int = 128,
            out_feature_dim: int = 128,
            positional_encoding_scale_factor: float = math.pi,
    ) -> None:
        super().__init__()

        self.position_encoding = ScaledNeRFEncoding(
            in_dim=3, num_frequencies=6, min_freq_exp=0.0, max_freq_exp=5.0, scale=positional_encoding_scale_factor,
            include_input=True
        )
        self.feature_pool_mlp = MLP(in_dim=self.position_encoding.get_out_dim() + 3 * in_feature_dim, num_layers=2,
                                    layer_width=256, out_dim=128 + 1, implementation="torch")
        self.volrend_mlp = MLP(in_dim=128, num_layers=4, layer_width=256, out_dim=out_feature_dim + 4,
                               implementation="torch")

    def forward(self, ray_samples: RaySamples, positions: torch.Tensor, image_features: torch.Tensor,
                neighboring_w2c: torch.Tensor, neighboring_cameras: Cameras, uv_scaling: torch.Tensor):
        neighbor_count = neighboring_cameras.shape[-1]
        xyz = repeat_interleave(positions, neighbor_count)

        xyz_rot = torch.matmul(neighboring_w2c[:, None, :3, :3], xyz.unsqueeze(-1))[..., 0]
        xyz_in_neighbor = xyz_rot + neighboring_w2c[:, None, :3, 3]

        uv = -xyz_in_neighbor[:, :, :2] / xyz_in_neighbor[:, :, 2:]
        focal = torch.cat(
            [neighboring_cameras.fx.view(-1, 1), -neighboring_cameras.fy.view(-1, 1)], -1)
        uv *= focal.unsqueeze(1)
        center = torch.cat(
            [neighboring_cameras.cx.view(-1, 1), neighboring_cameras.cy.view(-1, 1)], -1)
        uv += center.unsqueeze(1)

        interpolated_features = get_pixel_aligned_features(image_features, uv, uv_scaling)
        interpolated_features = interpolated_features.reshape(-1, neighbor_count,
                                                              *interpolated_features.shape[1:])
        features_mean = repeat_interleave(interpolated_features.mean(1), neighbor_count)
        features_var = repeat_interleave(interpolated_features.var(1), neighbor_count)
        encoded_xyz = repeat_interleave(self.position_encoding(positions), neighbor_count)
        feature_pool_input = torch.cat([
            interpolated_features.reshape(-1, interpolated_features.shape[-1]),
            features_mean.reshape(-1, features_mean.shape[-1]),
            features_var.reshape(-1, features_var.shape[-1]),
            encoded_xyz.reshape(-1, encoded_xyz.shape[-1])
        ], -1)
        pooled_features = self.feature_pool_mlp(feature_pool_input).view(*interpolated_features.shape[:-1], -1)
        feature_weights = torch.softmax(pooled_features[..., -1:], dim=1)
        volrend_mlp_input = (pooled_features[..., :-1] * feature_weights).sum(1)
        volrend_output = self.volrend_mlp(volrend_mlp_input.view(-1, volrend_mlp_input.shape[-1])).view(
            *ray_samples.frustums.shape, -1).to(ray_samples.frustums.directions)

        density = F.softplus(volrend_output[..., :1])
        rgb = torch.sigmoid(volrend_output[..., 1:4])
        features = volrend_output[..., 4:]

        return density, rgb, features
