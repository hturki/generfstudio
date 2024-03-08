"""
PixelNeRF field implementation.
"""
from typing import Tuple, Optional, Dict

import torch
import torch.nn.functional as F
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.fields.base_field import Field
from torch import Tensor
from torch import nn

from generfstudio.fields.resnet_fc import ResnetFC
from generfstudio.generfstudio_constants import NEIGHBORING_VIEW_CAMERAS, IMAGE_FEATURES, FEATURE_SCALING, GLOBAL_LATENT
from generfstudio.generfstudio_utils import repeat_interleave, get_pixel_aligned_features
from generfstudio.pixelnerf_utils.scaled_nerf_encoding import ScaledNeRFEncoding


class PixelNeRFField(Field):

    def __init__(
            self,
            encoder_latent_size: int,
            global_encoder_latent_size: int,
            out_feature_dim: int,
            combine_layer: int = 3,
            use_positional_encoding_xyz: bool = True,
            positional_encoding_scale_factor: float = 1.5,
            use_view_dirs: bool = True,
            normalize_z: bool = True,
    ) -> None:
        super().__init__()
        self.inner = PixelNeRFInnerField(encoder_latent_size, global_encoder_latent_size, out_feature_dim,
                                         combine_layer,
                                         use_positional_encoding_xyz, positional_encoding_scale_factor, use_view_dirs,
                                         normalize_z)
        self.out_feature_dim = out_feature_dim

    def get_density(self, ray_samples: RaySamples) -> Tuple[Tensor, Tensor]:
        neighboring_cameras = ray_samples.metadata[NEIGHBORING_VIEW_CAMERAS]

        neighboring_c2w = neighboring_cameras.camera_to_worlds.view(-1, 3, 4)
        rot = neighboring_c2w[:, :3, :3].transpose(1, 2)  # (B, 3, 3)
        trans = -torch.bmm(rot, neighboring_c2w[:, :3, 3:])  # (B, 3, 1)
        neighboring_w2c = torch.cat((rot, trans), dim=-1)

        image_count, neighbor_count = neighboring_cameras.shape
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


class PixelNeRFInnerField(nn.Module):

    def __init__(
            self,
            encoder_latent_size: int,
            global_encoder_latent_size: int,
            out_feature_dim: int,
            combine_layer: int = 3,
            use_positional_encoding_xyz: bool = True,
            positional_encoding_scale_factor: float = 1.5,
            use_view_dirs: bool = True,
            normalize_z: bool = True,
    ) -> None:
        super().__init__()
        self.use_global_encoder = global_encoder_latent_size > 0

        self.use_positional_encoding_xyz = use_positional_encoding_xyz
        if use_positional_encoding_xyz:
            self.positional_encoding_xyz = ScaledNeRFEncoding(
                in_dim=3, num_frequencies=6, min_freq_exp=0.0, max_freq_exp=5.0, scale=positional_encoding_scale_factor,
                include_input=True
            )
            d_in = self.positional_encoding_xyz.get_out_dim()
        else:
            d_in = 3

        self.use_view_dirs = use_view_dirs
        if self.use_view_dirs:
            d_in += 3

        self.mlp = ResnetFC(d_in=d_in, d_out=4 + out_feature_dim,
                            d_latent=encoder_latent_size + global_encoder_latent_size,
                            combine_layer=combine_layer)

        # Whether to shift z to align in canonical frame.
        # So that all objects, regardless of camera distance to center, will be centered at z=0.
        # Only makes sense in ShapeNet-type setting.
        self.normalize_z = normalize_z

    def forward(self, ray_samples: RaySamples, positions: torch.Tensor, image_features: torch.Tensor,
                neighboring_w2c: torch.Tensor, neighboring_cameras: Cameras, uv_scaling: torch.Tensor):
        image_count, neighbor_count = neighboring_cameras.shape
        xyz = repeat_interleave(positions, neighbor_count)

        xyz_rot = torch.matmul(neighboring_w2c[:, None, :3, :3], xyz.unsqueeze(-1))[..., 0]
        xyz = xyz_rot + neighboring_w2c[:, None, :3, 3]

        uv = -xyz[:, :, :2] / xyz[:, :, 2:]
        focal = torch.cat([neighboring_cameras.fx.view(-1, 1), -neighboring_cameras.fy.view(-1, 1)], -1)
        uv *= focal.unsqueeze(1)
        center = torch.cat([neighboring_cameras.cx.view(-1, 1), neighboring_cameras.cy.view(-1, 1)], -1)
        uv += center.unsqueeze(1)

        # (NV, latent_dim, num_samples)
        latent = get_pixel_aligned_features(image_features, uv, uv_scaling)
        latent = latent.reshape(-1, latent.shape[-1])  # (num_samples * NV, latent)

        if self.normalize_z:
            xyz_input = xyz_rot.reshape(-1, 3)  # (SB*B, 3)
        else:
            xyz_input = xyz.reshape(-1, 3)  # (SB*B, 3)

        if self.use_positional_encoding_xyz:
            xyz_input = self.positional_encoding_xyz(xyz_input)

        mlp_input = [latent, xyz_input]

        if self.use_view_dirs:
            dirs = ray_samples.frustums.directions.reshape(image_count, -1, 3)
            dirs = repeat_interleave(dirs, neighbor_count)
            dirs = torch.matmul(neighboring_w2c[:, None, :3, :3], dirs.unsqueeze(-1))
            dirs = dirs.reshape(-1, 3)
            mlp_input.append(dirs)

        if self.use_global_encoder:
            global_latent = ray_samples.metadata[GLOBAL_LATENT]
            num_repeats = positions.shape[0]
            global_latent = repeat_interleave(global_latent, num_repeats)
            mlp_input = [global_latent] + mlp_input

        mlp_input = torch.cat(mlp_input, -1)

        mlp_output = self.mlp(mlp_input, combine_inner_dims=(neighbor_count, positions.shape[1])).view(
            *ray_samples.frustums.shape, -1)

        density = F.softplus(mlp_output[..., :1])
        rgb = torch.sigmoid(mlp_output[..., 1:4])
        features = mlp_output[..., 4:]

        return density, rgb, features
