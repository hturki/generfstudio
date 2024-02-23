"""
PixelNeRF field implementation.
"""
from typing import Tuple, Optional, Dict

import torch
from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.encodings import NeRFEncoding
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.fields.base_field import Field
from rich.console import Console
from torch import Tensor
import torch.nn.functional as F

from generfstudio.fields.resnet_fc import ResnetFC
from generfstudio.generfstudio_constants import NEIGHBORING_VIEW_CAMERAS, LATENT, LATENT_SCALING, GLOBAL_LATENT, \
    ENCODER, NEIGHBORING_VIEW_COUNT
from generfstudio.pixelnerf_utils.scaled_nerf_encoding import ScaledNeRFEncoding

CONSOLE = Console(width=120)


class PixelNeRFField(Field):

    def __init__(
            self,
            encoder_latent_size: int,
            global_encoder_latent_size: int,
            combine_layer: int = 3,
            use_positional_encoding_xyz: bool = True,
            positional_encoding_scale_factor: float = 1.5,
            use_view_dirs: bool = True,
            normalize_z: bool = True,
            freeze_encoder: bool = False,
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

        self.mlp = ResnetFC(d_in=d_in, d_out=4, d_latent=encoder_latent_size + global_encoder_latent_size,
                            combine_layer=combine_layer)

        # Whether to shift z to align in canonical frame.
        # So that all objects, regardless of camera distance to center, will be centered at z=0.
        # Only makes sense in ShapeNet-type setting.
        self.normalize_z = normalize_z

        self.freeze_encoder = freeze_encoder

    def get_density(self, ray_samples: RaySamples) -> Tuple[Tensor, Tensor]:
        neighboring_view_cameras = ray_samples.metadata[NEIGHBORING_VIEW_CAMERAS]
        neighboring_view_count = ray_samples.metadata[NEIGHBORING_VIEW_COUNT]
        image_count = len(neighboring_view_cameras) // neighboring_view_count

        positions = ray_samples.frustums.get_positions().view(image_count, -1, 3)
        xyz = repeat_interleave(positions, neighboring_view_count)

        poses = neighboring_view_cameras.camera_to_worlds
        rot = poses[:, :3, :3].transpose(1, 2)  # (B, 3, 3)
        trans = -torch.bmm(rot, poses[:, :3, 3:])  # (B, 3, 1)
        poses = torch.cat((rot, trans), dim=-1)

        xyz_rot = torch.matmul(poses[:, None, :3, :3], xyz.unsqueeze(-1))[..., 0]
        xyz = xyz_rot + poses[:, None, :3, 3]

        uv = -xyz[:, :, :2] / xyz[:, :, 2:]
        focal = torch.cat([neighboring_view_cameras.fx, -neighboring_view_cameras.fy], -1)
        uv *= focal.unsqueeze(1)
        center = torch.cat([neighboring_view_cameras.cx, neighboring_view_cameras.cy], -1)
        uv += center.unsqueeze(1)

        latent_field = ray_samples.metadata[LATENT]
        latent_scaling = ray_samples.metadata[LATENT_SCALING]

        # TODO: This assumes that all camera views have the same dimensions
        width = neighboring_view_cameras.width[0].item()
        height = neighboring_view_cameras.height[0].item()

        # (NV, latent_dim, num_samples)
        latent = ray_samples.metadata[ENCODER].index(latent_field, latent_scaling, uv,
                                                     torch.FloatTensor([width, height]).to(uv.device))
        if self.freeze_encoder:
            latent = latent.detach()

        latent = latent.transpose(1, 2).reshape(-1, latent.shape[1])  # (num_samples * NV, latent)

        if self.normalize_z:
            xyz_input = xyz_rot.reshape(-1, 3)  # (SB*B, 3)
        else:
            xyz_input = xyz.reshape(-1, 3)  # (SB*B, 3)

        if self.use_positional_encoding_xyz:
            xyz_input = self.positional_encoding_xyz(xyz_input)

        mlp_input = [latent, xyz_input]

        if self.use_view_dirs:
            dirs = ray_samples.frustums.directions.reshape(image_count, -1, 3)
            dirs = repeat_interleave(dirs, neighboring_view_count)
            dirs = torch.matmul(poses[:, None, :3, :3], dirs.unsqueeze(-1))
            dirs = dirs.reshape(-1, 3)
            mlp_input.append(dirs)

        if self.use_global_encoder:
            global_latent = ray_samples.metadata[GLOBAL_LATENT]
            num_repeats = positions.shape[0]
            global_latent = repeat_interleave(global_latent, num_repeats)
            mlp_input = [global_latent] + mlp_input

        mlp_input = torch.cat(mlp_input, -1)

        mlp_output = self.mlp(mlp_input, combine_inner_dims=(neighboring_view_count, positions.shape[1])).view(
            *ray_samples.frustums.shape, -1)

        density = F.softplus(mlp_output[..., 3:])
        return density, mlp_output[..., :3]

    def get_outputs(
            self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None
    ) -> Dict[FieldHeadNames, Tensor]:
        return {FieldHeadNames.RGB: torch.sigmoid(density_embedding)}


@torch.compile
def repeat_interleave(input, repeats):
    """
    Repeat interleave along axis 0
    torch.repeat_interleave is currently very slow
    https://github.com/pytorch/pytorch/issues/31980
    """
    output = input.unsqueeze(1).expand(-1, repeats, *input.shape[1:])
    return output.reshape(-1, *input.shape[1:])
