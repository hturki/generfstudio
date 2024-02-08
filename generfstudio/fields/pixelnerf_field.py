"""
PixelNeRF field implementation.
"""
from abc import abstractmethod
from collections import defaultdict
from enum import Enum, auto
from typing import Tuple, Optional, Any

import torch
from jaxtyping import Shaped
from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.rays import RaySamples, Frustums
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.fields.base_field import Field, get_normalized_directions
from rich.console import Console
from torch import Tensor

from generfstudio.fields.image_encoder import ImageEncoder
from generfstudio.fields.spatial_encoder import SpatialEncoder

CONSOLE = Console(width=120)

class PixelNeRFField(Field):

    def __init__(
            self,
            use_encoder: bool = True,
            normalize_z: bool = True,
            freeze_encoder: bool = False,
            use_global_encoder: bool = False,
            use_positional_encoding_xyz: bool = True,
            use_dir: bool = True,
            use_positional_encoding_dir: bool = False,
    ) -> None:
        super().__init__()

        self.encoder = SpatialEncoder() if use_encoder else None

        # Whether to shift z to align in canonical frame.
        # So that all objects, regardless of camera distance to center, will be centered at z=0.
        # Only makes sense in ShapeNet-type setting.
        self.normalize_z = normalize_z

        self.freeze_encoder = freeze_encoder
        self.global_encoder = ImageEncoder() if use_global_encoder else None

    def get_density(self, ray_samples: RaySamples):
        positions = ray_samples.frustums.get_positions()

        # Transform query points into the camera spaces of the input views
        xyz = repeat_interleave(positions, self.num_views_per_obj)  # (SB*NS, B, 3)
        xyz_rot = torch.matmul(self.poses[:, None, :3, :3], xyz.unsqueeze(-1))[
            ..., 0
        ]

        xyz = xyz_rot + self.poses[:, None, :3, 3]

        if self.normalize_z:
            z_feature = xyz_rot.reshape(-1, 3)  # (SB*B, 3)
        else:
            z_feature = xyz.reshape(-1, 3)  # (SB*B, 3)

        mlp_input = z_feature

        if self.encoder is not None:
            # Grab encoder's latent code.
            uv = -xyz[:, :, :2] / xyz[:, :, 2:]  # (SB, B, 2)
            uv *= repeat_interleave(
                self.focal.unsqueeze(1), self.num_views_per_obj if self.focal.shape[0] > 1 else 1
            )
            uv += repeat_interleave(
                self.c.unsqueeze(1), self.num_views_per_obj if self.c.shape[0] > 1 else 1
            )  # (SB*NS, B, 2)

            latent = self.encoder.index(uv, self.image_shape)  # (SB * NS, latent, B)

            if self.freeze_encoder:
                latent = latent.detach()
            latent = latent.transpose(1, 2).reshape(-1, self.latent_size)  # (SB * NS * B, latent)

            mlp_input = torch.cat((latent, mlp_input), dim=-1)

        if self.use_global_encoder:
            # Concat global latent code if enabled
            global_latent = self.global_encoder.latent
            assert mlp_input.shape[0] % global_latent.shape[0] == 0
            num_repeats = mlp_input.shape[0] // global_latent.shape[0]
            global_latent = repeat_interleave(global_latent, num_repeats)
            mlp_input = torch.cat((global_latent, mlp_input), dim=-1)

        if self.spatial_distortion is not None:
            positions = self.spatial_distortion(positions)
            positions = (positions + 2.0) / 4.0
            positions_flat = positions.view(-1, 3)
        else:
            positions = SceneBox.get_normalized_positions(positions, self.aabb)
            positions_flat = positions.view(-1, 3)

        explicit_level = ray_samples.metadata is not None and EXPLICIT_LEVEL in ray_samples.metadata
        if explicit_level:
            level = ray_samples.metadata[EXPLICIT_LEVEL]
            pixel_levels = torch.full_like(ray_samples.frustums.starts[..., 0], level).view(-1)
        else:
            # Assuming pixels are square
            sample_distances = ((ray_samples.frustums.starts + ray_samples.frustums.ends) / 2)
            pixel_widths = (ray_samples.frustums.pixel_area.sqrt() * sample_distances).view(-1)

            pixel_levels = self.base_log - torch.log(pixel_widths) / self.log_scale_factor
            if self.trained_level_resolution is not None:
                reso_indices = (positions_flat * self.trained_level_resolution).floor().long().clamp(0,
                                                                                                     self.trained_level_resolution - 1)
                if self.training:
                    if update_levels:
                        flat_indices = reso_indices[
                                           ..., 0] * self.trained_level_resolution * self.trained_level_resolution \
                                       + reso_indices[..., 1] * self.trained_level_resolution + reso_indices[..., 2]
                        torch_scatter.scatter_min(pixel_levels, flat_indices, out=self.min_trained_level.view(-1))
                        torch_scatter.scatter_max(pixel_levels, flat_indices, out=self.max_trained_level.view(-1))
                else:
                    min_levels = self.min_trained_level[
                        reso_indices[..., 0], reso_indices[..., 1], reso_indices[..., 2]]
                    max_levels = self.max_trained_level[
                        reso_indices[..., 0], reso_indices[..., 1], reso_indices[..., 2]]
                    pixel_levels = torch.maximum(min_levels, torch.minimum(pixel_levels, max_levels))

        if self.level_interpolation == LevelInterpolation.NONE:
            level_indices = get_levels(pixel_levels, self.num_scales)
            level_weights = {}
            for level, indices in level_indices.items():
                level_weights[level] = torch.ones_like(indices, dtype=pixel_levels.dtype)
        elif self.level_interpolation == LevelInterpolation.LINEAR:
            level_indices, level_weights = get_weights(pixel_levels, self.num_scales)
        else:
            raise Exception(self.level_interpolation)

        if self.share_feature_grid:
            encoding = self.get_shared_encoding(positions_flat)

        if self.output_interpolation == OutputInterpolation.COLOR:
            density = None
            level_embeddings = {}
        else:
            interpolated_h = None

        for level, cur_level_indices in level_indices.items():
            if self.share_feature_grid:
                level_encoding = encoding[cur_level_indices][..., :self.encoding_input_dims[level]]
            else:
                level_encoding = self.get_level_encoding(level, positions_flat[cur_level_indices])

            level_h = self.mlp_bases[level](level_encoding).to(positions)
            cur_level_weights = level_weights[level]
            if self.output_interpolation == OutputInterpolation.COLOR:
                density_before_activation, level_mlp_out = torch.split(level_h, [1, self.geo_feat_dim], dim=-1)
                level_embeddings[level] = level_mlp_out
                level_density = trunc_exp(density_before_activation - 1)
                if density is None:
                    density = torch.zeros(positions_flat.shape[0], *level_density.shape[1:],
                                          dtype=level_density.dtype, device=level_density.device)
                density[cur_level_indices] += cur_level_weights.unsqueeze(-1) * level_density
            elif self.output_interpolation == OutputInterpolation.EMBEDDING:
                if interpolated_h is None:
                    interpolated_h = torch.zeros(positions_flat.shape[0], *level_h.shape[1:],
                                                 dtype=level_h.dtype, device=level_h.device)

                interpolated_h[cur_level_indices] += cur_level_weights.unsqueeze(-1) * level_h
            else:
                raise Exception(self.output_interpolation)

        if self.output_interpolation == OutputInterpolation.COLOR:
            additional_info = (level_indices, level_weights, level_embeddings)
        elif self.output_interpolation == OutputInterpolation.EMBEDDING:
            density_before_activation, mlp_out = torch.split(interpolated_h, [1, self.geo_feat_dim], dim=-1)
            density = trunc_exp(density_before_activation - 1)
            additional_info = mlp_out
        else:
            raise Exception(self.output_interpolation)

        if self.training:
            level_counts = defaultdict(int)
            for level, indices in level_indices.items():
                level_counts[level] = indices.shape[0] / pixel_levels.shape[0]

            return density.view(ray_samples.frustums.starts.shape), (additional_info, level_counts)
        else:
            return density.view(ray_samples.frustums.starts.shape), (additional_info, pixel_levels.view(density.shape))

    def get_outputs(self, ray_samples: RaySamples, density_embedding: Tuple[Any] = None):
        directions = get_normalized_directions(ray_samples.frustums.directions)
        directions_flat = directions.view(-1, 3)
        d = self.direction_encoding(directions_flat)

        if self.appearance_embedding_dim > 0:
            if ray_samples.metadata is not None and TRAIN_INDEX in ray_samples.metadata:
                embedded_appearance = self.embedding_appearance(
                    ray_samples.metadata[TRAIN_INDEX].squeeze().to(d.device))
            elif self.training:
                embedded_appearance = self.embedding_appearance(ray_samples.camera_indices.squeeze())
            else:
                embedded_appearance = torch.ones(
                    (*directions.shape[:-1], self.appearance_embedding_dim), device=directions.device
                ) * self.embedding_appearance.mean(dim=0)

            embedded_appearance = embedded_appearance.view(-1, self.appearance_embedding_dim)

        outputs = {}

        if self.training:
            density_embedding, level_counts = density_embedding
            outputs[PixelNeRFFieldHeadNames.LEVEL_COUNTS] = level_counts
        else:
            density_embedding, levels = density_embedding
            outputs[PixelNeRFFieldHeadNames.LEVELS] = levels.view(ray_samples.frustums.starts.shape)

        if ray_samples.metadata is not None and EXPLICIT_LEVEL in ray_samples.metadata:
            level = ray_samples.metadata[EXPLICIT_LEVEL]
            if self.output_interpolation == OutputInterpolation.COLOR:
                _, _, level_embeddings = density_embedding
                density_embedding = level_embeddings[level]

                mlp_head = self.mlp_heads[level]
            else:
                mlp_head = self.mlp_head

            color_inputs = [d, density_embedding]
            if self.appearance_embedding_dim > 0:
                color_inputs.append(embedded_appearance)

            h = torch.cat(color_inputs, dim=-1)

            outputs[FieldHeadNames.RGB] = mlp_head(h).view(directions.shape).to(directions)
            return outputs

        if self.output_interpolation != OutputInterpolation.COLOR:
            color_inputs = [d, density_embedding]
            if self.appearance_embedding_dim > 0:
                color_inputs.append(embedded_appearance)
            h = torch.cat(color_inputs, dim=-1)
            rgbs = self.mlp_head(h).view(directions.shape).to(directions)
            outputs[FieldHeadNames.RGB] = rgbs

            return outputs

        level_indices, level_weights, level_embeddings = density_embedding

        rgbs = None
        for level, cur_level_indices in level_indices.items():
            color_inputs = [d[cur_level_indices], level_embeddings[level]]
            if self.appearance_embedding_dim > 0:
                color_inputs.append(embedded_appearance[cur_level_indices])
            h = torch.cat(color_inputs, dim=-1)

            level_rgbs = self.mlp_heads[level](h).to(directions)
            if rgbs is None:
                rgbs = torch.zeros_like(directions)
            rgbs.view(-1, 3)[cur_level_indices] += level_weights[level].unsqueeze(-1) * level_rgbs

        outputs[FieldHeadNames.RGB] = rgbs
        return outputs

    def density_fn(self, positions: Shaped[Tensor, "*bs 3"], times: Optional[Shaped[Tensor, "*bs 1"]] = None,
                   step_size: int = None, origins: Optional[Shaped[Tensor, "*bs 3"]] = None,
                   directions: Optional[Shaped[Tensor, "*bs 3"]] = None,
                   starts: Optional[Shaped[Tensor, "*bs 1"]] = None,
                   ends: Optional[Shaped[Tensor, "*bs 1"]] = None, pixel_area: Optional[Shaped[Tensor, "*bs 1"]] = None) \
            -> Shaped[Tensor, "*bs 1"]:
        """Returns only the density. Used primarily with the density grid.
        """
        if origins is None:
            camera_ids = torch.randint(0, len(self.cameras), (positions.shape[0],), device=positions.device)
            cameras = self.cameras.to(camera_ids.device)[camera_ids]
            origins = cameras.camera_to_worlds[:, :, 3]
            directions = positions - origins
            directions, _ = camera_utils.normalize_with_norm(directions, -1)
            coords = torch.cat(
                [torch.rand_like(origins[..., :1]) * cameras.height, torch.rand_like(origins[..., :1]) * cameras.width],
                -1).floor().long()

            pixel_area = cameras.generate_rays(torch.arange(len(cameras)).unsqueeze(-1), coords=coords).pixel_area
            starts = (origins - positions).norm(dim=-1, keepdim=True) - step_size / 2
            ends = starts + step_size

        ray_samples = RaySamples(
            frustums=Frustums(
                origins=origins,
                directions=directions,
                starts=starts,
                ends=ends,
                pixel_area=pixel_area,
            ),
            times=times
        )

        density, _ = self.get_density(ray_samples, update_levels=False)
        return density


@torch.compile
def repeat_interleave(input: torch.Tensor, repeats: int) -> torch.Tensor:
    """
    Repeat interleave along axis 0
    torch.repeat_interleave is currently very slow
    https://github.com/pytorch/pytorch/issues/31980
    """
    output = input.unsqueeze(1).expand(-1, repeats, *input.shape[1:])
    return output.reshape(-1, *input.shape[1:])


def parse_output_interpolation(model: str) -> OutputInterpolation:
    if model.casefold() == 'color':
        return OutputInterpolation.COLOR
    elif model.casefold() == 'embedding':
        return OutputInterpolation.EMBEDDING
    else:
        raise Exception(model)


def parse_level_interpolation(model: str) -> LevelInterpolation:
    if model.casefold() == 'none':
        return LevelInterpolation.NONE
    elif model.casefold() == 'linear':
        return LevelInterpolation.LINEAR
    else:
        raise Exception(model)
