from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, Union

import math
import numpy as np
import torch
import torch.nn.functional as F
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.model_components.losses import MSELoss
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    RGBRenderer, SemanticRenderer,
)
from nerfstudio.model_components.scene_colliders import NearFarCollider, AABBBoxCollider
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps
from nerfstudio.utils.colormaps import ColormapOptions
from pytorch_msssim import SSIM
from rich.console import Console
from torch.nn import Parameter
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal

from pynerf.fields.pynerf_base_field import parse_output_interpolation, parse_level_interpolation
from pynerf.fields.pynerf_field import PixelNeRFField
from pynerf.pynerf_constants import LEVEL_COUNTS, LEVELS, WEIGHT, RENDER_LEVELS

CONSOLE = Console(width=120)


@dataclass
class PixelNeRFBaseModelConfig(ModelConfig):

    freeze_encoder: bool = False
    """Freeze encoder weights during training"""

    # near_plane: float = 0.2
    # """How far along the ray to start sampling."""
    # far_plane: float = 1000.0
    # """How far along the ray to stop sampling."""
    # geo_feat_dim: int = 15
    # """output geo feat dimensions"""
    # num_layers: int = 2
    # """Number of layers in the base mlp"""
    # num_layers_color: int = 3
    # """Number of layers in the color mlp"""
    # hidden_dim: int = 64
    # """Dimension of hidden layers"""
    # hidden_dim_color: int = 128
    # """Dimension of hidden layers for color network"""
    # num_levels: int = 16
    # """Number of levels of the hashmap for the base mlp."""
    # base_resolution: int = 16
    # """Base resolution of the hashmap for the base mlp."""
    # max_resolution: int = 2048
    # """Maximum resolution of the hashmap for the base mlp."""
    # log2_hashmap_size: int = 20
    # """Size of the hashmap for the base mlp"""
    # features_per_level: int = 4
    # """Number of features per resolution level"""
    # appearance_embedding_dim: int = 32
    # """Whether to use average appearance embedding or zeros for inference."""
    # disable_scene_contraction: bool = False
    # """Whether to disable scene contraction or not."""
    # background_color: Literal['random', 'last_sample', 'black', 'white'] = 'last_sample'
    # """Whether to randomize the background color."""
    # output_interpolation: Literal['color', 'embedding'] = 'embedding'
    # """Whether to interpolate between RGB outputs or density network embeddings."""
    # level_interpolation: Literal['none', 'linear'] = 'linear'
    # """How to interpolate between PixelNeRF levels."""
    # num_scales: int = 8
    # """Number of levels in the PixelNeRF hierarchy."""
    # scale_factor: float = 2
    # """Scale factor between levels in the PixelNeRF hierarchy."""
    # share_feature_grid: bool = True
    # """Whether to share the same feature grid between levels."""

class PixelNeRFBaseModel(Model):
    config: PixelNeRFBaseModelConfig
    """
    PixelNeRF base model.
    """

    def __init__(self, config: PixelNeRFBaseModelConfig, metadata: Dict[str, Any], **kwargs) -> None:
        self.near = metadata.get('near', None)
        self.far = metadata.get('far', None)
        self.pose_scale_factor = metadata.get('pose_scale_factor', 1)
        if self.near is not None or self.far is not None:
            CONSOLE.log(
                f'Using near and far bounds {self.near} {self.far} from metadata')

        self.cameras = metadata.get('cameras', None)

        super().__init__(config=config, **kwargs)

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        near = self.near if self.near is not None else (self.config.near_plane / self.pose_scale_factor)
        far = self.far if self.far is not None else (self.config.far_plane / self.pose_scale_factor)

        if self.config.disable_scene_contraction:
            self.scene_contraction = None
            self.collider = AABBBoxCollider(self.scene_box, near_plane=near)
        else:
            self.scene_contraction = SceneContraction(order=float('inf'))
            # Collider
            self.collider = NearFarCollider(near_plane=near, far_plane=far)

        self.field = PixelNeRFField(
            self.scene_box.aabb,
            num_images=self.num_train_data,
            num_layers=self.config.num_layers,
            geo_feat_dim=self.config.geo_feat_dim,
            num_layers_color=self.config.num_layers_color,
            hidden_dim_color=self.config.hidden_dim_color,
            appearance_embedding_dim=self.config.appearance_embedding_dim,
            base_resolution=self.config.base_resolution,
            max_resolution=self.config.max_resolution,
            features_per_level=self.config.features_per_level,
            num_levels=self.config.num_levels,
            log2_hashmap_size=self.config.log2_hashmap_size,
            spatial_distortion=self.scene_contraction,
            output_interpolation=parse_output_interpolation(self.config.output_interpolation),
            level_interpolation=parse_level_interpolation(self.config.level_interpolation),
            num_scales=self.config.num_scales,
            scale_factor=self.config.scale_factor,
            share_feature_grid=self.config.share_feature_grid,
            cameras=self.cameras,
        )

        # renderers
        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer(method='expected')

        # losses
        self.rgb_loss = MSELoss(reduction='none')

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        mlps = []
        fields = []
        field_children = [self.field.named_children()]

        for children in field_children:
            for name, child in children:
                if 'mlp' in name:
                    mlps += child.parameters()
                else:
                    fields += child.parameters()

        param_groups = {
            'mlps': mlps,
            'fields': fields,
        }

        return param_groups

    def get_outputs(self, camera: Cameras) -> Dict[str, Union[torch.Tensor, List]]:
        assert isinstance(camera, Cameras)

        camera_downscale = self._get_downscale_factor()
        camera.rescale_output_resolution(1 / camera_downscale)
        # shift the camera to center of scene looking at center
        R = camera.camera_to_worlds[0, :3, :3]  # 3 x 3
        T = camera.camera_to_worlds[0, :3, 3:4]  # 3 x 1
        # flip the z and y axes to align with gsplat conventions
        R_edit = torch.diag(torch.tensor([1, -1, -1], device=self.device, dtype=R.dtype))
        R = R @ R_edit
        # analytic matrix inverse to get world2camera matrix
        R_inv = R.T
        T_inv = -R_inv @ T
        viewmat = torch.eye(4, device=R.device, dtype=R.dtype)
        viewmat[:3, :3] = R_inv
        viewmat[:3, 3:4] = T_inv
        # calculate the FOV of the camera given fx and fy, width and height
        cx = camera.cx.item()
        cy = camera.cy.item()
        fovx = 2 * math.atan(camera.width / (2 * camera.fx))
        fovy = 2 * math.atan(camera.height / (2 * camera.fy))
        W, H = camera.width.item(), camera.height.item()
        self.last_size = (H, W)
        projmat = projection_matrix(0.001, 1000, fovx, fovy, device=self.device)
        BLOCK_X, BLOCK_Y = 16, 16
        tile_bounds = (
            (W + BLOCK_X - 1) // BLOCK_X,
            (H + BLOCK_Y - 1) // BLOCK_Y,
            1,
        )

        if crop_ids is not None:
            opacities_crop = self.opacities[crop_ids]
            means_crop = self.means[crop_ids]
            features_dc_crop = self.features_dc[crop_ids]
            features_rest_crop = self.features_rest[crop_ids]
            scales_crop = self.scales[crop_ids]
            quats_crop = self.quats[crop_ids]
        else:
            opacities_crop = self.opacities
            means_crop = self.means
            features_dc_crop = self.features_dc
            features_rest_crop = self.features_rest
            scales_crop = self.scales
            quats_crop = self.quats

        colors_crop = torch.cat((features_dc_crop[:, None, :], features_rest_crop), dim=1)

        self.xys, depths, self.radii, conics, num_tiles_hit, cov3d = ProjectGaussians.apply(
            means_crop,
            torch.exp(scales_crop),
            1,
            quats_crop / quats_crop.norm(dim=-1, keepdim=True),
            viewmat.squeeze()[:3, :],
            projmat.squeeze() @ viewmat.squeeze(),
            camera.fx.item(),
            camera.fy.item(),
            cx,
            cy,
            H,
            W,
            tile_bounds,
        )  # type: ignore
        if (self.radii).sum() == 0:
            return {"rgb": background.repeat(int(camera.height.item()), int(camera.width.item()), 1)}

        # Important to allow xys grads to populate properly
        if self.training:
            self.xys.retain_grad()

        if self.config.sh_degree > 0:
            viewdirs = means_crop.detach() - camera.camera_to_worlds.detach()[..., :3, 3]  # (N, 3)
            viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
            n = min(self.step // self.config.sh_degree_interval, self.config.sh_degree)
            rgbs = SphericalHarmonics.apply(n, viewdirs, colors_crop)
            rgbs = torch.clamp(rgbs + 0.5, min=0.0)  # type: ignore
        else:
            rgbs = torch.sigmoid(colors_crop[:, 0, :])

        # rescale the camera back to original dimensions
        camera.rescale_output_resolution(camera_downscale)

        # avoid empty rasterization
        num_intersects, _ = compute_cumulative_intersects(self.xys.size(0), num_tiles_hit)
        assert num_intersects > 0

        rgb = RasterizeGaussians.apply(
            self.xys,
            depths,
            self.radii,
            conics,
            num_tiles_hit,
            rgbs,
            torch.sigmoid(opacities_crop),
            H,
            W,
            background,
        )  # type: ignore
        rgb = torch.clamp(rgb, max=1.0)  # type: ignore
        depth_im = None
        if not self.training:
            depth_im = RasterizeGaussians.apply(  # type: ignore
                self.xys,
                depths,
                self.radii,
                conics,
                num_tiles_hit,
                depths[:, None].repeat(1, 3),
                torch.sigmoid(opacities_crop),
                H,
                W,
                torch.ones(3, device=self.device) * 10,
            )[..., 0:1]  # type: ignore

        return {"rgb": rgb, "depth": depth_im}  # type: ignore

    # def get_outputs(self, ray_bundle: RayBundle):
    #     outputs = self.get_outputs_inner(ray_bundle, None)
    #     if ray_bundle.metadata is not None and ray_bundle.metadata.get(RENDER_LEVELS, False):
    #         for i in range(self.field.min_trained_level, self.field.max_trained_level):
    #             level_outputs = self.get_outputs_inner(ray_bundle, i)
    #             outputs[f'rgb_level_{i}'] = level_outputs['rgb']
    #             outputs[f'depth_level_{i}'] = level_outputs['depth']
    #
    #     outputs['directions_norm'] = ray_bundle.metadata['directions_norm']
    #     return outputs

    @abstractmethod
    def get_outputs_inner(self, ray_bundle: RayBundle, explicit_level: Optional[int] = None):
        pass

    def get_metrics_dict(self, outputs: Dict[str, any], batch: Dict[str, any]) -> Dict[str, torch.Tensor]:
        metrics_dict = {}
        image = batch['image'].to(self.device)
        metrics_dict['psnr'] = self.psnr(outputs['rgb'], image)

        if 'depth_image' in batch:
            metrics_dict['depth'] = F.mse_loss(outputs['depth'], batch['depth_image'] * outputs['directions_norm'])

        if self.training:
            for key, val in outputs[LEVEL_COUNTS].items():
                metrics_dict[f'{LEVEL_COUNTS}_{key}'] = val

        return metrics_dict

    def get_loss_dict(self, outputs: Dict[str, any], batch: Dict[str, any],
                      metrics_dict: Optional[str, torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        loss_dict = self.get_loss_dict_inner(outputs, batch, metrics_dict)

        if self.training:
            for key, val in loss_dict.items():
                assert math.isfinite(val), f'Loss is not finite: {loss_dict}'

        return loss_dict

    def get_loss_dict_inner(self, outputs: Dict[str, any], batch: Dict[str, any],
                            metrics_dict: Optional[str, torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        image = batch['image'].to(self.device)
        rgb_loss = self.rgb_loss(image, outputs['rgb'])

        if WEIGHT in batch:
            weights = batch[WEIGHT].to(self.device).view(-1, 1)
            rgb_loss *= weights

        loss_dict = {'rgb_loss': rgb_loss.mean()}

        return loss_dict

    def get_image_metrics_and_images(
            self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        image = batch['image'].to(self.device)
        rgb = outputs['rgb']

        combined_rgb = torch.cat([image, rgb], dim=1)

        images_dict = {
            'img': combined_rgb,
        }

        acc = colormaps.apply_colormap(outputs['accumulation'])
        images_dict['accumulation'] = acc

        depth = colormaps.apply_depth_colormap(
            outputs['depth'],
            accumulation=outputs['accumulation'],
        )

        depth_vis = []
        if 'depth_image' in batch:
            depth_vis.append(colormaps.apply_depth_colormap(
                batch['depth_image'] * outputs['directions_norm'],
            ))

        depth_vis.append(depth)
        combined_depth = torch.cat(depth_vis, dim=1)

        images_dict['depth'] = combined_depth

        if not self.training:
            images_dict[LEVELS] = colormaps.apply_colormap(outputs[LEVELS] / self.config.num_levels,
                                                           colormap_options=ColormapOptions(colormap='turbo'))

        for i in range(self.config.num_levels):
            if f'rgb_level_{i}' in outputs:
                images_dict[f'rgb_level_{i}'] = torch.cat([image, outputs[f'rgb_level_{i}']], dim=1)
                images_dict[f'depth_level_{i}'] = colormaps.apply_depth_colormap(
                    outputs[f'depth_level_{i}'],
                    accumulation=outputs['accumulation'],
                )

        if 'mask' in batch:
            mask = batch['mask']
            assert torch.all(mask[:, mask.sum(dim=0) > 0])
            image = image[:, mask.sum(dim=0).squeeze() > 0]
            rgb = rgb[:, mask.sum(dim=0).squeeze() > 0]

        ssim = self.ssim(image, rgb)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

        psnr = self.psnr(image, rgb)
        lpips = self.lpips(image, rgb)
        mse = np.exp(-0.1 * np.log(10.) * float(psnr.item()))
        dssim = np.sqrt((1 - float(ssim)) / 2)
        avg_error = np.exp(np.mean(np.log(np.array([mse, dssim, float(lpips)]))))

        # all of these metrics will be logged as scalars
        metrics_dict = {
            'psnr': float(psnr.item()),
            'ssim': float(ssim),
            'lpips': float(lpips),
            'avg_error': avg_error
        }  # type: ignore

        if WEIGHT in batch:
            weight = int(torch.unique(batch[WEIGHT]).item())
            for key, val in set(metrics_dict.items()):
                metrics_dict[f'{key}_{weight}'] = val
            for key, val in set(images_dict.items()):
                if 'level' not in key:
                    images_dict[f'{key}_{weight}'] = val

        return metrics_dict, images_dict
