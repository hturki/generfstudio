from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union, Type, Literal, Any, Optional

import math
import numpy as np
import torch
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.model_components.losses import MSELoss
from nerfstudio.model_components.ray_samplers import UniformSampler
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    RGBRenderer, )
from nerfstudio.model_components.scene_colliders import NearFarCollider
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps, misc
from nerfstudio.utils.rich_utils import CONSOLE
from pytorch_msssim import SSIM
from torch.nn import Parameter
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from generfstudio.fields.ibrnet_field import IBRNetField
from generfstudio.fields.image_encoder import ImageEncoder
from generfstudio.fields.pixelnerf_field import PixelNeRFField
from generfstudio.fields.spatial_encoder import SpatialEncoder
from generfstudio.fields.unet import UNet
from generfstudio.generfstudio_constants import NEAR, FAR, IMAGE_FEATURES, FEATURE_SCALING, GLOBAL_LATENT, \
    DEFAULT_SCENE_METADATA
from generfstudio.pixelnerf_utils.pdf_and_depth_sampler import PDFAndDepthSampler


@dataclass
class PixelNeRFModelConfig(ModelConfig):
    _target: Type = field(default_factory=lambda: PixelNeRFModel)

    num_coarse_samples: int = 64
    """Number of samples in coarse field evaluation"""
    num_importance_samples: int = 32
    """Number of samples in fine field evaluation"""
    num_depth_samples: int = 16

    background_color: Literal["random", "black", "white"] = "black"

    combine_layer: int = 3
    use_positional_encoding_xyz: bool = True
    use_view_dirs: bool = True
    normalize_z: bool = True
    use_global_encoder: bool = False
    freeze_encoder: bool = False
    """Freeze encoder weights during training"""

    image_encoder_type: Literal["unet", "resnet", "dino"] = "resnet"
    pixelnerf_type: Literal["ibrnet", "pixelnerf"] = "ibrnet"

    default_scene_view_count: int = 3


class PixelNeRFModel(Model):
    config: PixelNeRFModelConfig
    """
    PixelNeRF model.
    """

    def __init__(self, config: PixelNeRFModelConfig, metadata: Dict[str, Any], **kwargs) -> None:
        self.near = metadata.get(NEAR, None)
        self.far = metadata.get(FAR, None)
        if self.near is not None or self.far is not None:
            CONSOLE.log(f"Using near and far bounds {self.near} {self.far} from metadata")

        self.default_scene = metadata[DEFAULT_SCENE_METADATA]

        super().__init__(config=config, **kwargs)

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        near = self.near if self.near is not None else self.config.collider_params["near_plane"]
        far = self.far if self.far is not None else self.config.collider_params["far_plane"]
        self.collider = NearFarCollider(near_plane=near, far_plane=far)

        if self.config.image_encoder_type == "resnet":
            assert not self.config.freeze_encoder
            self.encoder = SpatialEncoder()
            encoder_dim = self.encoder.latent_size
        elif self.config.image_encoder_type == "unet":
            assert not self.config.freeze_encoder
            encoder_dim = 128
            self.encoder = UNet(in_channels=3, n_classes=encoder_dim)
        else:
            raise Exception(self.config.image_encoder_type)

        if self.config.pixelnerf_type == "ibrnet":
            assert not self.config.use_global_encoder
            self.field_coarse = IBRNetField(encoder_dim, 0)
            self.field_fine = IBRNetField(encoder_dim, 0)
        elif self.config.pixelnerf_type == "pixelnerf":
            if self.config.use_global_encoder:
                self.global_encoder = ImageEncoder()

            self.field_coarse = PixelNeRFField(
                encoder_latent_size=encoder_dim,
                global_encoder_latent_size=self.global_encoder.latent_size if self.config.use_global_encoder else 0,
                out_feature_dim=0,
                combine_layer=self.config.combine_layer,
                use_positional_encoding_xyz=self.config.use_positional_encoding_xyz,
                use_view_dirs=self.config.use_view_dirs,
                normalize_z=self.config.normalize_z)

            self.field_fine = PixelNeRFField(
                encoder_latent_size=encoder_dim,
                global_encoder_latent_size=self.global_encoder.latent_size if self.config.use_global_encoder else 0,
                out_feature_dim=0,
                combine_layer=self.config.combine_layer,
                use_positional_encoding_xyz=self.config.use_positional_encoding_xyz,
                use_view_dirs=self.config.use_view_dirs,
                normalize_z=self.config.normalize_z)
        else:
            raise Exception(self.config.pixelnerf_type)

        # samplers
        self.sampler_uniform = UniformSampler(num_samples=self.config.num_coarse_samples)
        self.sampler_pdf = PDFAndDepthSampler(num_samples=self.config.num_importance_samples,
                                              num_depth_samples=self.config.num_depth_samples)

        # renderers
        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer(method='expected')

        # losses
        self.rgb_loss = MSELoss()

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

        self.default_scene_metadata = {}
        self.default_scene_latents = None

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        param_groups["fields"] = list(self.field_coarse.parameters()) + list(self.field_fine.parameters())
        if not self.config.freeze_encoder:
            param_groups["fields"] += list(self.encoder.parameters())
            if self.config.use_global_encoder:
                param_groups["fields"] += list(self.global_encoder.parameters())

        return param_groups

    @torch.no_grad()
    def get_outputs_for_camera(self, camera: Cameras, obb_box: Optional[OrientedBox] = None) -> Dict[str, torch.Tensor]:
        """Takes in a camera, generates the raybundle, and computes the output of the model.
        Assumes a ray-based model.

        Args:
            camera: generates raybundle
        """
        if camera.metadata is not None and NEIGHBOR_CAMERAS in camera.metadata:
            ray_bundle_metadata = {}

            ray_bundle_metadata[IMAGE_FEATURES] = self._get_latents(camera.metadata[NEIGHBOR_IMAGES])
            del camera.metadata[NEIGHBOR_IMAGES]

            ray_bundle_metadata[NEIGHBOR_CAMERAS] = camera.metadata[NEIGHBOR_CAMERAS]
            del camera.metadata[NEIGHBOR_CAMERAS]
        else:
            if self.default_scene_latents is None:
                neighboring_views = np.linspace(0, len(self.default_scene.cameras) - 1,
                                                self.config.default_scene_view_count, dtype=np.int32)
                scene_dataset = InputDataset(self.default_scene)
                neighbor_images = [scene_dataset.get_image_float32(i).to(self.device) for i in neighboring_views]
                self.default_scene_latents = self._get_latents(neighbor_images)
                self.default_scene_metadata[NEIGHBOR_CAMERAS] = self.default_scene.cameras[
                    torch.LongTensor(neighboring_views)].to(self.device)

            ray_bundle_metadata = self.default_scene_metadata
            ray_bundle_metadata[IMAGE_FEATURES] = self.default_scene_latents

        ray_bundle = camera.generate_rays(camera_indices=0, keep_shape=True, obb_box=obb_box)
        return self.get_outputs_for_camera_ray_bundle(ray_bundle, ray_bundle_metadata)

    def _get_latents(self, neighbor_images: torch.Tensor) -> Tuple:
        latent = self.encoder(neighbor_images.view(-1, *neighbor_images.shape[2:]).permute(0, 3, 1, 2))
        if self.config.freeze_encoder:
            latent = latent.detach()

        latent_scaling = torch.FloatTensor([latent.shape[-1], latent.shape[-2]]).to(latent.device)
        latent_scaling = latent_scaling / (latent_scaling - 1) * 2.0

        if self.config.use_global_encoder:
            global_latent = self.global_encoder(neighbor_images)
            if self.config.freeze_encoder:
                global_latent = global_latent.detach()

            latents = (latent, latent_scaling, global_latent)
        else:
            latents = (latent, latent_scaling)
        return latents

    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(self, camera_ray_bundle: RayBundle,
                                          ray_bundle_metadata: Dict = to_immutable_dict({})) -> Dict[str, torch.Tensor]:
        """Takes in camera parameters and computes the output of the model.

        Args:
            camera_ray_bundle: ray bundle to calculate outputs over
        """

        input_device = camera_ray_bundle.directions.device
        num_rays_per_chunk = self.config.eval_num_rays_per_chunk
        image_height, image_width = camera_ray_bundle.origins.shape[:2]
        num_rays = len(camera_ray_bundle)
        outputs_lists = defaultdict(list)
        for i in range(0, num_rays, num_rays_per_chunk):
            start_idx = i
            end_idx = i + num_rays_per_chunk
            ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
            # move the chunk inputs to the model device
            ray_bundle = ray_bundle.to(self.device)
            ray_bundle.metadata.update(ray_bundle_metadata)
            outputs = self.forward(ray_bundle=ray_bundle)
            for output_name, output in outputs.items():  # type: ignore
                if not isinstance(output, torch.Tensor):
                    # TODO: handle lists of tensors as well
                    continue
                # move the chunk outputs from the model device back to the device of the inputs.
                outputs_lists[output_name].append(output.to(input_device))
        outputs = {}
        for output_name, outputs_list in outputs_lists.items():
            outputs[output_name] = torch.cat(outputs_list).view(image_height, image_width, -1)  # type: ignore
        return outputs

    def get_outputs(self, ray_bundle: RayBundle) -> Dict[str, Union[torch.Tensor, List]]:
        if IMAGE_FEATURES in ray_bundle.metadata:
            latents = ray_bundle.metadata[IMAGE_FEATURES]
            del ray_bundle.metadata[IMAGE_FEATURES]
        else:
            latents = self._get_latents(ray_bundle.metadata[NEIGHBOR_IMAGES])
            del ray_bundle.metadata[NEIGHBOR_IMAGES]

        if self.config.use_global_encoder:
            latent, latent_scaling, global_latent = latents
        else:
            latent, latent_scaling = latents

        neighbor_cameras = ray_bundle.metadata[NEIGHBOR_CAMERAS]
        del ray_bundle.metadata[NEIGHBOR_CAMERAS]

        ray_sample_metadata = {}
        ray_sample_metadata[NEIGHBOR_CAMERAS] = neighbor_cameras
        ray_sample_metadata[IMAGE_FEATURES] = latent

        # TODO: This assumes that all camera views have the same dimensions
        width = neighbor_cameras.width[0, 0].item()
        height = neighbor_cameras.height[0, 0].item()
        ray_sample_metadata[FEATURE_SCALING] = latent_scaling / torch.FloatTensor([width, height]).to(self.device)

        if self.config.use_global_encoder:
            ray_sample_metadata[GLOBAL_LATENT] = global_latent

        ray_samples_uniform = self.sampler_uniform(ray_bundle)
        ray_samples_uniform.metadata.update(ray_sample_metadata)

        outputs = {}

        field_outputs_coarse = self.field_coarse.forward(ray_samples_uniform)
        weights_coarse = ray_samples_uniform.get_weights(field_outputs_coarse[FieldHeadNames.DENSITY])
        rgb_coarse = self.renderer_rgb(
            rgb=field_outputs_coarse[FieldHeadNames.RGB],
            weights=weights_coarse,
        )
        outputs["rgb_coarse"] = rgb_coarse

        depth_coarse = self.renderer_depth(weights_coarse, ray_samples_uniform)
        outputs["depth_coarse"] = depth_coarse

        # pdf sampling
        ray_samples_pdf = self.sampler_pdf(ray_bundle, ray_samples_uniform, weights_coarse, depth_coarse)
        ray_samples_pdf.metadata.update(ray_sample_metadata)

        field_outputs_fine = self.field_fine.forward(ray_samples_pdf)

        weights_fine = ray_samples_pdf.get_weights(field_outputs_fine[FieldHeadNames.DENSITY])
        rgb_fine = self.renderer_rgb(
            rgb=field_outputs_fine[FieldHeadNames.RGB],
            weights=weights_fine,
        )

        outputs["rgb_fine"] = rgb_fine
        accumulation_coarse = self.renderer_accumulation(weights_coarse)
        outputs["accumulation_coarse"] = accumulation_coarse

        accumulation_fine = self.renderer_accumulation(weights_fine)
        outputs["accumulation_fine"] = accumulation_fine

        if not self.training:
            depth_fine = self.renderer_depth(weights_fine, ray_samples_pdf)
            outputs["depth_fine"] = depth_fine
            outputs["depth"] = depth_fine

        return outputs

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        # Scaling metrics by coefficients to create the losses.
        device = outputs["rgb_coarse"].device
        image = batch["image"].to(device)
        coarse_pred, coarse_image = self.renderer_rgb.blend_background_for_loss_computation(
            pred_image=outputs["rgb_coarse"],
            pred_accumulation=outputs["accumulation_coarse"],
            gt_image=image,
        )
        fine_pred, fine_image = self.renderer_rgb.blend_background_for_loss_computation(
            pred_image=outputs["rgb_fine"],
            pred_accumulation=outputs["accumulation_fine"],
            gt_image=image,
        )

        rgb_loss_coarse = self.rgb_loss(coarse_image, coarse_pred)
        rgb_loss_fine = self.rgb_loss(fine_image, fine_pred)

        loss_dict = {"rgb_loss_coarse": rgb_loss_coarse, "rgb_loss_fine": rgb_loss_fine}
        loss_dict = misc.scale_dict(loss_dict, self.config.loss_coefficients)

        for key, val in loss_dict.items():
            if not math.isfinite(val):
                raise Exception(f"Loss is not finite: {loss_dict}")

        return loss_dict

    def get_image_metrics_and_images(
            self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        image = batch["image"].to(outputs["rgb_coarse"].device)
        image = self.renderer_rgb.blend_background(image)
        rgb_coarse = outputs["rgb_coarse"]
        rgb_fine = outputs["rgb_fine"]
        acc_coarse = colormaps.apply_colormap(outputs["accumulation_coarse"])
        acc_fine = colormaps.apply_colormap(outputs["accumulation_fine"])
        assert self.config.collider_params is not None
        depth_coarse = colormaps.apply_depth_colormap(
            outputs["depth_coarse"],
            accumulation=outputs["accumulation_coarse"],
        )
        depth_fine = colormaps.apply_depth_colormap(
            outputs["depth_fine"],
            accumulation=outputs["accumulation_fine"],
        )

        combined_rgb = torch.cat([image, rgb_coarse, rgb_fine], dim=1)
        combined_acc = torch.cat([acc_coarse, acc_fine], dim=1)
        combined_depth = torch.cat([depth_coarse, depth_fine], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb_coarse = torch.moveaxis(rgb_coarse, -1, 0)[None, ...]
        rgb_fine = torch.moveaxis(rgb_fine, -1, 0)[None, ...]

        coarse_psnr = self.psnr(image, rgb_coarse)
        fine_psnr = self.psnr(image, rgb_fine)
        fine_ssim = self.ssim(image, rgb_fine)
        fine_lpips = self.lpips(image, rgb_fine)
        assert isinstance(fine_ssim, torch.Tensor)

        metrics_dict = {
            "psnr": float(fine_psnr.item()),
            "coarse_psnr": float(coarse_psnr),
            "fine_psnr": float(fine_psnr),
            "fine_ssim": float(fine_ssim),
            "fine_lpips": float(fine_lpips),
        }
        images_dict = {"img": combined_rgb, "accumulation": combined_acc, "depth": combined_depth}
        return metrics_dict, images_dict
