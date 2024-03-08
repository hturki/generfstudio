from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union, Type, Optional, Literal

import math
import torch
import torch.nn.functional as F
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.model_components.ray_samplers import UniformLinDispPiecewiseSampler
from nerfstudio.model_components.renderers import RGBRenderer, DepthRenderer, SemanticRenderer, AccumulationRenderer
from nerfstudio.model_components.scene_colliders import NearFarCollider
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps
from omegaconf import OmegaConf
from pytorch_msssim import SSIM
from rich.console import Console
from torch import nn
from torch.nn import Parameter, MSELoss
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from extern.ldm_zero123.models.diffusion.ddim import DDIMSampler
from extern.ldm_zero123.util import instantiate_from_config
from generfstudio.fields.dino_v2_encoder import DinoV2Encoder
from generfstudio.fields.ibrnet_field import IBRNetInnerField
from generfstudio.fields.pixelnerf_field import PixelNeRFInnerField
from generfstudio.fields.spatial_encoder import SpatialEncoder
from generfstudio.fields.unet import UNet
from generfstudio.generfstudio_constants import NEIGHBORING_VIEW_IMAGES, NEIGHBORING_VIEW_CAMERAS

CONSOLE = Console(width=120)


@dataclass
class MVDiffusionConfig(ModelConfig):
    _target: Type = field(default_factory=lambda: MVDiffusion)

    # diffusion_checkpoint_path: str = "/data/hturki/zero123-105000.ckpt"  # "/data/hturki/sd-image-conditioned-v2.ckpt"
    diffusion_checkpoint_path: str = "/data/hturki/sd-image-conditioned-v2.ckpt"
    # diffusion_checkpoint_path: str = "/data/hturki/threestudio/load/zero123/stable_zero123.ckpt"
    diffusion_config: str = "config/mv-diffusion.yaml"
    guidance_scale: float = 3.0

    uncond: float = 0.05
    ddim_steps = 50

    default_scene_view_count: int = 3

    num_ray_samples: int = 128

    background_color: Literal["random", "last_sample", "black", "white"] = "last_sample"

    concat_crossattn_with_null_prompt: bool = False

    cond_image_encoder_type: Literal["unet", "resnet", "dino"] = "resnet"
    pixelnerf_type: Literal["ibrnet", "pixelnerf"] = "pixelnerf"

    cond_only: bool = False

    freeze_cond_image_encoder: bool = False


def cartesian_to_spherical(xyz):
    xy = xyz[:, 0] ** 2 + xyz[:, 1] ** 2
    z = torch.sqrt(xy + xyz[:, 2] ** 2)
    theta = torch.arctan2(torch.sqrt(xy), xyz[:, 2])  # for elevation angle defined from Z-axis down
    azimuth = torch.arctan2(xyz[:, 1], xyz[:, 0])
    return theta, azimuth, z


def get_T(T_target, T_cond):
    theta_cond, azimuth_cond, z_cond = cartesian_to_spherical(T_cond)
    theta_target, azimuth_target, z_target = cartesian_to_spherical(T_target)

    d_theta = theta_target - theta_cond
    d_azimuth = (azimuth_target - azimuth_cond) % (2 * math.pi)
    d_z = z_target - z_cond

    d_T = torch.stack([d_theta, torch.sin(d_azimuth), torch.cos(d_azimuth), d_z], dim=-1)
    return d_T


class MVDiffusion(Model):
    config: MVDiffusionConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()
        self.collider = None

        config = OmegaConf.load(self.config.diffusion_config)

        # Need to prefix with underscore or else the nerfstudio viewer will throw an exception
        self._model = instantiate_from_config(config.model)
        self._model.cc_projection = None

        if self.config.concat_crossattn_with_null_prompt:
            self.cc_projection = nn.Linear(768 * 2, 768)
            nn.init.eye_(list(self.cc_projection.parameters())[0][:768, :768])
            nn.init.zeros_(list(self.cc_projection.parameters())[1])

        old_state = torch.load(self.config.diffusion_checkpoint_path, map_location="cpu")
        if "state_dict" in old_state:
            CONSOLE.log("Found nested key 'state_dict' in checkpoint, loading this instead")
            old_state = old_state["state_dict"]

            # Check if we need to port weights from 4ch input to 8ch
            in_filters_load = old_state["model.diffusion_model.input_blocks.0.0.weight"]
            new_state = self._model.state_dict()
            in_filters_current = new_state["model.diffusion_model.input_blocks.0.0.weight"]
            in_shape = in_filters_current.shape
            if in_shape != in_filters_load.shape:
                input_keys = [
                    "model.diffusion_model.input_blocks.0.0.weight",
                    "model_ema.diffusion_modelinput_blocks00weight",
                ]

                for input_key in input_keys:
                    if input_key not in old_state or input_key not in new_state:
                        continue
                    input_weight = new_state[input_key]
                    if input_weight.size() != old_state[input_key].size():
                        CONSOLE.log(f"Manual init: {input_key}")
                        input_weight.zero_()
                        input_weight[:, :old_state[input_key].shape[1], :, :].copy_(old_state[input_key])
                        old_state[input_key] = torch.nn.parameter.Parameter(input_weight)

            m, u = self._model.load_state_dict(old_state, strict=False)

            # if len(m) > 0:
            #     CONSOLE.log("missing keys: \n", m)
            # if len(u) > 0:
            #     CONSOLE.log("unexpected keys: \n", u)

        if self.config.cond_image_encoder_type == "resnet":
            self.cond_image_encoder = SpatialEncoder()
            encoder_dim = self.cond_image_encoder.latent_size
        elif self.config.cond_image_encoder_type == "unet":
            assert not self.config.freeze_cond_image_encoder
            encoder_dim = 128
            self.cond_image_encoder = UNet(in_channels=3, n_classes=encoder_dim)
        elif self.config.cond_image_encoder_type == "dino":
            assert self.config.freeze_cond_image_encoder
            self.cond_image_encoder = DinoV2Encoder()
            encoder_dim = self.cond_image_encoder.out_feature_dim
        else:
            raise Exception(self.config.cond_image_encoder_type)

        if self.config.pixelnerf_type == "ibrnet":
            self.cond_feature_field = IBRNetInnerField(encoder_dim, 128)
        elif self.config.pixelnerf_type == "pixelnerf":
            self.cond_feature_field = PixelNeRFInnerField(
                encoder_latent_size=encoder_dim,
                global_encoder_latent_size=0,
                out_feature_dim=128)
        else:
            raise Exception(self.config.pixelnerf_type)

        self.ray_collider = NearFarCollider(0.1, 5)  # NearFarCollider(0.5, 1000)
        self.ray_sampler = UniformLinDispPiecewiseSampler(num_samples=self.config.num_ray_samples)
        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        self.renderer_depth = DepthRenderer(method="expected")
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_features = SemanticRenderer()

        self.ddim_sampler = DDIMSampler(self._model)

        # metrics
        self.rgb_loss = MSELoss()
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        for p in self.lpips.parameters():
            p.requires_grad_(False)

        self.register_buffer("null_prompt", self._model.get_learned_conditioning([""]).detach(), persistent=False)

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        param_groups["cond_encoder"] = list(self.cond_feature_field.parameters())
        if not self.config.freeze_cond_image_encoder:
            param_groups["cond_encoder"] += list(self.cond_image_encoder.parameters())
        param_groups["fields"] = list(self._model.parameters())

        return param_groups

    def get_training_callbacks(
            self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        def set_ema(step: int):
            if self._model.use_ema:
                self._model.model_ema(self._model.model)

        return [
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                func=set_ema,
                update_every_num_iters=1,
            ),
        ]

    def get_cond_features(self, cameras: Cameras, image_cond: torch.Tensor):
        neighboring_view_count = image_cond.shape[1]
        cond_features = self.cond_image_encoder(image_cond.view(-1, *image_cond.shape[2:]))
        if self.config.freeze_cond_image_encoder:
            cond_features = cond_features.detach()

        feature_scaling = torch.FloatTensor([cond_features.shape[-1], cond_features.shape[-2]]).to(cond_features.device)
        feature_scaling = 2.0 / feature_scaling / (feature_scaling - 1)
        uv_scaling = feature_scaling / torch.FloatTensor([image_cond.shape[-1], image_cond.shape[-2]]).to(
            feature_scaling)

        target_cameras = deepcopy(cameras)
        target_cameras.metadata = {}
        target_cameras.rescale_output_resolution(self._model.image_size / cameras.height[0, 0].item())

        neighboring_view_cameras = cameras.metadata[NEIGHBORING_VIEW_CAMERAS]
        cam_cond = neighboring_view_cameras.camera_to_worlds.view(-1, 3, 4)
        rot = cam_cond[:, :3, :3].transpose(1, 2)  # (B, 3, 3)
        trans = -torch.bmm(rot, cam_cond[:, :3, 3:])  # (B, 3, 1)
        cam_cond_w2c = torch.cat((rot, trans), dim=-1)

        rgbs = []
        features = []
        if not self.training:
            depths = []
            accumulations = []

        # cameras_per_chunk = self.config.eval_num_rays_per_chunk // (self._model.image_size * self._model.image_size)
        cameras_per_chunk = 8
        for start_index in range(0, len(target_cameras), cameras_per_chunk):
            end_index = min(start_index + cameras_per_chunk, len(target_cameras))
            chunk_indices = torch.arange(start_index, end_index).view(-1, 1)
            ray_bundle = target_cameras.generate_rays(camera_indices=chunk_indices)
            ray_bundle = self.ray_collider(ray_bundle)
            ray_samples = self.ray_sampler(ray_bundle.flatten())

            # torch.Size([32, 32, 4, 128, 3]) for 4 cams
            positions = ray_samples.frustums.get_positions().view(*ray_bundle.shape + (self.config.num_ray_samples, 3))
            positions = positions.permute(2, 0, 1, 3, 4).reshape(len(chunk_indices), -1, 3)

            sample_density, sample_rgb, sample_features = self.cond_feature_field(
                ray_samples, positions,
                cond_features[start_index * neighboring_view_count:end_index * neighboring_view_count],
                cam_cond_w2c[start_index * neighboring_view_count:end_index * neighboring_view_count],
                neighboring_view_cameras[start_index:end_index], uv_scaling)

            weights = ray_samples.get_weights(sample_density)
            rgbs.append(self.renderer_rgb(rgb=sample_rgb, weights=weights))
            features.append(self.renderer_features(semantics=sample_features, weights=weights))

            if not self.training:
                depths.append(self.renderer_depth(weights=weights, ray_samples=ray_samples))
                accumulations.append(self.renderer_accumulation(weights=weights))

        rgbs = torch.cat(rgbs).view(image_cond.shape[0], self._model.image_size, self._model.image_size, 3)
        features = torch.cat(features).view(image_cond.shape[0], self._model.image_size, self._model.image_size, -1)

        if not self.training:
            depths = torch.cat(depths).view(image_cond.shape[0], self._model.image_size, self._model.image_size, 1)
            accumulations = torch.cat(accumulations).view(image_cond.shape[0], self._model.image_size,
                                                          self._model.image_size, 1)
            return rgbs, features, depths, accumulations

        return rgbs, features

    def get_outputs(self, cameras: Cameras) -> Dict[str, Union[torch.Tensor, List]]:
        cam_cond = cameras.metadata[NEIGHBORING_VIEW_CAMERAS].camera_to_worlds
        neighboring_view_count = cam_cond.shape[1]
        # cam_cond = cam_cond.view(-1, cam_cond.shape[2], cam_cond.shape[3])
        # T = get_T(repeat_interleave(cameras.camera_to_worlds[..., 3], neighboring_view_count), cam_cond[..., 3])

        # Map images from [0, 1] to [-1, 1]

        image_cond = cameras.metadata[NEIGHBORING_VIEW_IMAGES].permute(0, 1, 4, 2, 3)  # * 2 - 1
        cond_features = self.get_cond_features(cameras, image_cond)

        # Move to [-1, 1] for CLIP
        image_cond = image_cond * 2 - 1

        outputs = {}
        if not self.training:
            rgbs, features, depths, accumulations = cond_features
            outputs["depth"] = depths.view(cam_cond.shape[0], self._model.image_size, self._model.image_size, 1)
            outputs["accumulation"] = accumulations.view(cam_cond.shape[0], self._model.image_size,
                                                         self._model.image_size, 1)
        else:
            rgbs, features = cond_features

        outputs["rgb"] = rgbs

        if self.config.cond_only:
            return outputs

        # Get CLIP embeddings for cross attention
        c_crossattn = self._model.get_learned_conditioning(image_cond.view(-1, *image_cond.shape[2:])).detach()
        c_crossattn = c_crossattn.view(-1, neighboring_view_count, c_crossattn.shape[-1])

        c_concat = torch.cat([rgbs.permute(0, 3, 1, 2), features.permute(0, 3, 1, 2)], 1)
        # c_concat = self._model.encode_first_stage(image_cond[:, 0].to(self.device)).mode().detach()

        if self.training:
            # To support classifier-free guidance, randomly drop out only text conditioning 5%, only image conditioning 5%, and both 5%.
            # null_prompt = self._model.get_learned_conditioning([""]).detach()
            random_mask = torch.rand((image_cond.shape[0], 1, 1), device=image_cond.device)
            prompt_mask = random_mask < 2 * self.config.uncond
            input_mask = torch.logical_not(
                torch.logical_and(random_mask >= self.config.uncond,
                                  random_mask < 3 * self.config.uncond)).float().unsqueeze(-1)

            c_crossattn = torch.where(prompt_mask, self.null_prompt, c_crossattn)
            c_concat = input_mask * c_concat

        if self.config.concat_crossattn_with_null_prompt:
            c_crossattn = self.cc_projection(
                torch.cat([self.null_prompt.expand(c_crossattn.shape[0], c_crossattn.shape[1], -1), c_crossattn],
                          dim=-1))

        if not self.training:
            with self._model.ema_scope():
                n_samples = 4
                cond = {}
                cond["c_crossattn"] = [c_crossattn.expand(n_samples, -1, -1)]
                cond["c_concat"] = [c_concat.expand(n_samples, -1, -1, -1)]
                samples_ddim, _ = self.ddim_sampler.sample(S=self.config.ddim_steps,
                                                           conditioning=cond,
                                                           batch_size=n_samples,
                                                           shape=(4,) + c_concat.shape[2:],
                                                           verbose=False,
                                                           unconditional_guidance_scale=self.config.guidance_scale,
                                                           unconditional_conditioning=None,
                                                           eta=1,
                                                           x_T=None)

                x_samples_ddim = (self._model.decode_first_stage(samples_ddim).clamp(-1, 1) + 1) / 2
                outputs["samples"] = x_samples_ddim.permute(0, 2, 3, 1)
                return outputs

        outputs["c_crossattn"] = [c_crossattn]
        outputs["c_concat"] = [c_concat]

        return outputs

    @torch.no_grad()
    def get_outputs_for_camera(self, camera: Cameras, obb_box: Optional[OrientedBox] = None) -> Dict[str, torch.Tensor]:
        return self.get_outputs(camera)

    def get_metrics_dict(self, outputs, batch):
        assert self.training

        image = batch["image"].to(self.device)
        cond_rgb = outputs["rgb"]
        downsampled_image = F.interpolate(image.permute(0, 3, 1, 2), cond_rgb.shape[1:3]).permute(0, 2, 3, 1)
        metrics_dict = {
            "loss_cond_image": self.rgb_loss(downsampled_image, cond_rgb),
            "psnr_cond_image": self.psnr(downsampled_image, cond_rgb)
        }

        if self.config.cond_only:
            return metrics_dict

        encoder_posterior = self._model.encode_first_stage(image.permute(0, 3, 1, 2) * 2 - 1)
        z = self._model.get_first_stage_encoding(encoder_posterior).detach()
        t = torch.randint(0, self._model.num_timesteps, (z.shape[0],), device=self.device).long()
        noise = torch.randn_like(z)
        x_noisy = self._model.q_sample(x_start=z, t=t, noise=noise)

        cond = {"c_crossattn": outputs["c_crossattn"], "c_concat": outputs["c_concat"]}
        model_output = self._model.apply_model(x_noisy, t, cond)

        if self._model.parameterization == "x0":
            target = z
        elif self._model.parameterization == "eps":
            target = noise
        else:
            raise NotImplementedError()

        loss_simple = self._model.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3))
        metrics_dict["loss_simple"] = loss_simple.mean()

        logvar_t = self._model.logvar.to(self.device)[t]
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        metrics_dict["loss_base"] = loss.mean()
        if self._model.learn_logvar:
            metrics_dict["logvar"] = self._model.logvar.data.mean()

        if self._model.original_elbo_weight > 0:
            loss_vlb = self._model.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3))
            loss_vlb = (self._model.lvlb_weights[t] * loss_vlb).mean()
            metrics_dict["loss_vlb"] = loss_vlb

        return metrics_dict

    def get_loss_dict(self, cond, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        loss_dict = {
            "loss_cond_image": metrics_dict["loss_cond_image"],
        }

        if self.config.cond_only:
            return loss_dict

        loss_dict["loss_base"] = self._model.l_simple_weight * metrics_dict["loss_base"]

        if "loss_vlb" in metrics_dict:
            loss_dict["loss_vlb"] = self.original_elbo_weight * metrics_dict["loss_vlb"]

        return loss_dict

    def get_image_metrics_and_images(
            self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        image = batch["image"].to(outputs["rgb"].device)
        cond_rgb = outputs["rgb"]
        downsampled_image = F.interpolate(image.unsqueeze(0).permute(0, 3, 1, 2), cond_rgb.shape[1:3]).permute(0, 2, 3,
                                                                                                               1)
        metrics_dict = {
            "psnr_cond_image": self.psnr(downsampled_image, cond_rgb)
        }

        cond_depth = colormaps.apply_depth_colormap(
            outputs["depth"].squeeze(0),
            accumulation=outputs["accumulation"].squeeze(0),
        )

        images_dict = {"cond_img": torch.cat([downsampled_image.squeeze(0), cond_rgb.squeeze(0), cond_depth], 1)}

        if self.config.cond_only:
            return metrics_dict, images_dict

        rgb_row_1 = torch.cat([image, *batch[NEIGHBORING_VIEW_IMAGES].squeeze(0)], 1)
        rgb_row_2 = torch.cat([*outputs["samples"]], 1)
        combined_rgb = torch.cat([rgb_row_1, rgb_row_2], dim=0)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]

        avg_metrics_dict = defaultdict(list)
        for i, sample in enumerate(outputs["samples"]):
            sample = torch.moveaxis(sample, -1, 0)[None, ...]

            psnr = self.psnr(image, sample)
            ssim = self.ssim(image, sample)
            lpips = self.lpips(image, sample)
            assert isinstance(ssim, torch.Tensor)
            metrics_dict.update({
                f"psnr/{i}": psnr,
                f"ssim/{i}": ssim,
                f"lpips/{i}": lpips
            })
            avg_metrics_dict["psnr"].append(psnr)
            avg_metrics_dict["ssim"].append(ssim)
            avg_metrics_dict["lpips"].append(lpips)

        for key, val in avg_metrics_dict.items():
            metrics_dict[f"{key}/min"] = torch.FloatTensor(val).min()
            metrics_dict[f"{key}/max"] = torch.FloatTensor(val).max()
            metrics_dict[f"{key}/mean"] = torch.FloatTensor(val).mean()

        images_dict["img"] = combined_rgb

        # torch.cuda.empty_cache()

        return metrics_dict, images_dict
