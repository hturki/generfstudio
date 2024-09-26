from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union, Type, Optional, Literal

import torch
import torch.nn.functional as F
from diffusers import UNet2DConditionModel, EMAModel, DDIMScheduler, DDPMScheduler
from diffusers.utils.torch_utils import randn_tensor
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.utils import colormaps, profiler
from nerfstudio.utils.comms import get_world_size
from nerfstudio.utils.rich_utils import CONSOLE
from torch import nn
from torch.nn import Parameter
from torch_scatter import scatter_min
from torchvision.transforms import transforms
from transformers import CLIPVisionModelWithProjection

from generfstudio.generfstudio_constants import RGB, ACCUMULATION, DEPTH, ALIGNMENT_LOSS, PTS3D, VALID_ALIGNMENT, \
    DEPTH_GT, BG_COLOR
from generfstudio.models.rgbd_diffusion_base import RGBDDiffusionBase, RGBDDiffusionBaseConfig


@dataclass
class RGBDDiffusionIFConfig(RGBDDiffusionBaseConfig):
    _target: Type = field(default_factory=lambda: RGBDDiffusionIF)

    unet_pretrained_path: str = "DeepFloyd/IF-I-XL-v1.0"

    beta_schedule: str = "squaredcos_cap_v2"
    prediction_type: str = "epsilon"

    image_crossattn: Literal["clip-expand", "clip-replace", "unet-replace", "none"] = "clip-replace"

    image_dim: int = 64

    rgb_mapping: Literal["clamp", "sigmoid"] = "clamp"
    depth_mapping: Literal["scaled", "disparity", "log"] = "scaled"

    is_upscaler: bool = False
    custom_upscale: bool = False
    upscaler_noise_level: int = 250


class RGBDDiffusionIF(RGBDDiffusionBase):
    config: RGBDDiffusionIFConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        if self.config.is_upscaler:
            assert self.config.unet_pretrained_path == "DeepFloyd/IF-II-L-v1.0" or self.config.unet_pretrained_path == "DeepFloyd/IF-II-M-v1.0", self.config.unet_pretrained_path
            assert self.config.image_dim == 256, self.config.image_dim
            if not self.config.custom_upscale:
                self.image_noising_scheduler = DDPMScheduler.from_pretrained(self.config.unet_pretrained_path,
                                                                             subfolder="image_noising_scheduler",
                                                                             beta_schedule=self.config.beta_schedule,
                                                                             prediction_type=self.config.prediction_type)
            self.unet_resize_downscaled = transforms.Resize(
                (self.config.image_dim // 4, self.config.image_dim // 4),
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True,
            )

        if self.config.image_crossattn == "clip-expand":
            image_encoder_base = CLIPVisionModelWithProjection.from_pretrained(
                self.config.image_encoder_pretrained_path, subfolder="image_encoder")
            self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                self.config.image_encoder_pretrained_path, subfolder="image_encoder", projection_dim=4096,
                ignore_mismatched_sizes=True)

            old_state = image_encoder_base.state_dict()
            new_state = self.image_encoder.state_dict()
            in_key = "visual_projection.weight"
            if old_state[in_key].shape != new_state[in_key].shape:
                CONSOLE.log(f"Manual init for CLIPVisionModelWithProjection: {in_key}")
                new_state[in_key].zero_()
                new_state[in_key][:old_state[in_key].shape[0]].copy_(old_state[in_key])
                self.image_encoder.load_state_dict(new_state)

            self.image_encoder.vision_model.requires_grad_(False)
        elif self.config.image_crossattn == "clip-replace":
            self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                self.config.image_encoder_pretrained_path, subfolder="image_encoder", projection_dim=4096,
                ignore_mismatched_sizes=True)
            self.image_encoder.vision_model.requires_grad_(False)
        elif self.config.image_crossattn == "unet-replace":
            self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                self.config.image_encoder_pretrained_path, subfolder="image_encoder")
            self.image_encoder.requires_grad_(False)
        elif self.config.image_crossattn == "none":
            pass
        else:
            raise Exception(self.config.image_crossattn)

        self.unet_resize = transforms.Resize(
            (self.config.image_dim, self.config.image_dim),
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=True,
        )

        unet_base = UNet2DConditionModel.from_pretrained(self.config.unet_pretrained_path, subfolder="unet",
                                                         low_cpu_mem_usage=False,
                                                         ignore_mismatched_sizes=False)

        if self.config.image_crossattn == "clip-expand" or self.config.image_crossattn == "clip-replace":
            encoder_hid_dim = unet_base.config.encoder_hid_dim
        elif self.config.image_crossattn == "unet-replace":
            encoder_hid_dim = 768
        elif self.config.image_crossattn == "none":
            encoder_hid_dim = None
        else:
            raise Exception(self.config.image_crossattn)

        # 4 for noise + 4 for RGBD cond + 1 for accumulation
        unet_in_channels = 4 + 4 + (0 if self.config.is_upscaler else 1) + (5 if self.config.custom_upscale else 0)
        unet_out_channels = 3 if self.config.rgb_only else 4
        self.unet = UNet2DConditionModel.from_pretrained(self.config.unet_pretrained_path, subfolder="unet",
                                                         in_channels=unet_in_channels,
                                                         out_channels=unet_out_channels,
                                                         low_cpu_mem_usage=False,
                                                         ignore_mismatched_sizes=True,
                                                         encoder_hid_dim=encoder_hid_dim)
        old_state = unet_base.state_dict()
        new_state = self.unet.state_dict()
        in_key = "conv_in.weight"
        if old_state[in_key].shape != new_state[in_key].shape:
            CONSOLE.log(f"Manual init: {in_key}")
            new_state[in_key].zero_()
            if self.config.is_upscaler:
                new_state[in_key][:, :old_state[in_key].shape[1] // 2, :, :].copy_(
                    old_state[in_key][:, :old_state[in_key].shape[1] // 2, :, :])
                new_state[in_key][:, old_state[in_key].shape[1] // 2 + 1:old_state[in_key].shape[1] // 2 + 4, :,
                :].copy_(old_state[in_key][:, old_state[in_key].shape[1] // 2:, :, :])
            else:
                new_state[in_key][:, :old_state[in_key].shape[1], :, :].copy_(old_state[in_key])

        out_keys = ["conv_out.weight", "conv_out.bias"]
        for out_key in out_keys:
            if old_state[out_key].shape != new_state[out_key].shape:
                CONSOLE.log(f"Manual init: {in_key}")
                # new_state[out_key].zero_()
                new_state[out_key][:old_state[out_key].shape[0] // 2].copy_(
                    old_state[out_key][:old_state[out_key].shape[0] // 2])
                # new_state[out_key][-1].copy_(old_state[out_key][old_state[out_key].shape[0] // 2:].mean(dim=0))

        self.unet.load_state_dict(new_state)
        self.unet.enable_xformers_memory_efficient_attention()

        if self.config.use_ema:
            self.ema_unet = EMAModel(self.unet.parameters(), model_cls=UNet2DConditionModel,
                                     model_config=self.unet.config)

        if self.config.use_ddim:
            # Need to explicitly set this for DDIMScheduler for some reason
            self.ddim_scheduler.config["variance_type"] = "fixed_small"
            self.ddim_scheduler = DDIMScheduler.from_config(self.ddim_scheduler.config)
            CONSOLE.log(f"DDIM Scheduler: {self.ddim_scheduler.config}")

            ddim_scheduler_no_threshold = DDPMScheduler.from_config(self.ddim_scheduler.config)
            ddim_scheduler_no_threshold.config["thresholding"] = False
            ddim_scheduler_no_threshold.config["clip_sample"] = False
            self.ddim_scheduler_no_thresh = DDIMScheduler.from_config(ddim_scheduler_no_threshold.config)

        ddpm_scheduler_no_threshold = DDPMScheduler.from_config(self.ddpm_scheduler.config)
        ddpm_scheduler_no_threshold.config["thresholding"] = False
        ddpm_scheduler_no_threshold.config["clip_sample"] = False
        self.ddpm_scheduler_no_thresh = DDPMScheduler.from_config(ddpm_scheduler_no_threshold.config)

        if not self.config.rgb_only:
            pixels_im = torch.stack(
                torch.meshgrid(
                    torch.arange(self.config.image_dim, dtype=torch.float32),
                    torch.arange(self.config.image_dim, dtype=torch.float32),
                    indexing="ij")).permute(2, 1, 0)
            self.register_buffer("pixels", pixels_im.reshape(-1, 2), persistent=False)

        if get_world_size() > 1:
            CONSOLE.log("Using sync batchnorm for DDP")
            self.unet = nn.SyncBatchNorm.convert_sync_batchnorm(self.unet)
            self.depth_estimator_field = nn.SyncBatchNorm.convert_sync_batchnorm(self.depth_estimator_field)

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {"fields": list(self.unet.parameters())}
        if self.config.image_crossattn == "clip-expand" or self.config.image_crossattn == "clip-replace":
            param_groups["fields"] += list(self.image_encoder.visual_projection.parameters())

        return param_groups

    def encode_rgbd(self, rgb: torch.Tensor, depth: Optional[torch.Tensor],
                    depth_scaling_factor: torch.Tensor) -> torch.Tensor:
        if self.config.rgb_mapping == "clamp":
            to_encode = (2 * rgb - 1).clamp(-1, 1).permute(0, 3, 1, 2)
        elif self.config.rgb_mapping == "sigmoid":
            to_encode = torch.logit((rgb.float().clamp(0, 1) * 0.99998) + 0.00001).permute(0, 3, 1, 2)
        else:
            raise Exception(self.config.rgb_mapping)

        if not self.config.rgb_only:
            encoded_depth = self.encode_depth(depth, depth_scaling_factor).permute(0, 3, 1, 2)
            to_encode = torch.cat([to_encode, encoded_depth], 1)

        return to_encode

    def decode_rgbd(self, to_decode: torch.Tensor, depth_scaling_factor: torch.Tensor) -> Tuple[
        torch.Tensor, Optional[torch.Tensor]]:
        to_decode = to_decode.permute(0, 2, 3, 1)

        rgb = self.decode_rgb(to_decode[..., :3])

        if self.config.rgb_only:
            return rgb, None

        return rgb, self.decode_depth(to_decode[..., 3:], depth_scaling_factor)

    def decode_rgb(self, encoded_rgb: torch.Tensor) -> torch.Tensor:
        if self.config.rgb_mapping == "clamp":
            return (0.5 * (encoded_rgb + 1)).clamp(0, 1)
        elif self.config.rgb_mapping == "sigmoid":
            return ((torch.sigmoid(encoded_rgb) - 0.00001) / 0.99998).clamp(0, 1)
        else:
            raise Exception(self.config.rgb_mapping)

    @profiler.time_function
    # @torch.inference_mode()
    def get_concat(self, w2cs: torch.Tensor, cameras: Cameras, pts3d: torch.Tensor, scales: torch.Tensor,
                   opacity: torch.Tensor, rgbs: torch.Tensor, cameras_per_scene: int, bg_colors: Optional[torch.Tensor],
                   depth_scaling_factor: torch.Tensor):
        camera_dim = cameras.width[0, 0].item()
        fx = cameras.fx.view(w2cs.shape[0], 1)
        fy = cameras.fy.view(w2cs.shape[0], 1)
        cx = cameras.cx.view(w2cs.shape[0], 1)
        cy = cameras.cy.view(w2cs.shape[0], 1)

        image_dim = self.config.image_dim
        if camera_dim != image_dim:
            scale_factor = image_dim / camera_dim
            camera_dim = image_dim
            fx = fx * scale_factor
            fy = fy * scale_factor
            cx = cx * scale_factor
            cy = cy * scale_factor

        S = w2cs.shape[0] // cameras_per_scene
        rendering = self.splat_gaussians(
            w2cs,
            fx,
            fy,
            cx,
            cy,
            pts3d.view(S, -1, 3).float(),
            scales.view(S, -1, 1).expand(-1, -1, 3),
            opacity.view(S, -1).float(),
            rgbs.reshape(S, -1, 3),
            camera_dim,
            not self.config.rgb_only,
            cameras_per_scene,
            bg_colors,
        )

        concat_list = [self.encode_rgbd(rendering[RGB],
                                        rendering[DEPTH] if not self.config.rgb_only else None, depth_scaling_factor),
                       rendering[ACCUMULATION].permute(0, 3, 1, 2)]

        return torch.cat(concat_list, 1), rendering

    @profiler.time_function
    # @torch.inference_mode()
    def project_pointcloud(self, w2cs: torch.Tensor, pts3d: torch.Tensor, rgb: torch.Tensor, fx: torch.Tensor,
                           fy: torch.Tensor, cx: torch.Tensor, cy: torch.Tensor,
                           existing_rgb: Optional[torch.Tensor], existing_depth: Optional[torch.Tensor]) -> Dict[
        str, torch.Tensor]:
        image_dim = self.config.image_dim

        if existing_rgb is None:
            rgb_map = torch.zeros(w2cs.shape[0] * image_dim * image_dim, 3, dtype=rgb.dtype, device=rgb.device)
        else:
            rgb_map = existing_rgb.reshape(-1, 3).clone().type(rgb.dtype)

        if existing_depth is None:
            depth_map = torch.full((w2cs.shape[0] * image_dim * image_dim,), torch.finfo(torch.float32).max - 1,
                                   device=pts3d.device)
        else:
            depth_map = existing_depth.view(-1).clone()

        with torch.cuda.amp.autocast(enabled=False):
            pts3d_cam_render = w2cs @ torch.cat([pts3d, torch.ones_like(pts3d[..., :1])], -1).transpose(1, 2)
            pts3d_cam_render = pts3d_cam_render.transpose(1, 2)[..., :3]
            z_pos = pts3d_cam_render[..., 2]

            uv = pts3d_cam_render[:, :, :2] / pts3d_cam_render[:, :, 2:]
            focal = torch.cat([fx.view(-1, 1), fy.view(-1, 1)], -1)
            uv *= focal.unsqueeze(1)
            center = torch.cat([cx.view(-1, 1), cy.view(-1, 1)], -1)
            uv += center.unsqueeze(1)

            is_valid_u = torch.logical_and(0 <= uv[..., 0], uv[..., 0] < image_dim)
            is_valid_v = torch.logical_and(0 <= uv[..., 1], uv[..., 1] < image_dim)
            is_valid_z = z_pos > 0
            is_valid_point = torch.logical_and(torch.logical_and(is_valid_u, is_valid_v), is_valid_z)

            max_val = torch.finfo(torch.float32).max
            u = torch.round(uv[..., 0]).long().clamp(0, image_dim - 1)
            v = torch.round(uv[..., 1]).long().clamp(0, image_dim - 1)
            z = torch.where(is_valid_point, z_pos, max_val)
            camera_index = torch.arange(w2cs.shape[0], device=z.device).view(-1, 1).expand(-1, z.shape[-1])

            indices = (camera_index * image_dim * image_dim + v * image_dim + u).view(-1)
            _, min_indices = scatter_min(z.view(-1), indices, out=depth_map)
            depth_map = depth_map.view(w2cs.shape[0], image_dim, image_dim, 1)

            rgb_to_fill = min_indices[min_indices <= indices.shape[0] - 1]
            rgb_map[indices[rgb_to_fill]] = rgb.reshape(-1, 3)[rgb_to_fill]
            rgb_map = rgb_map.reshape(*depth_map.shape[:-1], 3)  # .permute(0, 3, 1, 2)

            acc_map = depth_map < torch.finfo(torch.float32).max - 1
            depth_map[torch.logical_not(acc_map)] = 0

            return {
                RGB: rgb_map,
                DEPTH: depth_map,
                ACCUMULATION: acc_map.float(),
            }

    def not_in_existing_predictions(self, w2cs: torch.Tensor, pts3d: torch.Tensor, fx: torch.Tensor,
                                    fy: torch.Tensor, cx: torch.Tensor, cy: torch.Tensor,
                                    existing_depth: torch.Tensor, image_width: int, image_height: int) -> torch.Tensor:
        _, min_indices = self.project_depth_to_other_views(w2cs, pts3d, fx, fy, cx, cy, existing_depth, image_width,
                                                           image_height)
        invalid_indices = torch.where(min_indices < pts3d.shape[0] * pts3d.shape[1], min_indices % pts3d.shape[1], -1)
        mask = torch.ones_like(pts3d[0, :, 0], dtype=torch.bool)
        mask[invalid_indices[invalid_indices > 0]] = False
        return mask

    def get_outputs(self, cameras: Cameras) -> Dict[str, Union[torch.Tensor, List]]:
        outputs = {}
        images = cameras.metadata["image"]
        pts3d = cameras.metadata.get(PTS3D, None)
        depth = cameras.metadata.get(DEPTH, None)
        # fg_masks = cameras.metadata.get(FG_MASK, None)
        bg_colors = None #cameras.metadata.get(BG_COLOR, None)

        cameras.metadata = {}  # Should do deepcopy instead if we still need any metadata here

        if pts3d is not None:
            clip_rgbs = (images[:, 1:] if self.training else images[:, -1:]).permute(0, 1, 4, 2, 3)
            # if fg_masks is None:
            fg_masks = torch.ones_like(pts3d[..., :1], dtype=torch.bool)

            valid_alignment = torch.ones(cameras.shape[0], dtype=torch.bool, device=cameras.device)
        else:
            rgbs_chw = images.permute(0, 1, 4, 2, 3)
            clip_rgbs = (rgbs_chw[:, 1:] if self.training else rgbs_chw[:, -1:])

            c2ws_opencv = torch.eye(4, device=cameras.device).view(1, 1, 4, 4).repeat(cameras.shape[0],
                                                                                      cameras.shape[1], 1, 1)
            c2ws_opencv[:, :, :3] = cameras.camera_to_worlds
            c2ws_opencv[:, :, :, 1:3] *= -1  # opengl to opencv

            pts3d, depth, alignment_loss, valid_alignment, confs = self.depth_estimator_field.get_pts3d_and_depth(
                rgbs_chw, None, c2ws_opencv, cameras.fx, cameras.fy, cameras.cx, cameras.cy)

            outputs[ALIGNMENT_LOSS] = alignment_loss
            outputs[VALID_ALIGNMENT] = valid_alignment.float().mean()

            depth = depth.clamp_min(0).view(*cameras.shape[:2], *rgbs_chw.shape[3:])

            fg_masks = (confs > self.config.min_conf_thresh).view(*cameras.shape[:2], *rgbs_chw.shape[3:])
            # conf_mask = (confs > self.config.min_conf_thresh).view(*cameras.shape[:2], *rgbs_chw.shape[3:])
            # fg_masks = torch.logical_and(fg_masks, conf_mask) if fg_masks is not None else conf_mask
            pts3d = pts3d.view(*cameras.shape[:2], *pts3d.shape[1:])

        depth_scaling_factor = self.get_depth_scaling_factor(depth.view(cameras.shape[0], -1))

        if self.training:
            if self.config.depth_mapping == "scaled":
                valid_alignment = torch.logical_and(valid_alignment,
                                                    (depth_scaling_factor[:, 1] - depth_scaling_factor[:, 0]).view(
                                                        cameras.shape[0]) > 0.01)
            else:
                valid_alignment = torch.logical_and(valid_alignment,
                                                    depth_scaling_factor.view(cameras.shape[0], -1).max(dim=-1)[0] > 0)

            if valid_alignment.any():
                cameras = cameras[valid_alignment]
                images = images[valid_alignment]
                pts3d = pts3d[valid_alignment]
                depth = depth[valid_alignment]
                clip_rgbs = clip_rgbs[valid_alignment]
                fg_masks = fg_masks[valid_alignment]
                depth_scaling_factor = depth_scaling_factor[valid_alignment]

                if bg_colors is not None:
                    bg_colors = bg_colors[valid_alignment]

        if self.config.image_crossattn == "none":
            c_crossattn = torch.zeros(cameras.shape[0], 1, 1280, device=cameras.device)
        else:
            # Get CLIP embeddings for cross attention
            c_crossattn = self.image_encoder(
                self.clip_normalize(self.clip_resize(clip_rgbs.flatten(0, 1)).clamp(0, 1))).image_embeds
            c_crossattn = c_crossattn.view(cameras.shape[0], clip_rgbs.shape[1], c_crossattn.shape[-1])

        to_project = cameras[:, 1:] if self.training else cameras[:, -1:]
        if self.config.scale_with_pixel_area:
            scale_mult = self.get_pixel_area_scale_mult(to_project.flatten(),
                                                        cameras.width[0, 0].item()).view(*to_project.shape, -1)
        else:
            scale_mult = 1 / to_project.fx

        if self.training:
            timesteps = torch.randint(0, self.ddpm_scheduler.config.num_train_timesteps, (cameras.shape[0],),
                                      device=self.device, dtype=torch.long)
            outputs["timesteps"] = timesteps

            # Hack - we include the gt image in the forward loop, so that all model calculation is done in the
            # forward call as expected in DDP
            image_gt = images[:, 0]
            if image_gt.shape[-1] != self.config.image_dim:
                with torch.inference_mode():
                    image_gt = self.unet_resize(image_gt.permute(0, 3, 1, 2)).permute(0, 2, 3, 1).clamp(0, 1)

            if not self.config.rgb_only:
                depth_gt = depth[:, :1]
                if depth_gt.shape[-1] != self.config.image_dim:
                    with torch.inference_mode():
                        depth_gt = self.unet_resize(depth_gt).clamp_min(0)
                depth_gt = depth_gt.permute(0, 2, 3, 1)
            else:
                depth_gt = None

            input_gt = self.encode_rgbd(image_gt, depth_gt, depth_scaling_factor)
            noise = torch.randn_like(input_gt)

            if self.config.is_upscaler:
                to_upscale_rgb = images[:, 0]
                if to_upscale_rgb.shape[-1] != self.config.image_dim // 4:
                    with torch.inference_mode():
                        to_upscale_rgb = self.unet_resize_downscaled(
                            to_upscale_rgb.permute(0, 3, 1, 2)).permute(0, 2, 3, 1).clamp(0, 1)

                if not self.config.rgb_only:
                    to_upscale_depth = depth[:, :1]
                    if to_upscale_depth.shape[-1] != self.config.image_dim // 4:
                        with torch.inference_mode():
                            to_upscale_depth = self.unet_resize_downscaled(to_upscale_depth).clamp_min(0)
                    to_upscale_depth = to_upscale_depth.permute(0, 2, 3, 1)
                else:
                    to_upscale_depth = None

                with torch.inference_mode():
                    c_concat = self.unet_resize(
                        self.encode_rgbd(to_upscale_rgb, to_upscale_depth, depth_scaling_factor))

                if self.config.custom_upscale:
                    target_w2cs_opencv = self.c2ws_opengl_to_w2cs_opencv(cameras.camera_to_worlds[:, :1]).squeeze(1)

                    cond_pts3d = pts3d[:, 1:]
                    c_concat_2, _ = self.get_concat(
                        target_w2cs_opencv,
                        cameras[:, 0],
                        cond_pts3d,
                        depth[:, 1:].flatten(-2, -1) * scale_mult,
                        fg_masks[:, 1:].view(cond_pts3d[..., :1].shape).float(),
                        images[:, 1:],
                        1,
                        bg_colors,
                        depth_scaling_factor
                    )
                    c_concat = torch.cat([c_concat, c_concat_2], dim=1)
                else:
                    c_concat = self.ddpm_scheduler.add_noise(c_concat, noise, timesteps)
            else:
                target_w2cs_opencv = self.c2ws_opengl_to_w2cs_opencv(cameras.camera_to_worlds[:, :1]).squeeze(1)

                cond_pts3d = pts3d[:, 1:]
                c_concat, _ = self.get_concat(
                    target_w2cs_opencv,
                    cameras[:, 0],
                    cond_pts3d,
                    depth[:, 1:].flatten(-2, -1) * scale_mult,
                    fg_masks[:, 1:].view(cond_pts3d[..., :1].shape).float(),
                    images[:, 1:],
                    1,
                    bg_colors,
                    depth_scaling_factor
                )

            random_mask = torch.rand((c_concat.shape[0], 1, 1), device=c_concat.device)
            prompt_mask = random_mask < 2 * self.config.uncond
            input_mask = torch.logical_not(
                torch.logical_and(random_mask >= self.config.uncond,
                                  random_mask < 3 * self.config.uncond)).float().unsqueeze(-1)

            c_crossattn = torch.where(prompt_mask, 0, c_crossattn)

            c_concat = input_mask * c_concat

            noise = torch.randn_like(input_gt)
            noisy_input_gt = self.ddpm_scheduler.add_noise(input_gt, noise, timesteps)
            model_output = self.unet(
                torch.cat([noisy_input_gt, c_concat], 1), timesteps,
                c_crossattn,
                class_labels=timesteps if self.config.is_upscaler else None,
                return_dict=False)[0]

            if self.ddpm_scheduler.config.prediction_type == "epsilon":
                outputs["target"] = noise
            elif self.ddpm_scheduler.config.prediction_type == "sample":
                outputs["target"] = input_gt
            elif self.ddpm_scheduler.config.prediction_type == "v_prediction":
                outputs["target"] = self.ddpm_scheduler.get_velocity(input_gt, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {self.ddpm_scheduler.config.prediction_type}")

            outputs["model_output"] = model_output

            if not torch.isfinite(outputs["model_output"]).all():
                torch.save([outputs, images, depth, input_gt, image_gt, depth_gt, depth_scaling_factor, noisy_input_gt,
                            c_concat, c_crossattn, timesteps],
                           "nan-debug.pt")

                import pdb;
                pdb.set_trace()
        else:
            assert cameras.shape[0] == 1, cameras.shape

            depth_gt = depth.squeeze(0).unsqueeze(1)
            if depth_gt.shape[-1] != self.config.image_dim:
                with torch.inference_mode():
                    depth_gt = self.unet_resize(depth_gt).clamp_min(0)
            outputs[DEPTH_GT] = depth_gt.permute(0, 2, 3, 1)

            target_w2cs_opencv = self.c2ws_opengl_to_w2cs_opencv(cameras.camera_to_worlds[:, :-1]).squeeze(0)
            target_c2ws_opencv = torch.eye(4, device=cameras.device).view(1, 4, 4).repeat(target_w2cs_opencv.shape[0],
                                                                                          1, 1)
            target_c2ws_opencv[:, :3] = cameras.camera_to_worlds[0, :-1]
            target_c2ws_opencv[:, :, 1:3] *= -1  # opengl to opencv

            if self.config.use_ddim:
                scheduler = self.ddim_scheduler
                scheduler_no_thresh = self.ddim_scheduler_no_thresh
            else:
                scheduler = self.ddpm_scheduler
                scheduler_no_thresh = self.ddpm_scheduler_no_thresh

            scheduler.set_timesteps(self.config.inference_steps, device=self.device)
            scheduler_no_thresh.set_timesteps(self.config.inference_steps, device=self.device)

            cond_pts3d = pts3d[:, -1:]
            opacity = fg_masks[:, -1:].view(cond_pts3d[..., :1].shape).float()
            scales = depth[:, -1:].flatten(-2, -1) * scale_mult
            cond_rgbs = images[:, -1:].view(1, -1, 3)

            c_concat, rendering = self.get_concat(
                target_w2cs_opencv,
                cameras[0, :-1],
                cond_pts3d,
                scales,
                opacity,
                cond_rgbs,
                target_w2cs_opencv.shape[0],
                bg_colors,
                depth_scaling_factor
            )

            outputs[RGB] = torch.cat([*rendering[RGB]], 1)
            if not self.config.rgb_only:
                outputs[DEPTH] = torch.cat([*rendering[DEPTH]], 1)
                outputs[ACCUMULATION] = torch.cat([*rendering[ACCUMULATION]], 1)

            if self.config.is_upscaler:
                to_upscale_rgb = images[0, :-1]
                if to_upscale_rgb.shape[-1] != self.config.image_dim // 4:
                    with torch.inference_mode():
                        to_upscale_rgb = self.unet_resize_downscaled(to_upscale_rgb.permute(0, 3, 1, 2)
                                                                     ).permute(0, 2, 3, 1).clamp(0, 1)

                if not self.config.rgb_only:
                    to_upscale_depth = depth[0, :-1].unsqueeze(1)
                    if to_upscale_depth.shape[-1] != self.config.image_dim // 4:
                        with torch.inference_mode():
                            to_upscale_depth = self.unet_resize_downscaled(to_upscale_depth).clamp_min(0)
                    to_upscale_depth = to_upscale_depth.permute(0, 2, 3, 1)
                else:
                    to_upscale_depth = None

                with torch.inference_mode():
                    upscaled = self.unet_resize(
                        self.encode_rgbd(to_upscale_rgb, to_upscale_depth, depth_scaling_factor))

                if not self.config.custom_upscale:
                    noise_level = torch.tensor([self.config.upscaler_noise_level] * upscaled.shape[0],
                                               device=upscaled.device)
                    upscaled_noise = randn_tensor(upscaled.shape, device=upscaled.device, dtype=upscaled.dtype)
                    upscaled = self.image_noising_scheduler.add_noise(upscaled, upscaled_noise, timesteps=noise_level)

                c_concat = torch.cat([upscaled, c_concat], 1)

            c_crossattn = c_crossattn.expand(target_w2cs_opencv.shape[0], -1, -1)

            if self.config.guidance_scale > 1:
                c_concat = torch.cat([torch.zeros_like(c_concat), c_concat])
                c_crossattn = torch.cat([torch.zeros_like(c_crossattn), c_crossattn])

            sample = scheduler.init_noise_sigma * randn_tensor(
                (target_w2cs_opencv.shape[0], 3 if self.config.rgb_only else 4,
                 self.config.image_dim, self.config.image_dim),
                device=c_concat.device, dtype=c_concat.dtype)

            if self.config.use_ema:
                self.ema_unet.to(self.device)
                self.ema_unet.store(self.unet.parameters())
                self.ema_unet.copy_to(self.unet.parameters())
            try:
                for t_index, t in enumerate(scheduler.timesteps):
                    model_input = torch.cat([sample] * 2) if self.config.guidance_scale > 1 else sample
                    model_input = scheduler.scale_model_input(model_input, t)
                    model_input = torch.cat([model_input, c_concat], dim=1)

                    if self.config.is_upscaler:
                        class_labels = (
                            torch.tensor([t], device=model_input.device).expand(target_w2cs_opencv.shape[0], -1)) \
                            if self.config.custom_upscale else noise_level
                        if self.config.guidance_scale > 1:
                            class_labels = torch.cat([class_labels, class_labels])
                    else:
                        class_labels = None

                    model_output = \
                        self.unet(model_input, t, c_crossattn, class_labels=class_labels, return_dict=False)[0]

                    if self.config.guidance_scale > 1:
                        model_output_uncond, model_output_cond = model_output.chunk(2)
                        model_output = model_output_uncond + self.config.guidance_scale * \
                                       (model_output_cond - model_output_uncond)

                    if self.config.rgb_mapping == "clamp":
                        sample_rgb = scheduler.step(model_output[:, :3], t, sample[:, :3], return_dict=False)[0]
                    else:
                        sample_rgb = scheduler_no_thresh.step(model_output[:, :3], t, sample[:, :3], return_dict=False)[
                            0]

                    if self.config.depth_mapping == "scaled" or self.config.depth_mapping == "disparity":
                        sample_depth = scheduler.step(model_output[:, 3:], t, sample[:, 3:], return_dict=False)[0]
                    else:
                        sample_depth = \
                            scheduler_no_thresh.step(model_output[:, 3:], t, sample[:, 3:], return_dict=False)[0]

                    sample = torch.cat([sample_rgb, sample_depth], dim=1)

                sample_rgb, sample_depth = self.decode_rgbd(sample, depth_scaling_factor)
                outputs["samples_rgb"] = torch.cat([*sample_rgb], 1)
                outputs["samples_depth"] = torch.cat([*sample_depth], 1)
            finally:
                if self.config.use_ema:
                    self.ema_unet.restore(self.unet.parameters())

        return outputs

    @torch.no_grad()
    def get_outputs_for_camera(self, camera: Cameras, obb_box: Optional[OrientedBox] = None) -> Dict[str, torch.Tensor]:
        return self.get_outputs(camera)

    def get_metrics_dict(self, outputs, batch):
        assert self.training
        if ALIGNMENT_LOSS in outputs:
            return {ALIGNMENT_LOSS: outputs[ALIGNMENT_LOSS], VALID_ALIGNMENT: outputs[VALID_ALIGNMENT]}
        else:
            return {}

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        model_output = outputs["model_output"]
        target = outputs["target"]

        loss = F.mse_loss(model_output, target.to(model_output), reduction="mean")
        assert torch.isfinite(loss).all(), loss

        loss_dict = {"loss": loss}

        if VALID_ALIGNMENT in outputs and outputs[VALID_ALIGNMENT] == 0:
            CONSOLE.log("No valid outputs, ignoring training batch")
            return {k: 0 * v for k, v in loss_dict.items()}

        return loss_dict

    @profiler.time_function
    def get_image_metrics_and_images(
            self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        image_gt = batch["image"].squeeze(0).to(outputs[RGB].device)
        if image_gt.shape[-1] != self.config.image_dim:
            with torch.inference_mode():
                image_gt = self.unet_resize(image_gt.permute(0, 3, 1, 2)).permute(0, 2, 3, 1).clamp(0, 1)

        depth_gt = outputs[DEPTH_GT]

        source_view = torch.cat([image_gt[-1], colormaps.apply_depth_colormap(depth_gt[-1])], 1)

        target_rgb = torch.cat([*image_gt[:-1]], 1)
        concat_rgb = torch.cat([outputs[RGB], target_rgb])
        samples_rgb = torch.cat([outputs["samples_rgb"], target_rgb])

        images_dict = {
            "source_view": source_view,
            "concat_rgb": concat_rgb,
            "samples_rgb": samples_rgb
        }

        if not self.config.rgb_only:
            target_depth = torch.cat([*depth_gt[:-1]], 1)
            near_plane = target_depth.min()
            far_plane = target_depth.max()
            images_dict["concat_depth"] = torch.cat(
                [colormaps.apply_depth_colormap(outputs[DEPTH], outputs[ACCUMULATION], near_plane=near_plane,
                                                far_plane=far_plane),
                 colormaps.apply_depth_colormap(target_depth)])
            images_dict["samples_depth"] = colormaps.apply_depth_colormap(
                torch.cat([outputs["samples_depth"], target_depth]), near_plane=near_plane, far_plane=far_plane)

        return {}, images_dict

    def load_state_dict(self, dict, **kwargs):  # type: ignore
        if not self.depth_available:
            for key, val in self.depth_estimator_field.state_dict().items():
                dict[f"depth_estimator_field.{key}"] = val

        super().load_state_dict(dict, **kwargs)
