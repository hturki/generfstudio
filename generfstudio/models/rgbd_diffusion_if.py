from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union, Type, Optional, Literal

import torch
import torch.nn.functional as F
from diffusers import UNet2DConditionModel, EMAModel, DDIMScheduler
from diffusers.utils.torch_utils import randn_tensor
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.utils import colormaps, profiler
from nerfstudio.utils.comms import get_world_size
from nerfstudio.utils.rich_utils import CONSOLE
from torch import nn
from torch.nn import Parameter
from torch_scatter import scatter_min
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity
from torchvision.transforms import transforms
from transformers import CLIPVisionModelWithProjection

from generfstudio.fields.batched_pc_optimizer import fast_depthmap_to_pts3d
from generfstudio.generfstudio_constants import RGB, ACCUMULATION, DEPTH, ALIGNMENT_LOSS, PTS3D, VALID_ALIGNMENT, \
    DEPTH_GT, FG_MASK, BG_COLOR
from generfstudio.models.rgbd_diffusion_base import RGBDDiffusionBase, RGBDDiffusionBaseConfig


@dataclass
class RGBDDiffusionIFConfig(RGBDDiffusionBaseConfig):
    _target: Type = field(default_factory=lambda: RGBDDiffusionIF)

    unet_pretrained_path: str = "DeepFloyd/IF-I-L-v1.0"
    image_encoder_pretrained_path: str = "lambdalabs/sd-image-variations-diffusers"
    scheduler_pretrained_path: str = "DeepFloyd/IF-I-L-v1.0"

    beta_schedule: str = "squaredcos_cap_v2"
    prediction_type: str = "epsilon"

    image_crossattn: Literal["clip-expand", "clip-replace", "unet-replace"] = "clip-replace"

    predict_with_projection: bool = False
    image_dim: int = 64
    image_dim_project: Optional[int] = None
    lpips_weight: float = 0.2

    rgb_mapping: Literal["clamp", "sigmoid"] = "clamp"


class RGBDDiffusionIF(RGBDDiffusionBase):
    config: RGBDDiffusionIFConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

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

        # 4 for noise + 4 for RGBD cond + 1 for accumulation
        unet_in_channels = 4 + 4 + 1
        self.unet = UNet2DConditionModel.from_pretrained(self.config.unet_pretrained_path, subfolder="unet",
                                                         in_channels=unet_in_channels,
                                                         out_channels=3 if self.config.rgb_only else 4,
                                                         low_cpu_mem_usage=False,
                                                         ignore_mismatched_sizes=True,
                                                         encoder_hid_dim=768 if self.config.image_crossattn == "unet-replace" else unet_base.config.encoder_hid_dim,
                                                         )
        old_state = unet_base.state_dict()
        new_state = self.unet.state_dict()
        in_key = "conv_in.weight"
        if old_state[in_key].shape != new_state[in_key].shape:
            CONSOLE.log(f"Manual init: {in_key}")
            new_state[in_key].zero_()
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
            self.ddim_scheduler.config["thresholding"] = (
                    (self.config.depth_mapping == "disparity" or self.config.depth_mapping == "scaled")
                    and self.config.rgb_mapping == "clamp")
            self.ddim_scheduler.config["clip_sample"] = False
            self.ddim_scheduler = DDIMScheduler.from_config(self.ddim_scheduler.config)
            CONSOLE.log(f"DDIM Scheduler: {self.ddim_scheduler.config}")

        if not self.config.rgb_only:
            pixels_im = torch.stack(
                torch.meshgrid(
                    torch.arange(self.config.image_dim, dtype=torch.float32),
                    torch.arange(self.config.image_dim, dtype=torch.float32),
                    indexing="ij")).permute(2, 1, 0)
            self.register_buffer("pixels", pixels_im.reshape(-1, 2), persistent=False)

        if self.config.predict_with_projection:
            self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
            for p in self.lpips.parameters():
                p.requires_grad_(False)

        if get_world_size() > 1:
            CONSOLE.log("Using sync batchnorm for DDP")
            self.unet = nn.SyncBatchNorm.convert_sync_batchnorm(self.unet)
            self.dust3r_field = nn.SyncBatchNorm.convert_sync_batchnorm(self.dust3r_field)

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {"fields": list(self.unet.parameters())}
        if self.config.image_crossattn != "unet-replace":
            param_groups["fields"] += list(self.image_encoder.visual_projection.parameters())

        return param_groups

    def encode_rgbd(self, rgb: torch.Tensor, depth: Optional[torch.Tensor],
                    depth_scaling_factor: torch.Tensor) -> torch.Tensor:
        if self.config.rgb_mapping == "clamp":
            to_encode = (2 * rgb - 1).clamp(-1, 1).permute(0, 3, 1, 2)
        elif self.config.rgb_mapping == "sigmoid":
            to_encode = torch.logit((rgb.clamp(0, 1) * 0.99998) + 0.00001).permute(0, 3, 1, 2)
        else:
            raise Exception(self.config.rgb_mapping)

        if not self.config.rgb_only:
            encoded_depth = self.encode_depth(depth, depth_scaling_factor).permute(0, 3, 1, 2)
            to_encode = torch.cat([to_encode, encoded_depth], 1)

        return to_encode

    def decode_rgbd(self, to_decode: torch.Tensor, depth_scaling_factor: torch.Tensor) -> Tuple[
        torch.Tensor, Optional[torch.Tensor]]:
        to_decode = to_decode.permute(0, 2, 3, 1)

        if self.config.rgb_mapping == "clamp":
            rgb = (0.5 * (to_decode[..., :3] + 1)).clamp(0, 1)
        elif self.config.rgb_mapping == "sigmoid":
            rgb = (torch.sigmoid(to_decode[..., :3]) - 0.00001).clamp_min(0) / 0.99998
        else:
            raise Exception(self.config.rgb_mapping)

        if self.config.rgb_only:
            return rgb, None

        return rgb, self.decode_depth(to_decode[..., 3:], depth_scaling_factor)

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

        image_dim = self.config.image_dim_project if self.config.image_dim_project is not None else self.config.image_dim
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

        if self.config.image_dim_project is not None:
            for key in rendering:
                rendering[key] = F.interpolate(rendering[key].permute(0, 3, 1, 2), self.config.image_dim,
                                               mode="bicubic", antialias=True).permute(0, 2, 3, 1)

        concat_list = [self.encode_rgbd(rendering[RGB],
                                        rendering[DEPTH] if not self.config.rgb_only else None, depth_scaling_factor),
                       rendering[ACCUMULATION].permute(0, 3, 1, 2)]

        return torch.cat(concat_list, 1), rendering

    @profiler.time_function
    # @torch.inference_mode()
    def overlay_encoded_rgbd(self, encoded: torch.Tensor, render_cameras: Cameras, c2ws: torch.Tensor,
                             w2cs: torch.Tensor, depth_fx: torch.Tensor, depth_fy: torch.Tensor, depth_cx: torch.Tensor,
                             depth_cy: torch.Tensor, fg_masks: Optional[torch.Tensor],
                             depth_scale_factor: torch.Tensor, alpha: torch.Tensor):
        rgb = (encoded[:, :, :3] + 1) / 2
        depth = self.decode_depth(encoded[:, :, 3:]) * depth_scale_factor.unsqueeze(1)

        pts3d_cam = fast_depthmap_to_pts3d(
            depth[:, 1:].view(depth_fx.shape[0], -1),
            self.pixels,
            torch.cat([depth_fx, depth_fy], 1),
            torch.cat([depth_cx, depth_cy], 1))

        camera_dim = render_cameras.width[0, 0].item()
        fx = render_cameras.fx.view(-1)
        fy = render_cameras.fy.view(-1)
        cx = render_cameras.cx.view(-1)
        cy = render_cameras.cy.view(-1)

        image_dim = self.config.image_dim_project if self.config.image_dim_project is not None else self.config.image_dim
        if camera_dim != image_dim:
            scale_factor = image_dim / camera_dim
            camera_dim = image_dim
            fx = fx * scale_factor
            fy = fy * scale_factor
            cx = cx * scale_factor
            cy = cy * scale_factor

        with torch.cuda.amp.autocast(enabled=False):
            pts3d = c2ws.reshape(-1, 4, 4) @ torch.cat([pts3d_cam, torch.ones_like(pts3d_cam[..., :1])], -1).transpose(
                1, 2)
            pts3d_cam_render = w2cs @ pts3d.transpose(1, 2).reshape(w2cs.shape[0], -1, 4).transpose(1, 2)
            pts3d_cam_render = pts3d_cam_render.transpose(1, 2)[..., :3]
            z_pos = pts3d_cam_render[..., 2]

            uv = pts3d_cam_render[:, :, :2] / pts3d_cam_render[:, :, 2:]
            focal = torch.cat([fx.view(-1, 1), fy.view(-1, 1)], -1)
            uv *= focal.unsqueeze(1)
            center = torch.cat([cx.view(-1, 1), cy.view(-1, 1)], -1)
            uv += center.unsqueeze(1)

            is_valid_u = torch.logical_and(0 <= uv[..., 0], uv[..., 0] < camera_dim - 1)
            is_valid_v = torch.logical_and(0 <= uv[..., 1], uv[..., 1] < camera_dim - 1)
            is_valid_z = z_pos > 0
            is_valid_point = torch.logical_and(torch.logical_and(is_valid_u, is_valid_v), is_valid_z)
            if fg_masks is not None:
                is_valid_point = torch.logical_and(fg_masks > 0, is_valid_point)

            max_val = torch.finfo(torch.float32).max
            u = torch.round(uv[..., 0]).long().clamp(0, image_dim - 1)
            v = torch.round(uv[..., 1]).long().clamp(0, image_dim - 1)
            z = torch.where(is_valid_point, z_pos, max_val)
            camera_index = torch.arange(w2cs.shape[0], device=z.device).view(-1, 1).expand(-1, z.shape[-1])
            depth_original = depth[:, 0].permute(0, 2, 3, 1)
            depth_map = depth_original.clone()
            indices = (camera_index * image_dim * image_dim + v * image_dim + u).view(-1)
            _, min_indices = scatter_min(z.view(-1), indices, out=depth_map.view(-1))

            rgb_to_fill = min_indices[min_indices <= indices.shape[0] - 1]
            rgb_original = rgb[:, 0]
            rgb_map = rgb_original.permute(0, 2, 3, 1).reshape(-1, 3).clone()
            rgb_map[indices[rgb_to_fill]] = rgb.permute(0, 1, 3, 4, 2)[:, 1:].reshape(-1, 3)[rgb_to_fill]
            rgb_map = rgb_map.view(*depth_map.shape[:-1], 3).permute(0, 3, 1, 2)

            if self.config.opacity_style == "constant":
                pass
            elif self.config.opacity_style == "increasing":
                rgb_map = alpha * rgb_map + (1 - alpha) * rgb_original
                depth_map = alpha * depth_map + (1 - alpha) * depth_original
            elif self.config.opacity_style == "decreasing":
                rgb_map = (1 - alpha) * rgb_map + alpha * rgb_original
                depth_map = (1 - alpha) * depth_map + alpha * depth_original
            else:
                raise Exception(self.config.opacity_style)

            if self.config.rgb_only:
                concat_list = [rgb_map * 2 - 1]
            else:
                encoded_depth = (self.encode_depth(depth_map / depth_scale_factor)).permute(0, 3, 1, 2)
                concat_list = [rgb_map * 2 - 1, encoded_depth]

            return torch.cat(concat_list, 1)

    @profiler.time_function
    # @torch.inference_mode()
    def project_encoded_rgbd(self, encoded: torch.Tensor, render_cameras: Cameras, c2ws: torch.Tensor,
                             w2cs: torch.Tensor, depth_fx: torch.Tensor, depth_fy: torch.Tensor, depth_cx: torch.Tensor,
                             depth_cy: torch.Tensor, bg_colors: Optional[torch.Tensor],
                             fg_masks: Optional[torch.Tensor], scale_mult: torch.Tensor,
                             depth_scale_factor: torch.Tensor):
        rgb = (encoded[:, :, :3] + 1) / 2
        depth = self.decode_depth(encoded[:, :, 3:]) * depth_scale_factor.unsqueeze(1)
        pts3d_cam = fast_depthmap_to_pts3d(
            depth.view(depth_fx.shape[0], -1),
            self.pixels,
            torch.cat([depth_fx, depth_fy], 1),
            torch.cat([depth_cx, depth_cy], 1))

        with torch.cuda.amp.autocast(enabled=False):
            pts3d = c2ws.reshape(-1, 4, 4) \
                    @ torch.cat([pts3d_cam, torch.ones_like(pts3d_cam[..., :1])], -1).transpose(1, 2)
            pts3d = pts3d.transpose(1, 2)[..., :3].reshape(w2cs.shape[0], -1, 3)

        scales = depth.view(*scale_mult.shape[:-1], -1) * scale_mult
        opacity = fg_masks if fg_masks is not None else torch.ones_like(pts3d[..., :1])

        return self.get_concat(
            w2cs,
            render_cameras,
            pts3d,
            scales,
            opacity,
            rgb,
            1,
            bg_colors,
            depth_scale_factor,
            self.config.calc_compensations,
            True,
            self.config.opacity_multiplier,
        )

    def get_outputs(self, cameras: Cameras) -> Dict[str, Union[torch.Tensor, List]]:
        outputs = {}
        images = cameras.metadata["image"]
        cameras.metadata = {}  # Should do deepcopy instead if we still need any metadata here

        input_dim = cameras.width[0, 0].item()
        bg_colors = cameras.metadata.get(BG_COLOR, None)

        if PTS3D in cameras.metadata:
            clip_rgbs = (images[:, 1:] if self.training else images[:, -1:]).permute(0, 1, 4, 2, 3)
            pts3d = cameras.metadata[PTS3D]
            depth = cameras.metadata[DEPTH]
            fg_masks = cameras.metadata.get(FG_MASK, None)
            rgbs_to_project = images
            project_dim = input_dim
            project_scale_factor = 1
        else:
            rgbs_chw = images.permute(0, 1, 4, 2, 3)
            clip_rgbs = (rgbs_chw[:, 1:] if self.training else rgbs_chw[:, -1:])
            project_dim = 224

            with torch.inference_mode():
                rgbs_to_project_chw = self.dust3r_resize(rgbs_chw.flatten(0, 1)).clamp(0, 1)
                rgbs_to_project_chw = rgbs_to_project_chw.view(cameras.shape[0], -1, *rgbs_to_project_chw.shape[1:])

            rgbs_to_project = rgbs_to_project_chw.permute(0, 1, 3, 4, 2)

            if FG_MASK in cameras.metadata:
                fg_masks = cameras.metadata[FG_MASK]
                with torch.inference_mode():
                    fg_masks = self.dust3r_resize(fg_masks.flatten(0, 1).unsqueeze(1).float()).squeeze(1).bool()
                    fg_masks = fg_masks.view(cameras.shape[0], -1, *fg_masks.shape[1:])
            else:
                fg_masks = None

            c2ws_opencv = torch.eye(4, device=cameras.device).view(1, 1, 4, 4).repeat(cameras.shape[0],
                                                                                      cameras.shape[1], 1, 1)
            c2ws_opencv[:, :, :3] = cameras.camera_to_worlds
            c2ws_opencv[:, :, :, 1:3] *= -1  # opengl to opencv

            project_scale_factor = 224 / input_dim
            fx = cameras.fx * project_scale_factor
            fy = cameras.fy * project_scale_factor
            cx = cameras.cx * project_scale_factor
            cy = cameras.cy * project_scale_factor

            pts3d, depth, alignment_loss, valid_alignment = self.dust3r_field.get_pts3d_and_depth(
                rgbs_to_project_chw, fg_masks, c2ws_opencv, fx, fy, cx, cy)

            depth = depth.clamp_min(0).view(*cameras.shape[:2], *rgbs_to_project_chw.shape[3:])
            pts3d = pts3d.view(*cameras.shape[:2], *pts3d.shape[1:])

            if self.training:
                max_camera_depth = depth[:, 0].flatten(1, 2).max(dim=-1)[0]
                valid_alignment = torch.logical_and(valid_alignment, max_camera_depth > 0)

                if valid_alignment.any():
                    cameras = cameras[valid_alignment]
                    images = images[valid_alignment]
                    pts3d = pts3d[valid_alignment]
                    depth = depth[valid_alignment]
                    rgbs_to_project = rgbs_to_project[valid_alignment]
                    clip_rgbs = clip_rgbs[valid_alignment]

                    if fg_masks is not None:
                        fg_masks = fg_masks[valid_alignment]
                        bg_colors = bg_colors[valid_alignment]

            outputs[ALIGNMENT_LOSS] = alignment_loss
            outputs[VALID_ALIGNMENT] = valid_alignment.float().mean()

        depth_scaling_factor = self.get_depth_scaling_factor(depth.view(cameras.shape[0], -1))

        # Get CLIP embeddings for cross attention
        c_crossattn = self.image_encoder(
            self.clip_normalize(self.clip_resize(clip_rgbs.flatten(0, 1)).clamp(0, 1))).image_embeds
        c_crossattn = c_crossattn.view(cameras.shape[0], clip_rgbs.shape[1], c_crossattn.shape[-1])

        to_project = cameras[:, 1:] if self.training else cameras[:, -1:]
        if self.config.scale_with_pixel_area:
            scale_mult = self.get_pixel_area_scale_mult(to_project.flatten(), project_dim).view(*to_project.shape, -1)
        else:
            scale_mult = 1 / (to_project.fx * project_scale_factor)

        if self.training:
            timesteps = torch.randint(0, self.ddpm_scheduler.config.num_train_timesteps, (cameras.shape[0],),
                                      device=self.device, dtype=torch.long)
            outputs["timesteps"] = timesteps

            target_w2cs_opencv = self.c2ws_opengl_to_w2cs_opencv(cameras.camera_to_worlds[:, :1]).squeeze(1)

            cond_pts3d = pts3d[:, 1:]
            scales = depth[:, 1:].flatten(-2, -1) * scale_mult
            opacity = fg_masks[:, 1:].view(pts3d[..., :1].shape).float() if fg_masks is not None \
                else torch.ones_like(cond_pts3d[..., :1])

            c_concat, _ = self.get_concat(
                target_w2cs_opencv,
                cameras[:, 0],
                cond_pts3d,
                scales,
                opacity,
                rgbs_to_project[:, 1:],
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

                if self.config.predict_with_projection:
                    predict_scale_factor = self.config.image_dim / input_dim
                    predict_focals = torch.cat([cameras.fx[:, 0], cameras.fy[:, 0]], -1) * predict_scale_factor
                    predict_pp = torch.cat([cameras.cx[:, 0], cameras.cy[:, 0]], -1) * predict_scale_factor

                    if self.config.scale_with_pixel_area:
                        predict_scale_mult = self.get_pixel_area_scale_mult(
                            cameras[:, 0], self.config.image_dim).view(cameras.shape[0], -1)
                    else:
                        predict_scale_mult = 1 / (cameras[:, 0].fx * predict_scale_factor)

                    target_c2ws_opencv = torch.eye(4, device=cameras.device).view(1, 4, 4).repeat(
                        target_w2cs_opencv.shape[0], 1, 1)
                    target_c2ws_opencv[:, :3] = cameras.camera_to_worlds[:, 0]
                    target_c2ws_opencv[:, :, 1:3] *= -1  # opengl to opencv

                    with torch.inference_mode():
                        predict_pts3d = self.depth_to_pts3d(depth_gt, target_c2ws_opencv, self.pixels,
                                                            predict_focals, predict_pp)

                        predict_scales = (depth_gt.view(cameras.shape[0], -1) * predict_scale_mult).unsqueeze(-1)
                        predict_opacity = fg_masks[:, 0].view(
                            predict_pts3d[..., 0].shape).float() if fg_masks is not None \
                            else torch.ones_like(predict_pts3d[..., 0])

                        predict_rendering = self.splat_gaussians(
                            target_w2cs_opencv,
                            cameras.fx[:, 0] * predict_scale_factor,
                            cameras.fy[:, 0] * predict_scale_factor,
                            cameras.cx[:, 0] * predict_scale_factor,
                            cameras.cy[:, 0] * predict_scale_factor,
                            predict_pts3d,
                            predict_scales.expand(-1, -1, 3),
                            predict_opacity,
                            image_gt.reshape(cameras.shape[0], -1, 3),
                            self.config.image_dim,
                            True,
                            1,
                            bg_colors,
                        )
                        image_gt = predict_rendering[RGB]
                        depth_gt = predict_rendering[DEPTH]

            else:
                depth_gt = None

            input_gt = self.encode_rgbd(image_gt, depth_gt, depth_scaling_factor)

            noise = torch.randn_like(input_gt)
            noisy_input_gt = self.ddpm_scheduler.add_noise(input_gt, noise, timesteps)
            model_pred = self.unet(torch.cat([noisy_input_gt, c_concat], 1), timesteps,
                                   c_crossattn, return_dict=False)[0]

            if self.config.predict_with_projection:
                # Scale target depth to make loss scale invariant
                outputs["target"] = torch.cat(
                    [image_gt, self.encode_depth(depth_gt, depth_scaling_factor)], -1)

                if self.config.prediction_type == "epsilon":
                    alpha_prod_t = self.ddpm_scheduler.alphas_cumprod.to(self.device)[timesteps].view(-1, 1, 1, 1)
                    beta_prod_t = 1 - alpha_prod_t
                    pred_original_sample = (noisy_input_gt - beta_prod_t ** 0.5 * model_pred) / alpha_prod_t ** 0.5
                elif self.config.prediction_type == "sample":
                    pred_original_sample = model_pred
                elif self.config.prediction_type == "v_prediction":
                    alpha_prod_t = self.ddpm_scheduler.alphas_cumprod.to(self.device)[timesteps].view(-1, 1, 1, 1)
                    beta_prod_t = 1 - alpha_prod_t
                    pred_original_sample = (alpha_prod_t ** 0.5) * noisy_input_gt - (beta_prod_t ** 0.5) * model_pred
                else:
                    raise ValueError(f"Unknown prediction type {self.ddpm_scheduler.config.prediction_type}")

                pred_rgb, pred_depth = self.decode_rgbd(pred_original_sample, depth_scaling_factor)
                pred_pts3d = self.depth_to_pts3d(pred_depth, target_c2ws_opencv, self.pixels, predict_focals,
                                                 predict_pp)
                pred_scales = (pred_depth.view(cameras.shape[0], -1) * predict_scale_mult).unsqueeze(-1)

                decoded_rendering = self.splat_gaussians(
                    target_w2cs_opencv,
                    cameras.fx[:, 0] * predict_scale_factor,
                    cameras.fy[:, 0] * predict_scale_factor,
                    cameras.cx[:, 0] * predict_scale_factor,
                    cameras.cy[:, 0] * predict_scale_factor,
                    torch.cat([cond_pts3d.view(cameras.shape[0], -1, 3), pred_pts3d], 1),
                    torch.cat([scales.view(cameras.shape[0], -1, 1), pred_scales], 1).expand(-1, -1, 3),
                    torch.cat([opacity.view(cameras.shape[0], -1), torch.ones_like(pred_pts3d[..., 0])], 1),
                    torch.cat([rgbs_to_project[:, 1:].reshape(cameras.shape[0], -1, 3),
                               pred_rgb.reshape(cameras.shape[0], -1, 3)], 1),
                    self.config.image_dim,
                    True,
                    1,
                    bg_colors,
                )

                outputs["model_output"] = torch.cat(
                    [decoded_rendering[RGB], self.encode_depth(decoded_rendering[DEPTH], depth_scaling_factor)], -1)
            else:
                if self.ddpm_scheduler.config.prediction_type == "epsilon":
                    outputs["target"] = noise
                elif self.ddpm_scheduler.config.prediction_type == "sample":
                    outputs["target"] = input_gt
                elif self.ddpm_scheduler.config.prediction_type == "v_prediction":
                    outputs["target"] = self.ddpm_scheduler.get_velocity(input_gt, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {self.ddpm_scheduler.config.prediction_type}")

                outputs["model_output"] = model_pred
        else:
            assert cameras.shape[0] == 1, cameras.shape
            target_w2cs_opencv = self.c2ws_opengl_to_w2cs_opencv(cameras.camera_to_worlds[:, :-1]).squeeze(0)
            target_c2ws_opencv = torch.eye(4, device=cameras.device).view(1, 4, 4).repeat(target_w2cs_opencv.shape[0],
                                                                                          1, 1)
            target_c2ws_opencv[:, :3] = cameras.camera_to_worlds[0, :-1]
            target_c2ws_opencv[:, :, 1:3] *= -1  # opengl to opencv
            decode_scale_factor = self.config.image_dim / cameras.width[0, 0].item()
            focals_for_decode = torch.cat([cameras.fx[:, :-1].view(-1, 1), cameras.fy[:, :-1].view(-1, 1)],
                                          -1) * decode_scale_factor
            pp_for_decode = torch.cat([cameras.cx[:, :-1].view(-1, 1), cameras.cy[:, :-1].view(-1, 1)],
                                      -1) * decode_scale_factor

            if self.config.scale_with_pixel_area:
                decode_scale_mult = self.get_pixel_area_scale_mult(
                    cameras[:, :-1].flatten(), self.config.image_dim).view(*cameras[:, :-1].shape, -1)
            else:
                decode_scale_mult = 1 / (cameras[:, :-1].fx * decode_scale_factor)

            if self.config.use_ddim:
                scheduler = self.ddim_scheduler
                scheduler.set_timesteps(self.config.ddim_steps, device=self.device)
            else:
                scheduler = self.ddpm_scheduler

            samples_rgb = []
            samples_depth = []
            sample_order = []
            concat_rgb = []
            concat_depth = []
            concat_acc = []
            cond_pts3d = pts3d[:, -1:]
            opacity = fg_masks[:, -1:].view(pts3d[..., -1:].shape).float() if fg_masks is not None \
                else torch.ones_like(cond_pts3d[..., :1])
            scales = depth[:, -1:].flatten(-2, -1) * scale_mult
            cond_rgbs = rgbs_to_project[:, -1:].view(1, -1, 3)

            if self.config.guidance_scale > 1:
                c_crossattn = torch.cat([torch.zeros_like(c_crossattn), c_crossattn])

            if self.config.use_ema:
                self.ema_unet.to(self.device)
                self.ema_unet.store(self.unet.parameters())
                self.ema_unet.copy_to(self.unet.parameters())
            try:
                while len(sample_order) < target_w2cs_opencv.shape[0]:
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

                    concat_rgb.append(torch.cat([*rendering[RGB]], 1))

                    if not self.config.rgb_only:
                        concat_depth.append(torch.cat([*rendering[DEPTH]], 1))
                        concat_acc.append(torch.cat([*rendering[ACCUMULATION]], 1))

                    acc_sums = rendering[ACCUMULATION].view(target_w2cs_opencv.shape[0], -1).sum(dim=-1)
                    for already_rendered in sample_order:
                        acc_sums[already_rendered] = -1
                    cur_render = acc_sums.argmax()
                    c_concat = c_concat[cur_render:cur_render + 1]

                    inference_pred = randn_tensor((1, 3 if self.config.rgb_only else 4,
                                                   self.config.image_dim, self.config.image_dim),
                                                  device=c_concat.device, dtype=c_concat.dtype)

                    if self.config.guidance_scale > 1:
                        c_concat = torch.cat([torch.zeros_like(c_concat), c_concat])

                    for t_index, t in enumerate(scheduler.timesteps):
                        model_input = torch.cat([inference_pred] * 2) \
                            if self.config.guidance_scale > 1 else inference_pred
                        model_input = scheduler.scale_model_input(model_input, t)
                        model_input = torch.cat([model_input, c_concat], dim=1)
                        model_pred = self.unet(model_input, t, c_crossattn, return_dict=False)[0]
                        if not torch.isfinite(model_pred).all():
                            import pdb; pdb.set_trace()

                        if self.config.guidance_scale > 1:
                            model_pred_uncond, model_pred_cond = model_pred.chunk(2)
                            model_pred = model_pred_uncond + self.config.guidance_scale * \
                                         (model_pred_cond - model_pred_uncond)

                        inference_pred = scheduler.step(model_pred, t, inference_pred, return_dict=False)[0]

                    sample_order.append(cur_render)
                    pred_rgb, pred_depth = self.decode_rgbd(inference_pred, depth_scaling_factor)
                    samples_rgb.append(pred_rgb)

                    if not self.config.rgb_only:
                        samples_depth.append(pred_depth)
                        pred_pts3d = self.depth_to_pts3d(pred_depth,
                                                         target_c2ws_opencv[cur_render:cur_render + 1],
                                                         self.pixels,
                                                         focals_for_decode[cur_render:cur_render + 1],
                                                         pp_for_decode[cur_render:cur_render + 1]).unsqueeze(0)

                        cond_pts3d = torch.cat([cond_pts3d, pred_pts3d], -2)
                        scales = torch.cat(
                            [scales, pred_depth.view(1, 1, -1) * decode_scale_mult[:, cur_render:cur_render + 1]],
                            -1)
                        opacity = torch.cat([opacity, torch.ones_like(pred_pts3d[..., :1])], -2)
                        cond_rgbs = torch.cat([cond_rgbs, pred_rgb.view(1, -1, 3)], 1)
            finally:
                if self.config.use_ema:
                    self.ema_unet.restore(self.unet.parameters())

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

            concat_rgb.append(torch.cat([*rendering[RGB]], 1))
            outputs[RGB] = torch.cat(concat_rgb)
            sorted_rgb_samples = sorted(zip(sample_order, samples_rgb))
            outputs["samples_rgb"] = torch.cat([x[1].squeeze(0) for x in sorted_rgb_samples], 1)

            depth_gt = depth.squeeze(0).unsqueeze(1)
            if depth_gt.shape[-1] != self.config.image_dim:
                with torch.inference_mode():
                    depth_gt = self.unet_resize(depth_gt).clamp_min(0)
            outputs[DEPTH_GT] = depth_gt.permute(0, 2, 3, 1)

            if not self.config.rgb_only:
                concat_depth.append(torch.cat([*rendering[DEPTH]], 1))
                concat_acc.append(torch.cat([*rendering[ACCUMULATION]], 1))
                outputs[DEPTH] = torch.cat(concat_depth)
                outputs[ACCUMULATION] = torch.cat(concat_acc)
                sorted_depth_samples = sorted(zip(sample_order, samples_depth))
                outputs["samples_depth"] = torch.cat([x[1].squeeze(0) for x in sorted_depth_samples], 1)

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
        if self.config.predict_with_projection:
            loss_dict["lpips_loss"] = self.config.lpips_weight * self.lpips(
                model_output[..., :3].permute(0, 3, 1, 2).clamp(0, 1),
                target[..., :3].permute(0, 3, 1, 2).clamp(0, 1))

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
        for key, val in self.dust3r_field.state_dict().items():
            dict[f"dust3r_field.{key}"] = val

        super().load_state_dict(dict, **kwargs)
