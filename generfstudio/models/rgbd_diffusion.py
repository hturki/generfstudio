from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union, Type, Optional, Literal

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, UNet2DConditionModel, EMAModel, DDIMScheduler, DDPMScheduler
from diffusers.utils.torch_utils import randn_tensor
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.utils import colormaps, profiler
from nerfstudio.utils.comms import get_world_size
from nerfstudio.utils.rich_utils import CONSOLE
from torch import nn
from torch.nn import Parameter
from transformers import CLIPVisionModelWithProjection

from generfstudio.generfstudio_constants import RGB, ACCUMULATION, DEPTH, ALIGNMENT_LOSS, PTS3D, VALID_ALIGNMENT, \
    DEPTH_GT, FG_MASK, BG_COLOR
from generfstudio.models.rgbd_diffusion_base import RGBDDiffusionBaseConfig, RGBDDiffusionBase


@dataclass
class RGBDDiffusionConfig(RGBDDiffusionBaseConfig):
    _target: Type = field(default_factory=lambda: RGBDDiffusion)

    unet_pretrained_path: str = "Intel/ldm3d-4c"
    vae_pretrained_path: str = "Intel/ldm3d-4c"

    beta_schedule: str = "scaled_linear"

    infer_with_projection: bool = False
    rgbd_concat_strategy: Literal["mode", "sample", "mean_std", "downsample"] = "sample"


class RGBDDiffusion(RGBDDiffusionBase):
    config: RGBDDiffusionConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        self.vae = AutoencoderKL.from_pretrained(self.config.vae_pretrained_path, subfolder="vae")
        self.vae.requires_grad_(False)
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(self.config.image_encoder_pretrained_path,
                                                                           subfolder="image_encoder")
        self.image_encoder.requires_grad_(False)

        # 4 for latent + 4 for RGBD + 1 for accumulation + feature dim
        unet_base = UNet2DConditionModel.from_pretrained(self.config.unet_pretrained_path, subfolder="unet",
                                                         low_cpu_mem_usage=False,
                                                         ignore_mismatched_sizes=False)

        concat_channels = 5  # RGBD or VAE + acc mask = 4 + 1 = 5
        if self.config.rgbd_concat_strategy == "mean_std":
            concat_channels += 4

        self.unet = UNet2DConditionModel.from_pretrained(self.config.unet_pretrained_path, subfolder="unet",
                                                         in_channels=4 + concat_channels,
                                                         low_cpu_mem_usage=False,
                                                         ignore_mismatched_sizes=True)
        old_state = unet_base.state_dict()
        new_state = self.unet.state_dict()
        in_key = "conv_in.weight"
        # Check if we need to port weights from 4ch input
        if old_state[in_key].shape != new_state[in_key].shape:
            CONSOLE.log(f"Manual init: {in_key}")
            new_state[in_key].zero_()
            new_state[in_key][:, :old_state[in_key].shape[1], :, :].copy_(old_state[in_key])
            self.unet.load_state_dict(new_state)

        self.unet.enable_xformers_memory_efficient_attention()

        if self.config.use_ema:
            self.ema_unet = EMAModel(self.unet.parameters(), model_cls=UNet2DConditionModel,
                                     model_config=self.unet.config)

        if self.config.use_ddim and self.config.infer_with_projection:
            self.ddim_scheduler.config["prediction_type"] = "sample"
            self.ddim_scheduler = DDIMScheduler.from_config(self.ddim_scheduler.config)
            CONSOLE.log(f"DDIM Scheduler: {self.ddim_scheduler.config}")
        elif self.config.infer_with_projection:
            self.ddpm_scheduler.config["prediction_type"] = "sample"
            self.ddpm_scheduler = DDPMScheduler.from_config(self.ddpm_scheduler.config)

        if get_world_size() > 1:
            CONSOLE.log("Using sync batchnorm for DDP")
            self.unet = nn.SyncBatchNorm.convert_sync_batchnorm(self.unet)
            self.depth_estimator_field = nn.SyncBatchNorm.convert_sync_batchnorm(self.depth_estimator_field)

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        return {"fields": list(self.unet.parameters())}

    @profiler.time_function
    def encode_with_vae(self, rgb: torch.Tensor, depth: Optional[torch.Tensor], depth_scaling_factor: torch.Tensor,
                        sample_override: bool = False) -> torch.Tensor:
        to_encode = rgb.permute(0, 3, 1, 2) * 2 - 1
        if not self.config.rgb_only:
            encoded_depth = self.encode_depth(depth, depth_scaling_factor).permute(0, 3, 1, 2)
            to_encode = torch.cat([to_encode, encoded_depth], 1)

        return self.encode_with_vae_inner(to_encode, sample_override)

    def encode_with_vae_inner(self, to_encode: torch.Tensor, sample_override: bool = False,
                              autocast_enabled: bool = True) -> torch.Tensor:
        with torch.cuda.amp.autocast(enabled=autocast_enabled):
            latent_dist = self.vae.encode(to_encode).latent_dist
            if sample_override or self.config.rgbd_concat_strategy == "sample":
                encoded = latent_dist.sample() * self.vae.config.scaling_factor
            elif self.config.rgbd_concat_strategy == "mode":
                encoded = latent_dist.mode() * self.vae.config.scaling_factor
            elif self.config.rgbd_concat_strategy == "mean_std":
                encoded = torch.cat([latent_dist.mean, latent_dist.std], 1) * self.vae.config.scaling_factor
            else:
                raise Exception(self.config.rgbd_concat_strategy)

        if autocast_enabled:
            to_upcast = torch.logical_not(torch.isfinite(encoded.view(encoded.shape[0], -1)).all(dim=-1))
            if to_upcast.any():
                encoded[to_upcast] = self.encode_with_vae_inner(to_encode[to_upcast], sample_override, False)

        return encoded

    @profiler.time_function
    def decode_with_vae(self, to_decode: torch.Tensor, depth_scaling_factor: torch.Tensor) -> Tuple[
        torch.Tensor, Optional[torch.Tensor]]:
        decoded = self.decode_with_vae_inner(to_decode).permute(0, 2, 3, 1)
        rgb = (0.5 * (decoded[..., :3] + 1)).clamp(0, 1)
        if self.config.rgb_only:
            return rgb, None

        return rgb, self.decode_depth(decoded[..., 3:], depth_scaling_factor)

    def decode_with_vae_inner(self, to_decode: torch.Tensor, autocast_enabled: bool = True) -> torch.Tensor:
        with torch.cuda.amp.autocast(enabled=autocast_enabled):
            decoded = self.vae.decode(to_decode / self.vae.config.scaling_factor, return_dict=False)[0]

        if autocast_enabled:
            to_upcast = torch.logical_not(torch.isfinite(decoded.view(decoded.shape[0], -1)).all(dim=-1))
            if to_upcast.any():
                decoded[to_upcast] = self.decode_with_vae_inner(decoded[to_upcast].float(), False)

        return decoded

    @profiler.time_function
    def get_concat(self, w2cs: torch.Tensor, cameras: Cameras, pts3d: torch.Tensor, scales: torch.Tensor,
                   opacity: torch.Tensor, rgbs: torch.Tensor, cameras_per_scene: int, bg_colors: Optional[int],
                   depth_scaling_factor: torch.Tensor):
        camera_dim = cameras.width[0, 0].item()
        fx = cameras.fx.view(w2cs.shape[0], 1)
        fy = cameras.fy.view(w2cs.shape[0], 1)
        cx = cameras.cx.view(w2cs.shape[0], 1)
        cy = cameras.cy.view(w2cs.shape[0], 1)
        if self.config.rgbd_concat_strategy == "downsample":
            camera_dim = camera_dim / self.vae_scale_factor
            fx = fx / self.vae_scale_factor
            fy = fy / self.vae_scale_factor
            cx = cx / self.vae_scale_factor
            cy = cy / self.vae_scale_factor

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

        rendered_rgb = rendering[RGB]
        rendered_accumulation = rendering[ACCUMULATION].permute(0, 3, 1, 2)

        if self.config.rgb_only:
            if self.config.rgbd_concat_strategy != "downsample":
                concat_list = self.encode_with_vae(rendered_rgb, None, depth_scaling_factor)
            else:
                concat_list = [rendered_rgb.permute(0, 3, 1, 2) * 2 - 1, rendered_accumulation]
        else:
            rendered_depth = rendering[DEPTH]
            if self.config.rgbd_concat_strategy != "downsample":
                encoded = self.encode_with_vae(rendered_rgb, rendered_depth, depth_scaling_factor)

                encoded_accumulation = F.interpolate(rendered_accumulation, encoded.shape[2:],
                                                     mode="bicubic", antialias=True).clamp_min(0)

                concat_list = [encoded, encoded_accumulation]
            else:
                concat_list = [
                    rendered_rgb.permute(0, 3, 1, 2) * 2 - 1,
                    self.encode_depth(rendered_depth, depth_scaling_factor).permute(0, 3, 1, 2),
                    rendered_accumulation]

        return torch.cat(concat_list, 1), rendering

    def get_outputs(self, cameras: Cameras) -> Dict[str, Union[torch.Tensor, List]]:
        outputs = {}
        images = cameras.metadata["image"]
        image_dim = cameras.width[0, 0].item()
        pts3d = cameras.metadata.get(PTS3D, None)
        depth = cameras.metadata.get(DEPTH, None)
        # fg_masks = cameras.metadata.get(FG_MASK, None)
        bg_colors = cameras.metadata.get(BG_COLOR, None)

        cameras.metadata = {}  # Should do deepcopy instead if we still need any metadata here

        if pts3d is not None:
            clip_rgbs = (images[:, 1:] if self.training else images[:, -1:]).permute(0, 1, 4, 2, 3)
            # if fg_masks is None:
            fg_masks = torch.ones_like(pts3d[..., :1])

            valid_alignment = torch.ones(cameras.shape[0], dtype=torch.bool, device=cameras.device)
        else:
            rgbs_chw = images.permute(0, 1, 4, 2, 3)
            clip_rgbs = (rgbs_chw[:, 1:] if self.training else rgbs_chw[:, -1:])

            c2ws_opencv = torch.eye(4, device=cameras.device).view(1, 1, 4, 4).repeat(cameras.shape[0],
                                                                                      cameras.shape[1], 1, 1)
            c2ws_opencv[:, :, :3] = cameras.camera_to_worlds
            c2ws_opencv[:, :, :, 1:3] *= -1  # opengl to opencv

            fg_masks = torch.ones_like(pts3d[..., :1])
            pts3d, depth, alignment_loss, valid_alignment, confs = self.depth_estimator_field.get_pts3d_and_depth(
                rgbs_chw, fg_masks, c2ws_opencv, cameras.fx, cameras.fy, cameras.cx, cameras.cy)

            outputs[ALIGNMENT_LOSS] = alignment_loss
            outputs[VALID_ALIGNMENT] = valid_alignment.float().mean()

            depth = depth.clamp_min(0).view(*cameras.shape[:2], *rgbs_chw.shape[3:])

            fg_masks = (confs > self.config.min_conf_thresh).view(*cameras.shape[:2], *rgbs_chw.shape[3:])
            # conf_mask = (confs > self.config.min_conf_thresh).view(*cameras.shape[:2], *rgbs_chw.shape[3:])
            # fg_masks = torch.logical_and(fg_masks, conf_mask) if fg_masks is not None else conf_mask
            pts3d = pts3d.view(*cameras.shape[:2], *pts3d.shape[1:])

        depth_scaling_factor = self.get_depth_scaling_factor(depth.view(cameras.shape[0], -1))

        with torch.inference_mode():
            # Get CLIP embeddings for cross attention
            c_crossattn = self.image_encoder(
                self.clip_normalize(self.clip_resize(clip_rgbs.flatten(0, 1)).clamp(0, 1))).image_embeds
            c_crossattn = c_crossattn.view(cameras.shape[0], clip_rgbs.shape[1], c_crossattn.shape[-1])

        to_project = cameras[:, 1:] if self.training else cameras
        if self.config.scale_with_pixel_area:
            scale_mult = self.get_pixel_area_scale_mult(to_project.flatten(), image_dim).view(*to_project.shape, -1)
        else:
            scale_mult = 1 / to_project.fx

        if self.training:
            timesteps = torch.randint(0, self.ddpm_scheduler.config.num_train_timesteps, (cameras.shape[0],),
                                      device=self.device, dtype=torch.long)
            outputs["timesteps"] = timesteps

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

            # Hack - we include the gt image in the forward loop, so that all model calculation is done in the
            # forward call as expected in DDP
            rgb_to_encode = images[:, 0]
            if not self.config.rgb_only:
                depth_to_encode = depth[:, :1].permute(0, 2, 3, 1)
            else:
                depth_to_encode = None

            latents = self.encode_with_vae(rgb_to_encode, depth_to_encode, depth_scaling_factor, sample_override=True)
            noise = torch.randn_like(latents)

            if self.ddpm_scheduler.config.prediction_type == "epsilon":
                outputs["target"] = noise
            elif self.ddpm_scheduler.config.prediction_type == "sample":
                outputs["target"] = latents
            elif self.ddpm_scheduler.config.prediction_type == "v_prediction":
                outputs["target"] = self.ddpm_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {self.ddpm_scheduler.config.prediction_type}")

            noisy_latents = self.ddpm_scheduler.add_noise(latents, noise, timesteps)
            model_output = self.unet(torch.cat([noisy_latents, c_concat], 1), timesteps, c_crossattn).sample
            # if self.config.predict_with_projection:
            #     decoded_rgb, decoded_depth = self.decode_with_vae(model_output, depth_scaling_factor)
            #     decoded_pts3d = self.depth_to_pts3d(decoded_depth, target_c2ws_opencv, pixels_for_encode,
            #                                         focals_for_encode, pp_for_encode)
            #     decoded_scales = (decoded_depth.view(cameras.shape[0], -1) * to_encode_scale_mult).unsqueeze(-1)
            #
            #     decoded_rendering = self.splat_gaussians(
            #         target_w2cs_opencv,
            #         cameras.fx[:, 0],
            #         cameras.fy[:, 0],
            #         cameras.cx[:, 0],
            #         cameras.cy[:, 0],
            #         torch.cat([cond_pts3d.view(cameras.shape[0], -1, 3), decoded_pts3d], 1),
            #         torch.cat([scales.view(cameras.shape[0], -1, 1), decoded_scales], 1).expand(-1, -1, 3),
            #         torch.cat([opacity.view(cameras.shape[0], -1), torch.ones_like(decoded_pts3d[..., 0])], 1),
            #         torch.cat([rgbs_to_project[:, 1:].reshape(cameras.shape[0], -1, 3),
            #                    decoded_rgb.reshape(cameras.shape[0], -1, 3)], 1),
            #         image_dim,
            #         True,
            #         1,
            #         bg_colors,
            #     )
            #
            #     outputs["model_output"] = torch.cat(
            #         [decoded_rendering[RGB], self.encode_depth(decoded_rendering[DEPTH], depth_scaling_factor)], -1)
            # else:
            outputs["model_output"] = model_output
        else:
            assert cameras.shape[0] == 1, cameras.shape
            target_w2cs_opencv = self.c2ws_opengl_to_w2cs_opencv(cameras.camera_to_worlds[:, :-1]).squeeze(0)
            target_c2ws_opencv = torch.eye(4, device=cameras.device).view(1, 4, 4).repeat(target_w2cs_opencv.shape[0],
                                                                                          1, 1)
            target_c2ws_opencv[:, :3] = cameras.camera_to_worlds[0, :-1]
            target_c2ws_opencv[:, :, 1:3] *= -1  # opengl to opencv
            focals_for_decode = torch.cat([cameras.fx[:, :-1].view(-1, 1), cameras.fy[:, :-1].view(-1, 1)], -1)
            pp_for_decode = torch.cat([cameras.cx[:, :-1].view(-1, 1), cameras.cy[:, :-1].view(-1, 1)], -1)

            pixels_im = torch.stack(
                torch.meshgrid(
                    torch.arange(image_dim, dtype=torch.float32, device=cameras.device),
                    torch.arange(image_dim, dtype=torch.float32, device=cameras.device),
                    indexing="ij")).permute(2, 1, 0)
            pixels_for_decode = pixels_im.reshape(-1, 2)

            if self.config.scale_with_pixel_area:
                decode_scale_mult = self.get_pixel_area_scale_mult(cameras[:, :-1].flatten(), image_dim).view(
                    *cameras[:, :-1].shape, -1)
            else:
                decode_scale_mult = 1 / cameras[:, :-1].fx

            if self.config.use_ddim:
                scheduler = self.ddim_scheduler
                scheduler.set_timesteps(self.config.inference_steps, device=self.device)
            else:
                scheduler = self.ddpm_scheduler

            samples_rgb = []
            samples_depth = []
            sample_order = []
            concat_rgb = []
            concat_depth = []
            concat_acc = []
            cond_pts3d = pts3d[:, -1:]
            opacity = fg_masks[:, -1:].view(cond_pts3d[..., :1].shape).float()
            scales = depth[:, -1:].flatten(-2, -1) * scale_mult
            cond_rgbs = images[:, -1:].view(1, -1, 3)

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

                    sample = randn_tensor((1, self.vae.config.latent_channels,
                                           image_dim // self.vae_scale_factor,
                                           image_dim // self.vae_scale_factor),
                                          device=c_concat.device,
                                          dtype=c_concat.dtype)

                    if self.config.guidance_scale > 1:
                        c_concat = torch.cat([torch.zeros_like(c_concat), c_concat])

                    for t_index, t in enumerate(scheduler.timesteps):
                        model_input = torch.cat([sample] * 2) if self.config.guidance_scale > 1 else sample
                        model_input = scheduler.scale_model_input(model_input, t)
                        model_input = torch.cat([model_input, c_concat], dim=1)
                        model_output = self.unet(model_input, t, c_crossattn, return_dict=False)[0]

                        if self.config.guidance_scale > 1:
                            model_output_uncond, model_output_cond = model_output.chunk(2)
                            model_output = model_output_uncond + self.config.guidance_scale * \
                                           (model_output_cond - model_output_uncond)

                        if self.config.infer_with_projection:
                            alpha_prod_t = scheduler.alphas_cumprod.to(self.device)[t].view(-1, 1, 1, 1)
                            beta_prod_t = 1 - alpha_prod_t
                            if self.config.prediction_type == "epsilon":
                                pred_original_sample = (sample - beta_prod_t ** (
                                    0.5) * model_output) / alpha_prod_t ** (
                                                           0.5)
                            elif self.config.prediction_type == "sample":
                                pred_original_sample = model_output
                            elif self.config.prediction_type == "v_prediction":
                                pred_original_sample = (alpha_prod_t ** 0.5) * sample - (
                                        beta_prod_t ** 0.5) * model_output
                            else:
                                raise ValueError(f"Unknown prediction type {self.config.prediction_type}")

                            pred_rgb, pred_depth = self.decode_with_vae(pred_original_sample, depth_scaling_factor)

                            blend_weight = rendering[ACCUMULATION][cur_render:cur_render + 1].clone()
                            other_depth = rendering[DEPTH][cur_render:cur_render + 1]
                            blend_weight[other_depth > pred_depth] = 0

                            model_output = self.encode_with_vae(
                                (1 - blend_weight) * pred_rgb + blend_weight * rendering[RGB][
                                                                               cur_render:cur_render + 1],
                                (1 - blend_weight) * pred_depth + blend_weight * other_depth,
                                depth_scaling_factor, sample_override=True)

                        sample = scheduler.step(model_output, t, sample, return_dict=False)[0]

                    sample_order.append(cur_render)
                    decoded_rgb, decoded_depth = self.decode_with_vae(sample, depth_scaling_factor)
                    samples_rgb.append(decoded_rgb)

                    if not self.config.rgb_only:
                        samples_depth.append(decoded_depth)
                        decoded_pts3d = self.depth_to_pts3d(decoded_depth,
                                                            target_c2ws_opencv[cur_render:cur_render + 1],
                                                            pixels_for_decode,
                                                            focals_for_decode[cur_render:cur_render + 1],
                                                            pp_for_decode[cur_render:cur_render + 1]).unsqueeze(0)

                        cond_pts3d = torch.cat([cond_pts3d, decoded_pts3d], -2)
                        scales = torch.cat(
                            [scales, decoded_depth.view(1, 1, -1) * decode_scale_mult[:, cur_render:cur_render + 1]],
                            -1)
                        opacity = torch.cat([opacity, torch.ones_like(decoded_pts3d[..., :1])], -2)
                        cond_rgbs = torch.cat([cond_rgbs, decoded_rgb.view(1, -1, 3)], 1)
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
            if depth_gt.shape[-1] != image_dim:
                with torch.inference_mode():
                    depth_gt = F.interpolate(depth_gt, image_dim, mode="bicubic", antialias=True).clamp_min(0)
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

        loss = F.mse_loss(model_output.float(), target.float(), reduction="mean")
        assert torch.isfinite(loss).all(), loss

        if VALID_ALIGNMENT in outputs and outputs[VALID_ALIGNMENT] == 0:
            CONSOLE.log("No valid outputs, ignoring training batch")
            loss = loss * 0  # Do this so that backprop doesn't complain

        return {"loss": loss}

    @profiler.time_function
    def get_image_metrics_and_images(
            self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        image_gt = batch["image"].squeeze(0).to(outputs[RGB].device)
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
        for key, val in self.depth_estimator_field.state_dict().items():
            dict[f"depth_estimator_field.{key}"] = val

        for key, val in self.image_encoder.state_dict().items():
            dict[f"image_encoder.{key}"] = val

        for key, val in self.vae.state_dict().items():
            dict[f"vae.{key}"] = val

        super().load_state_dict(dict, **kwargs)

    # def state_dict(self, *args, **kwargs):
    #     base_state_dict = super().state_dict(*args, **kwargs)
    #
    #     state_dict = {}
    #     for key, val in base_state_dict.items():
    #         if "vae." in key or "image_encoder." in key:
    #             print("SKIP", key)
    #             continue
    #         state_dict[key] = val
    #
    #     import pdb; pdb.set_trace()
    #
    #     return state_dict
