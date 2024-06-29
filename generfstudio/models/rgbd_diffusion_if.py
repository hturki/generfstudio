from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union, Type, Optional, Literal, Any

import torch
import torch.nn.functional as F
from diffusers import UNet2DConditionModel, DDPMScheduler, EMAModel, DDIMScheduler
from diffusers.utils.torch_utils import randn_tensor
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.engine.callbacks import TrainingCallbackAttributes, TrainingCallback, TrainingCallbackLocation
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps, profiler
from nerfstudio.utils.comms import get_world_size
from nerfstudio.utils.rich_utils import CONSOLE
from pytorch_msssim import SSIM
from torch import nn
from torch.nn import Parameter
from torch_scatter import scatter_min
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchvision.transforms import transforms
from transformers import CLIPVisionModelWithProjection

from generfstudio.fields.batched_pc_optimizer import fast_depthmap_to_pts3d
from generfstudio.fields.dino_v2_encoder import DinoV2Encoder
from generfstudio.fields.dust3r_field import Dust3rField
from generfstudio.fields.spatial_encoder import SpatialEncoder
from generfstudio.generfstudio_constants import NEIGHBOR_IMAGES, NEIGHBOR_CAMERAS, RGB, ACCUMULATION, DEPTH, \
    ALIGNMENT_LOSS, PTS3D, VALID_ALIGNMENT, \
    NEIGHBOR_DEPTH, DEPTH_GT, NEIGHBOR_FG_MASK, FG_MASK, BG_COLOR
from generfstudio.models.rgbd_diffusion import RGBDDiffusion


@dataclass
class RGBDDiffusionIFConfig(ModelConfig):
    _target: Type = field(default_factory=lambda: RGBDDiffusionIF)

    uncond: float = 0.05
    ddim_steps = 50

    cond_image_encoder_type: Literal["unet", "resnet", "dino"] = "resnet"
    cond_feature_out_dim: int = 0
    freeze_cond_image_encoder: bool = False

    unet_pretrained_path: str = "DeepFloyd/IF-I-L-v1.0"
    image_encoder_pretrained_path: str = "lambdalabs/sd-image-variations-diffusers"
    scheduler_pretrained_path: str = "DeepFloyd/IF-I-L-v1.0"

    beta_schedule: str = "squaredcos_cap_v2"
    prediction_type: str = "epsilon"

    dust3r_model_name: str = "/data/hturki/dust3r/checkpoints/DUSt3R_ViTLarge_BaseDecoder_224_linear.pth"
    noisy_cond_views: int = 0
    calc_compensations: bool = False
    independent_noise: bool = False
    noise_after_reproj: bool = False
    project_overlay: bool = False

    opacity_multiplier: float = 50
    image_crossattn: Literal["clip-expand", "clip-replace", "unet-replace"] = "clip-expand"
    depth_mapping: Literal["simple", "disparity", "log"] = "log"
    center_depth: bool = False

    scale_with_pixel_area: bool = True

    guidance_scale: float = 2.0
    image_dim: int = 64
    image_dim_project: Optional[int] = None

    use_ema: bool = True
    allow_tf32: bool = False
    rgb_only: bool = False


class RGBDDiffusionIF(Model):
    config: RGBDDiffusionIFConfig

    def __init__(self, config: RGBDDiffusionIFConfig, metadata: Dict[str, Any], **kwargs) -> None:
        if config.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

        self.depth_available = DEPTH in metadata
        CONSOLE.log(f"Depth available: {self.depth_available}")

        super().__init__(config=config, **kwargs)

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()
        self.collider = None

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

        self.image_encoder_resize = transforms.Resize(
            (224, 224),
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=False,
        )
        self.image_encoder_normalize = transforms.Normalize(
            [0.48145466, 0.4578275, 0.40821073],
            [0.26862954, 0.26130258, 0.27577711])
        unet_base = UNet2DConditionModel.from_pretrained(self.config.unet_pretrained_path, subfolder="unet",
                                                         low_cpu_mem_usage=False,
                                                         ignore_mismatched_sizes=False)

        # 4 for noise + 4 for RGBD cond + 1 for accumulation
        unet_in_channels = 4 + 4 + 1
        self.unet = UNet2DConditionModel.from_pretrained(self.config.unet_pretrained_path, subfolder="unet",
                                                         in_channels=unet_in_channels,
                                                         out_channels=4,
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
        # Check if we need to port weights from 4ch input
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
            ema_unet = UNet2DConditionModel.from_pretrained(self.config.unet_pretrained_path, subfolder="unet",
                                                            in_channels=unet_in_channels,
                                                            out_channels=4,
                                                            low_cpu_mem_usage=False,
                                                            ignore_mismatched_sizes=True,
                                                            encoder_hid_dim=768 if self.config.image_crossattn == "unet-replace" else unet_base.config.encoder_hid_dim,
                                                            )
            ema_unet.load_state_dict(new_state)

            self.ema_unet = EMAModel(ema_unet.parameters(), model_cls=UNet2DConditionModel,
                                     model_config=ema_unet.config)

        self.ddpm_scheduler = DDPMScheduler.from_pretrained(self.config.scheduler_pretrained_path,
                                                            subfolder="scheduler",
                                                            beta_schedule=self.config.beta_schedule,
                                                            prediction_type=self.config.prediction_type,
                                                            variance_type="fixed_small")
        ddim_scheduler = DDIMScheduler.from_pretrained(self.config.scheduler_pretrained_path,
                                                       subfolder="scheduler",
                                                       beta_schedule=self.config.beta_schedule,
                                                       prediction_type=self.config.prediction_type)
        # Need to explicitly set this for DDIMScheduler for some reason
        ddim_scheduler.config["variance_type"] = "fixed_small"
        self.ddim_scheduler = DDIMScheduler.from_config(ddim_scheduler.config)
        CONSOLE.log(f"DDIM Scheduler: {self.ddim_scheduler.config}")

        self.dust3r_field = Dust3rField(model_name=self.config.dust3r_model_name,
                                        depth_precomputed=self.depth_available)

        # metrics
        # self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        # self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)
        # self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        # for p in self.lpips.parameters():
        #     p.requires_grad_(False)

        if self.config.noisy_cond_views > 0:
            pixels_im = torch.stack(
                torch.meshgrid(
                    torch.arange(self.config.image_dim, dtype=torch.float32),
                    torch.arange(self.config.image_dim, dtype=torch.float32),
                    indexing="ij")).permute(2, 1, 0)
            self.register_buffer("pixels", pixels_im.reshape(-1, 2), persistent=False)

        if get_world_size() > 1:
            CONSOLE.log("Using sync batchnorm for DDP")
            self.unet = nn.SyncBatchNorm.convert_sync_batchnorm(self.unet)
            self.dust3r_field = nn.SyncBatchNorm.convert_sync_batchnorm(self.dust3r_field)

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        # param_groups["cond_encoder"] = list(self.dust3r_field.parameters())

        param_groups["fields"] = list(self.unet.parameters())
        if self.config.image_crossattn != "unet-replace":
            param_groups["fields"] += list(self.image_encoder.visual_projection.parameters())

        return param_groups

    def get_training_callbacks(
            self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        def set_ema(step: int):
            if self.config.use_ema:
                self.ema_unet.to(self.device)
                self.ema_unet.step(self.unet.parameters())

        return [
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                func=set_ema,
                update_every_num_iters=1,
            ),
        ]

    def encode_depth(self, to_encode: torch.Tensor) -> torch.Tensor:
        if self.config.depth_mapping == "simple":
            encoded_0_1 = to_encode
        elif self.config.depth_mapping == "disparity":
            encoded_0_1 = 1 - 1 / (1 + to_encode.float())
        elif self.config.depth_mapping == "log":
            encoded_0_1 = to_encode.float().clamp_min(1e-8).log()
        else:
            raise Exception(self.config.depth_mapping)

        if self.config.center_depth:
            return encoded_0_1 * 2 - 1
        else:
            return encoded_0_1

    def decode_depth(self, to_decode_m1_1: torch.Tensor) -> torch.Tensor:
        if self.config.center_depth:
            to_decode = (to_decode_m1_1 + 1) / 2
        else:
            to_decode = to_decode_m1_1

        if self.config.depth_mapping == "simple":
            scaled = to_decode.clamp_min(0)
        elif self.config.depth_mapping == "disparity":
            scaled = 1 / (1 - to_decode.float().clamp(0, 0.999)) - 1
        elif self.config.depth_mapping == "log":
            scaled = to_decode.float().exp()
        else:
            raise Exception(self.config.depth_mapping)

        return scaled

    @profiler.time_function
    @torch.inference_mode()
    def get_concat(self, w2cs: torch.Tensor, cameras: Cameras, pts3d: torch.Tensor, scales: torch.Tensor,
                   opacity: torch.Tensor, rgbs: torch.Tensor, cameras_per_scene: int, bg_colors: Optional[torch.Tensor],
                   depth_scale_factor: torch.Tensor, calc_compensations: bool = False, normalize_rgb: bool = False,
                   opacity_multiplier: float = 1):
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
        opacity = opacity.view(S, -1).float() * opacity_multiplier
        # if calc_compensations:
        #     opacity = opacity * 20  # Hack, otherwise opacity seems to be too small

        rendering = RGBDDiffusion.splat_gaussians(
            w2cs,
            fx,
            fy,
            cx,
            cy,
            pts3d.view(S, -1, 3).float(),
            scales.view(S, -1, 1).expand(-1, -1, 3),
            opacity,
            rgbs.permute(0, 1, 3, 4, 2).reshape(S, -1, 3),
            camera_dim,
            not self.config.rgb_only,
            cameras_per_scene,
            bg_colors,
            calc_compensations=calc_compensations,
            normalize_rgb=normalize_rgb,
        )

        if self.config.image_dim_project is not None:
            for key in rendering:
                rendering[key] = F.interpolate(rendering[key].permute(0, 3, 1, 2), self.config.image_dim,
                                               mode="bicubic", antialias=True).permute(0, 2, 3, 1)

        rendered_rgb = rendering[RGB].permute(0, 3, 1, 2)
        rendered_accumulation = rendering[ACCUMULATION].permute(0, 3, 1, 2)

        if self.config.rgb_only:
            concat_list = [rendered_rgb * 2 - 1, rendered_accumulation]
        else:
            encoded_depth = (self.encode_depth(rendering[DEPTH] / depth_scale_factor)).permute(0, 3, 1, 2)
            concat_list = [rendered_rgb * 2 - 1, encoded_depth, rendered_accumulation]

        return torch.cat(concat_list, 1), rendering

    @profiler.time_function
    @torch.inference_mode()
    def overlay_encoded_rgbd(self, encoded: torch.Tensor, render_cameras: Cameras, c2ws: torch.Tensor,
                             w2cs: torch.Tensor, depth_fx: torch.Tensor, depth_fy: torch.Tensor, depth_cx: torch.Tensor,
                             depth_cy: torch.Tensor, fg_masks: Optional[torch.Tensor],
                             depth_scale_factor: torch.Tensor):
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
            depth_map = depth[:, 0].permute(0, 2, 3, 1).clone()
            indices = (camera_index * image_dim * image_dim + v * image_dim + u).view(-1)
            _, min_indices = scatter_min(z.view(-1), indices, out=depth_map.view(-1))

            rgb_to_fill = min_indices[min_indices <= indices.shape[0] - 1]
            rgb_map = rgb[:, 0].permute(0, 2, 3, 1).reshape(-1, 3).clone()
            rgb_map[indices[rgb_to_fill]] = rgb.permute(0, 1, 3, 4, 2)[:, 1:].reshape(-1, 3)[rgb_to_fill]
            rgb_map = rgb_map.view(*depth_map.shape[:-1], 3).permute(0, 3, 1, 2)

            if self.config.rgb_only:
                concat_list = [rgb_map * 2 - 1]
            else:
                encoded_depth = (self.encode_depth(depth_map / depth_scale_factor)).permute(0, 3, 1, 2)
                concat_list = [rgb_map * 2 - 1, encoded_depth]

            return torch.cat(concat_list, 1)

    @profiler.time_function
    @torch.inference_mode()
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

    def get_pixel_area_scale_mult(self, cameras: Cameras, camera_dim: int):
        if cameras.width[0, 0].item() != camera_dim:
            cameras = copy.deepcopy(cameras)
            cameras.rescale_output_resolution(camera_dim / cameras.width[0, 0].item())

        ray_bundle = cameras.generate_rays(camera_indices=torch.arange(len(cameras)).view(-1, 1))

        pixel_area = ray_bundle.pixel_area.squeeze(-1).permute(2, 0, 1)
        directions_norm = ray_bundle.metadata["directions_norm"].squeeze(-1).permute(2, 0, 1)
        return pixel_area * directions_norm

    def get_outputs(self, cameras: Cameras) -> Dict[str, Union[torch.Tensor, List]]:
        image_gt = cameras.metadata["image"]
        neighbor_cameras = cameras.metadata[NEIGHBOR_CAMERAS]
        bg_colors = cameras.metadata.get(BG_COLOR, None)

        noisy_cond_views = self.config.noisy_cond_views

        if noisy_cond_views > 0:
            neighbor_images = cameras.metadata[NEIGHBOR_IMAGES]

        if self.training:
            camera_w2cs_opencv = torch.eye(4, device=cameras.device).unsqueeze(0).repeat(cameras.shape[0], 1, 1)

            with torch.cuda.amp.autocast(enabled=False):
                R = cameras.camera_to_worlds[:, :3, :3].clone()  # 3 x 3, clone to avoid changing original cameras
                R[:, :, 1:3] *= -1  # opengl to opencv
                T = cameras.camera_to_worlds[:, :3, 3:]  # 3 x 1

                R_inv = R[:, :3, :3].transpose(1, 2)
                camera_w2cs_opencv[:, :3, :3] = R_inv
                camera_w2cs_opencv[:, :3, 3:] = -torch.bmm(R_inv, T)

        outputs = {}
        if PTS3D in cameras.metadata:
            pts3d = cameras.metadata[PTS3D]
            camera_depth = cameras.metadata[DEPTH]
            neighbor_depth = cameras.metadata[NEIGHBOR_DEPTH]
            camera_fg_masks = cameras.metadata.get(FG_MASK, None)
            neighbor_fg_masks = cameras.metadata.get(NEIGHBOR_FG_MASK, None)
            depth_dim = cameras.width[0, 0].item()

            if self.training:
                cond_rgbs = cameras.metadata[NEIGHBOR_IMAGES]
                clip_rgbs = cond_rgbs[:, noisy_cond_views:]
            else:
                cond_rgbs = cameras.metadata["image"].unsqueeze(1)
                clip_rgbs = cond_rgbs

            cond_rgbs = cond_rgbs.permute(0, 1, 4, 2, 3)
            clip_rgbs = self.image_encoder_resize(clip_rgbs.flatten(0, 1))
            clip_rgbs = clip_rgbs.view(cond_rgbs.shape[0], -1, *clip_rgbs.shape[1:])

            if self.training and noisy_cond_views > 0:
                c2ws_for_noisy = torch.eye(4, device=cameras.device).view(1, 1, 4, 4).repeat(
                    neighbor_cameras.shape[0], 1 + noisy_cond_views, 1, 1)
                c2ws_for_noisy[:, :1, :3, :3] = R
                c2ws_for_noisy[:, :1, :3, 3:] = T
                c2ws_for_noisy[:, 1:, :3] = neighbor_cameras.camera_to_worlds[:, :noisy_cond_views]
                c2ws_for_noisy[:, 1:, :, 1:3] *= -1  # opengl to opencv
        else:
            depth_dim = 224
            rgbs = torch.cat([cameras.metadata["image"].unsqueeze(1), cameras.metadata[NEIGHBOR_IMAGES]], 1) \
                .permute(0, 1, 4, 2, 3)
            with torch.inference_mode():
                rgbs = self.image_encoder_resize(rgbs.flatten(0, 1))
                rgbs = rgbs.view(cameras.shape[0], -1, *rgbs.shape[1:])

            if FG_MASK in cameras.metadata:
                fg_masks = torch.cat([cameras.metadata[FG_MASK].unsqueeze(1), cameras.metadata[NEIGHBOR_FG_MASK]],
                                     1)
                with torch.inference_mode():
                    fg_masks = F.interpolate(fg_masks.unsqueeze(1), depth_dim).squeeze(1).bool()

                camera_fg_masks = fg_masks[:, 0]
                neighbor_fg_masks = fg_masks[:, 1:]
            else:
                fg_masks = None
                camera_fg_masks = None
                neighbor_fg_masks = None

            c2ws_opencv = torch.eye(4, device=cameras.device).view(1, 1, 4, 4).repeat(rgbs.shape[0], rgbs.shape[1],
                                                                                      1, 1)
            c2ws_opencv[:, 0, :3] = cameras.camera_to_worlds
            c2ws_opencv[:, 1:, :3] = neighbor_cameras.camera_to_worlds
            c2ws_opencv[:, :, :, 1:3] *= -1  # opengl to opencv

            camera_width = cameras.width[0].item()  # Assume camera dimensions are identical
            dust3r_scale_factor = depth_dim / camera_width
            fx = torch.cat([cameras.fx.unsqueeze(1), neighbor_cameras.fx], 1) * dust3r_scale_factor
            fy = torch.cat([cameras.fy.unsqueeze(1), neighbor_cameras.fy], 1) * dust3r_scale_factor
            cx = torch.cat([cameras.cx.unsqueeze(1), neighbor_cameras.cx], 1) * dust3r_scale_factor
            cy = torch.cat([cameras.cy.unsqueeze(1), neighbor_cameras.cy], 1) * dust3r_scale_factor

            all_pts3d, all_depth, alignment_loss, valid_alignment = self.dust3r_field.get_pts3d_and_depth(
                rgbs, fg_masks, c2ws_opencv, fx, fy, cx, cy)

            camera_depth = all_depth.view(rgbs.shape[0], -1, *all_depth.shape[1:])[:, 0].view(
                -1, depth_dim, depth_dim).clamp_min(0)
            neighbor_depth = all_depth.view(rgbs.shape[0], -1, *all_depth.shape[1:])[:, 1:].clamp_min(0)
            neighbor_depth = neighbor_depth.view(*neighbor_depth.shape[:2], depth_dim, depth_dim)

            if self.training:
                max_depth = camera_depth.flatten(1, 2).max(dim=-1)[0]
                valid_alignment = torch.logical_and(valid_alignment, max_depth > 0)
                pts3d = all_pts3d.view(rgbs.shape[0], -1, *all_pts3d.shape[1:])[:, 1:]
                cond_rgbs = rgbs[:, 1:]
                clip_rgbs = cond_rgbs[:, noisy_cond_views:]

                if noisy_cond_views > 0:
                    c2ws_for_noisy = c2ws_opencv[:, :noisy_cond_views + 1]

                if valid_alignment.any():
                    cameras.metadata = {}  # Should do deepcopy instead if we still need any metadata here
                    cameras = cameras[valid_alignment]
                    neighbor_cameras = neighbor_cameras[valid_alignment]
                    image_gt = image_gt[valid_alignment]
                    camera_w2cs_opencv = camera_w2cs_opencv[valid_alignment]
                    pts3d = pts3d[valid_alignment]
                    camera_depth = camera_depth[valid_alignment]
                    neighbor_depth = neighbor_depth[valid_alignment]
                    cond_rgbs = cond_rgbs[valid_alignment]
                    clip_rgbs = clip_rgbs[valid_alignment]

                    if neighbor_fg_masks is not None:
                        neighbor_fg_masks = neighbor_fg_masks[valid_alignment]
                        bg_colors = bg_colors[valid_alignment]

                    if noisy_cond_views > 0:
                        c2ws_for_noisy = c2ws_for_noisy[valid_alignment]
                        neighbor_images = neighbor_images[valid_alignment]
            else:
                pts3d = all_pts3d.view(rgbs.shape[0], -1, *all_pts3d.shape[1:])[:, :1]
                cond_rgbs = rgbs[:, :1]
                clip_rgbs = cond_rgbs

            outputs[ALIGNMENT_LOSS] = alignment_loss
            outputs[VALID_ALIGNMENT] = valid_alignment.to(rgbs.dtype).mean()

        # Get CLIP embeddings for cross attention
        if self.config.image_crossattn != "unet-replace":
            c_crossattn = self.image_encoder(self.image_encoder_normalize(clip_rgbs.flatten(0, 1))).image_embeds
            c_crossattn = c_crossattn.view(-1, clip_rgbs.shape[1], c_crossattn.shape[-1])
        else:
            with torch.inference_mode():
                c_crossattn = self.image_encoder(self.image_encoder_normalize(clip_rgbs.flatten(0, 1))).image_embeds
                c_crossattn = c_crossattn.view(-1, clip_rgbs.shape[1], c_crossattn.shape[-1])

        depth_scale_factor = torch.cat([camera_depth.unsqueeze(1), neighbor_depth], 1) \
            .view(cameras.shape[0], -1).mean(dim=-1).view(-1, 1, 1, 1)

        if self.training:
            timesteps = torch.randint(0, self.ddpm_scheduler.config.num_train_timesteps, (cameras.shape[0],),
                                      device=self.device, dtype=torch.long)
            outputs["timesteps"] = timesteps

            # Hack - we include the gt image in the forward loop, so that all model calculation is done in the
            # forward call as expected in DDP
            image_gt = image_gt.permute(0, 3, 1, 2)
            if image_gt.shape[-1] != self.config.image_dim:
                with torch.inference_mode():
                    image_gt = F.interpolate(image_gt, self.config.image_dim, mode="bicubic", antialias=True)

            if self.config.rgb_only:
                input_gt = image_gt * 2 - 1
            else:
                depth_gt = camera_depth.unsqueeze(1)
                if depth_gt.shape[-1] != self.config.image_dim:
                    with torch.inference_mode():
                        depth_gt = F.interpolate(depth_gt, self.config.image_dim, mode="bicubic", antialias=True)

                input_gt = torch.cat([image_gt * 2 - 1, self.encode_depth(depth_gt / depth_scale_factor)], 1)

            noise = torch.randn_like(input_gt)
            noisy_input_gt = self.ddpm_scheduler.add_noise(input_gt, noise, timesteps)
            # if self.config.project_overlay:
            #     noise_orig = noise
            #     noisy_input_gt_old = noisy_input_gt
            #     alpha_prod_t = self.ddpm_scheduler.alphas_cumprod.to(self.device)[timesteps]
            #     beta_prod_t = 1 - alpha_prod_t
            #     noise_old = (noisy_input_gt - input_gt * alpha_prod_t.view(-1, 1, 1, 1) ** 0.5) / (
            #             beta_prod_t.view(-1, 1, 1, 1) ** 0.5)
            #     input_gt_old = input_gt

            if noisy_cond_views > 0:
                noisy_scale_factor = self.config.image_dim / cameras.width[0, 0].item()

                depth_to_noise = neighbor_depth[:, :noisy_cond_views].flatten(0, 1).unsqueeze(1)
                if depth_to_noise.shape[-1] != self.config.image_dim:
                    with torch.inference_mode():
                        depth_to_noise = F.interpolate(depth_to_noise, self.config.image_dim, mode="bicubic",
                                                       antialias=True)

                rgb_to_noise = neighbor_images[:, :noisy_cond_views].permute(0, 1, 4, 2, 3).flatten(0, 1)
                if rgb_to_noise.shape[-1] != self.config.image_dim:
                    with torch.inference_mode():
                        rgb_to_noise = F.interpolate(rgb_to_noise, self.config.image_dim, mode="bicubic",
                                                     antialias=True)

                rgbd_to_noise = torch.cat([rgb_to_noise * 2 - 1,
                                           self.encode_depth(depth_to_noise / depth_scale_factor)], 1)

                if self.config.independent_noise:
                    noise_to_add = torch.randn_like(rgbd_to_noise)
                else:
                    noise_to_add = noise.repeat_interleave(noisy_cond_views, dim=0)

                noisy_rgbd = self.ddpm_scheduler.add_noise(rgbd_to_noise, noise_to_add,
                                                           timesteps.repeat_interleave(noisy_cond_views))

                if self.config.project_overlay:
                    depth_fx = neighbor_cameras.fx[:, :noisy_cond_views].view(-1, 1) * noisy_scale_factor
                    depth_fy = neighbor_cameras.fy[:, :noisy_cond_views].view(-1, 1) * noisy_scale_factor
                    depth_cx = neighbor_cameras.cx[:, :noisy_cond_views].view(-1, 1) * noisy_scale_factor
                    depth_cy = neighbor_cameras.cy[:, :noisy_cond_views].view(-1, 1) * noisy_scale_factor
                    input_gt = self.overlay_encoded_rgbd(
                        torch.cat([input_gt.unsqueeze(1),
                                   rgbd_to_noise.view(cameras.shape[0], noisy_cond_views, *rgbd_to_noise.shape[1:])],
                                  1), cameras, c2ws_for_noisy[:, 1:], camera_w2cs_opencv, depth_fx, depth_fy, depth_cx,
                        depth_cy, neighbor_fg_masks, depth_scale_factor)

                    noisy_input_gt = self.overlay_encoded_rgbd(
                        torch.cat([noisy_input_gt.unsqueeze(1),
                                   noisy_rgbd.view(cameras.shape[0], noisy_cond_views, *noisy_rgbd.shape[1:])], 1),
                        cameras, c2ws_for_noisy[:, 1:], camera_w2cs_opencv, depth_fx, depth_fy, depth_cx, depth_cy,
                        neighbor_fg_masks, depth_scale_factor)
                else:
                    depth_fx = torch.cat([cameras.fx.unsqueeze(1), neighbor_cameras.fx[:, :noisy_cond_views]],
                                         1) * noisy_scale_factor
                    depth_fy = torch.cat([cameras.fy.unsqueeze(1), neighbor_cameras.fy[:, :noisy_cond_views]],
                                         1) * noisy_scale_factor
                    depth_cx = torch.cat([cameras.cx.unsqueeze(1), neighbor_cameras.cx[:, :noisy_cond_views]],
                                         1) * noisy_scale_factor
                    depth_cy = torch.cat([cameras.cy.unsqueeze(1), neighbor_cameras.cy[:, :noisy_cond_views]],
                                         1) * noisy_scale_factor

                    if self.config.scale_with_pixel_area:
                        camera_scale_mult = self.get_pixel_area_scale_mult(cameras, self.config.image_dim).flatten(
                            -2, -1).unsqueeze(1)
                        noisy_scale_mult = self.get_pixel_area_scale_mult(
                            neighbor_cameras[:, :noisy_cond_views].flatten(),
                            self.config.image_dim).flatten(-2, -1).unsqueeze(
                            1)
                        noise_scale_mult = torch.cat([camera_scale_mult, noisy_scale_mult], 1)
                    else:
                        noise_scale_mult = 1 / depth_fx

                    noise_masks = torch.cat([fg_masks.unsqueeze(1), neighbor_fg_masks],
                                            1) if fg_masks is not None else None

                    input_gt, _ = self.project_encoded_rgbd(
                        torch.cat([input_gt.unsqueeze(1),
                                   rgbd_to_noise.view(cameras.shape[0], noisy_cond_views, *rgbd_to_noise.shape[1:])],
                                  1),
                        cameras, c2ws_for_noisy, camera_w2cs_opencv, depth_fx.view(-1, 1), depth_fy.view(-1, 1),
                        depth_cx.view(-1, 1), depth_cy.view(-1, 1), bg_colors, noise_masks, noise_scale_mult,
                        depth_scale_factor)
                    input_gt = input_gt[:, :-1]

                    if self.config.noise_after_reproj:
                        noisy_input_gt = self.ddpm_scheduler.add_noise(input_gt, noise, timesteps)

                    noisy_input_gt, _ = self.project_encoded_rgbd(
                        torch.cat([noisy_input_gt.unsqueeze(1),
                                   noisy_rgbd.view(cameras.shape[0], noisy_cond_views, *noisy_rgbd.shape[1:])], 1),
                        cameras,
                        c2ws_for_noisy, camera_w2cs_opencv, depth_fx.view(-1, 1), depth_fy.view(-1, 1),
                        depth_cx.view(-1, 1), depth_cy.view(-1, 1), bg_colors, noise_masks, noise_scale_mult,
                        depth_scale_factor)
                    noisy_input_gt = noisy_input_gt[:, :-1]

                alpha_prod_t = self.ddpm_scheduler.alphas_cumprod.to(self.device)[timesteps]
                beta_prod_t = 1 - alpha_prod_t
                noise = (noisy_input_gt - input_gt * alpha_prod_t.view(-1, 1, 1, 1) ** 0.5) / (
                        beta_prod_t.view(-1, 1, 1, 1) ** 0.5)

                # if self.config.project_overlay:
                #     lol = (torch.cat([input_gt_old, input_gt, noisy_input_gt_old, noisy_input_gt], -1).permute(0, 2, 3,  1).flatten( 0, 1)[..., :3] + 1) / 2
                #     from PIL import Image
                #     Image.fromarray((lol * 255).byte().cpu().numpy()).save('/scratch/hturki/nt.png')
                #     import pdb;
                #     pdb.set_trace()
                #     lol2 = (noisy_rgbd.permute(0, 2, 3, 1).flatten(0, 1)[..., :3] + 1) / 2
                #     Image.fromarray((lol2 * 255).byte().cpu().numpy()).save('/scratch/hturki/nt2.png')

                pts3d = pts3d[:, noisy_cond_views:]
                neighbor_depth = neighbor_depth[:, noisy_cond_views:]
                cond_rgbs = cond_rgbs[:, noisy_cond_views:]
                if neighbor_fg_masks is not None:
                    neighbor_fg_masks = neighbor_fg_masks[:, noisy_cond_views:]

            opacity = neighbor_fg_masks.view(pts3d[..., :1].shape) if neighbor_fg_masks is not None \
                else torch.ones_like(pts3d[..., :1])

            if self.config.scale_with_pixel_area:
                neighbor_scale_mult = self.get_pixel_area_scale_mult(neighbor_cameras[:, noisy_cond_views:].flatten(),
                                                                     depth_dim).view(*neighbor_depth.shape[:2], -1)
            else:
                neighbor_scale_mult = 1 / (
                        neighbor_cameras[:, noisy_cond_views:].fx * depth_dim / neighbor_cameras.width[0, 0].item())

            c_concat, _ = self.get_concat(
                camera_w2cs_opencv,
                cameras,
                pts3d,
                neighbor_depth.flatten(-2, -1) * neighbor_scale_mult,
                opacity,
                cond_rgbs,
                1,
                bg_colors,
                depth_scale_factor
            )

            random_mask = torch.rand((c_concat.shape[0], 1, 1), device=c_concat.device)
            prompt_mask = random_mask < 2 * self.config.uncond
            input_mask = torch.logical_not(
                torch.logical_and(random_mask >= self.config.uncond,
                                  random_mask < 3 * self.config.uncond)).unsqueeze(-1)

            c_crossattn = torch.where(prompt_mask, 0, c_crossattn)
            c_concat = input_mask * c_concat

            if self.ddpm_scheduler.config.prediction_type == "epsilon":
                outputs["target"] = noise
            elif self.ddpm_scheduler.config.prediction_type == "sample":
                outputs["target"] = input_gt
            elif self.ddpm_scheduler.config.prediction_type == "v_prediction":
                outputs["target"] = self.ddpm_scheduler.get_velocity(input_gt, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {self.ddpm_scheduler.config.prediction_type}")

            outputs["model_output"] = self.unet(torch.cat([noisy_input_gt, c_concat], 1), timesteps,
                                                c_crossattn).sample
        else:
            with torch.cuda.amp.autocast(enabled=False):
                neighbor_w2cs = torch.eye(4, device=cameras.device).view(1, 1, 4, 4).repeat(
                    neighbor_cameras.shape[0], neighbor_cameras.shape[1], 1, 1)
                neighbor_R = neighbor_cameras.camera_to_worlds[:, :, :3, :3].clone()
                neighbor_R[:, :, :, 1:3] *= -1  # opengl to opencv
                neighbor_R_inv = neighbor_R.transpose(2, 3)
                neighbor_w2cs[:, :, :3, :3] = neighbor_R_inv
                neighbor_T = neighbor_cameras.camera_to_worlds[:, :, :3, 3:]
                neighbor_inv_T = -torch.bmm(neighbor_R_inv.view(-1, 3, 3), neighbor_T.view(-1, 3, 1))
                neighbor_w2cs[:, :, :3, 3:] = neighbor_inv_T.view(neighbor_w2cs[:, :, :3, 3:].shape)

            assert neighbor_cameras.shape[0] == 1, neighbor_cameras.shape
            neighbor_w2cs = neighbor_w2cs.squeeze(0)

            if self.config.scale_with_pixel_area:
                cameras.metadata = {}
                camera_scale_mult = self.get_pixel_area_scale_mult(cameras, depth_dim).view(cameras.shape[0], -1)
            else:
                camera_scale_mult = 1 / (cameras.fx * depth_dim / cameras.width[0, 0].item())

            opacity = camera_fg_masks.view(pts3d[..., :1].shape) if camera_fg_masks is not None \
                else torch.ones_like(pts3d[..., :1])

            c_concat, rendering = self.get_concat(
                neighbor_w2cs,
                neighbor_cameras,
                pts3d,
                camera_depth.flatten(1, 2) * camera_scale_mult,
                opacity,
                cond_rgbs,
                neighbor_cameras.shape[1],
                bg_colors,
                depth_scale_factor,
                normalize_rgb=False,
            )

            concat_rgb = [torch.cat([*rendering[RGB]], 1)]
            concat_depth = [torch.cat([*rendering[DEPTH]], 1)]
            concat_acc = [torch.cat([*rendering[ACCUMULATION]], 1)]

            self.ddim_scheduler.set_timesteps(self.config.ddim_steps, device=self.device)

            inference_rgbd = randn_tensor(
                (neighbor_cameras.shape[1], 4, self.config.image_dim, self.config.image_dim),
                device=c_concat.device,
                dtype=c_concat.dtype)

            if noisy_cond_views > 0:
                neighbor_c2ws_opencv = torch.eye(4, device=cameras.device).view(1, 4, 4).repeat(
                    neighbor_cameras.shape[1], 1, 1)
                neighbor_c2ws_opencv[:, :3] = neighbor_cameras[0].camera_to_worlds
                neighbor_c2ws_opencv[:, :, 1:3] *= -1  # opengl to opencv

                render_scale_factor = self.config.image_dim / cameras.width[0, 0].item()
                focals_for_decode = torch.cat([neighbor_cameras.fx.view(-1, 1), neighbor_cameras.fy.view(-1, 1)],
                                              -1) * render_scale_factor
                pp_for_decode = torch.cat([neighbor_cameras.cx.view(-1, 1), neighbor_cameras.cy.view(-1, 1)],
                                          -1) * render_scale_factor

                if self.config.scale_with_pixel_area:
                    render_scale_mult = self.get_pixel_area_scale_mult(neighbor_cameras[0], self.config.image_dim).view(
                        neighbor_cameras.shape[1], -1)
                else:
                    render_scale_mult = 1 / focals_for_decode[:, :1]

            c_crossattn_inference = c_crossattn.expand(neighbor_cameras.shape[1], -1, -1)

            if self.config.guidance_scale > 1:
                c_concat = torch.cat([torch.zeros_like(c_concat), c_concat])
                c_crossattn_inference = torch.cat([torch.zeros_like(c_crossattn_inference), c_crossattn_inference])

            samples_rgb = []
            samples_depth = []

            if self.config.use_ema:
                self.ema_unet.to(self.device)
                self.ema_unet.store(self.unet.parameters())
                self.ema_unet.copy_to(self.unet.parameters())
            try:
                decoded = inference_rgbd.permute(0, 2, 3, 1)
                samples_rgb.append(((torch.cat([*decoded[..., :3]], 1) + 1) / 2).clamp(0, 1))
                decoded_depth = self.decode_depth(decoded[..., 3:]) * depth_scale_factor
                samples_depth.append(torch.cat([*decoded_depth], 1))
                # for t_index, t in enumerate(self.ddpm_scheduler.timesteps):
                for t_index, t in enumerate(self.ddim_scheduler.timesteps):
                    if noisy_cond_views > 0:
                        decoded_depth = self.decode_depth(inference_rgbd[:, 3:]) * depth_scale_factor

                        decoded_pts3d_cam = fast_depthmap_to_pts3d(
                            decoded_depth.view(decoded_depth.shape[0], -1),
                            self.pixels,
                            focals_for_decode,
                            pp_for_decode)

                        with torch.cuda.amp.autocast(enabled=False):
                            # neighbors, 4, num_points
                            decoded_pts3d = neighbor_c2ws_opencv @ torch.cat(
                                [decoded_pts3d_cam, torch.ones_like(decoded_pts3d_cam[..., :1])], -1).transpose(1, 2)

                        if self.config.project_overlay:
                            decoded_rgb = ((inference_rgbd[:, :3] + 1) / 2).permute(0, 2, 3, 1)

                            pts3d_to_render = []
                            rgb_to_overlay = []

                            for neighbor_index in range(neighbor_cameras.shape[1]):
                                cur_pts3d_to_render = []
                                cur_rgb_to_overlay = []
                                for other_neighbor_index in range(neighbor_cameras.shape[1]):
                                    if neighbor_index == other_neighbor_index:
                                        continue
                                    cur_pts3d_to_render.append(decoded_pts3d[other_neighbor_index])
                                    cur_rgb_to_overlay.append(decoded_rgb[other_neighbor_index].view(-1, 3))

                                pts3d_to_render.append(torch.cat(cur_pts3d_to_render, -1))
                                rgb_to_overlay.append(torch.cat(cur_rgb_to_overlay, -1))

                            pts3d_to_render = torch.stack(pts3d_to_render)
                            rgb_to_overlay = torch.stack(rgb_to_overlay)

                            image_dim = self.config.image_dim
                            with torch.cuda.amp.autocast(enabled=False):
                                pts3d_cam_render = neighbor_w2cs @ pts3d_to_render
                                pts3d_cam_render = pts3d_cam_render.transpose(1, 2)[..., :3]
                                z_pos = pts3d_cam_render[..., 2]
                                uv = pts3d_cam_render[:, :, :2] / pts3d_cam_render[:, :, 2:]
                                uv *= focals_for_decode.unsqueeze(1)
                                uv += pp_for_decode.unsqueeze(1)

                                is_valid_u = torch.logical_and(0 <= uv[..., 0], uv[..., 0] < image_dim - 1)
                                is_valid_v = torch.logical_and(0 <= uv[..., 1], uv[..., 1] < image_dim - 1)
                                is_valid_z = z_pos > 0
                                is_valid_point = torch.logical_and(torch.logical_and(is_valid_u, is_valid_v),
                                                                   is_valid_z)

                                max_val = torch.finfo(torch.float32).max
                                u = torch.round(uv[..., 0]).long().clamp(0, image_dim - 1)
                                v = torch.round(uv[..., 1]).long().clamp(0, image_dim - 1)
                                z = torch.where(is_valid_point, z_pos, max_val)
                                camera_index = torch.arange(neighbor_cameras.shape[1], device=z.device).view(-1,
                                                                                                             1).expand(
                                    -1, z.shape[-1])
                                depth_map = decoded_depth.permute(0, 2, 3, 1).clone()
                                indices = (camera_index * image_dim * image_dim + v * image_dim + u).view(-1)
                                _, min_indices = scatter_min(z.view(-1), indices, out=depth_map.view(-1))

                                rgb_to_fill = min_indices[min_indices <= indices.shape[0] - 1]
                                rgb_map = decoded_rgb.reshape(-1, 3).clone()
                                rgb_map[indices[rgb_to_fill]] = rgb_to_overlay.view(-1, 3)[rgb_to_fill]
                                rgb_map = rgb_map.view(*depth_map.shape[:-1], 3)

                                encoded_depth = (self.encode_depth(depth_map / depth_scale_factor)).permute(0, 3, 1, 2)
                                inference_rgbd = torch.cat([rgb_map.permute(0, 3, 1, 2) * 2 - 1, encoded_depth], 1)

                                # if t_index % 20 == 0:
                                if t_index == 0 or (t_index > self.config.ddim_steps - 10 and t_index % 2 == 0):
                                    concat_rgb.append(torch.cat([*rgb_map.clamp(0, 1)], 1))
                                    concat_depth.append(torch.cat([*depth_map], 1))
                                    concat_acc.append(torch.ones_like(concat_depth[-1]))
                        else:
                            decoded_pts3d = decoded_pts3d.transpose(1, 2)[..., :3]
                            decoded_pts3d = decoded_pts3d.reshape(1, -1, 3)
                            inference_rgbd, rendering = self.get_concat(
                                neighbor_w2cs,
                                neighbor_cameras,
                                decoded_pts3d,
                                decoded_depth.view(neighbor_cameras.shape[1], -1) * render_scale_mult,
                                torch.ones_like(decoded_pts3d[..., 0]),
                                (inference_rgbd[:, :3].unsqueeze(0) + 1) / 2,
                                neighbor_cameras.shape[1],
                                bg_colors,
                                depth_scale_factor,
                                calc_compensations=False,
                                normalize_rgb=True,
                                opacity_multiplier=self.config.opacity_multiplier,
                            )
                            inference_rgbd = inference_rgbd[:, :-1]

                            # if t_index % 20 == 0:
                            if t_index == 0 or (t_index > self.config.ddim_steps - 10 and t_index % 2 == 0):
                                concat_rgb.append(torch.cat([*rendering[RGB].clamp(0, 1)], 1))
                                concat_depth.append(torch.cat([*rendering[DEPTH]], 1))
                                concat_acc.append(torch.cat([*rendering[ACCUMULATION]], 1))

                    rgbd_input = torch.cat([inference_rgbd] * 2) \
                        if self.config.guidance_scale > 1 else inference_rgbd

                    # rgbd_input = self.ddpm_scheduler.scale_model_input(rgbd_input, t)
                    rgbd_input = self.ddim_scheduler.scale_model_input(rgbd_input, t)
                    rgbd_input = torch.cat([rgbd_input, c_concat], dim=1)
                    noise_pred = self.unet(rgbd_input, t, c_crossattn_inference, return_dict=False)[0]

                    if self.config.guidance_scale > 1:
                        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + self.config.guidance_scale * \
                                     (noise_pred_cond - noise_pred_uncond)

                    # inference_rgbd = self.ddpm_scheduler.step(noise_pred, t, inference_rgbd, return_dict=False)[0]
                    inference_rgbd = self.ddim_scheduler.step(noise_pred, t, inference_rgbd, return_dict=False)[0]
                    # if t_index % 20 == 0:
                    if t_index == 0 or (t_index > self.config.ddim_steps - 10 and t_index % 2 == 0):
                        decoded = inference_rgbd.permute(0, 2, 3, 1)
                        samples_rgb.append(((torch.cat([*decoded[..., :3]], 1) + 1) / 2).clamp(0, 1))
                        decoded_depth = self.decode_depth(decoded[..., 3:]) * depth_scale_factor
                        samples_depth.append(torch.cat([*decoded_depth], 1))
            finally:
                if self.config.use_ema:
                    self.ema_unet.restore(self.unet.parameters())

            decoded = inference_rgbd.permute(0, 2, 3, 1)
            decoded_rgb = ((decoded[..., :3] + 1) / 2).clamp(0, 1)
            samples_rgb.append(torch.cat([*decoded_rgb], 1))
            outputs["samples_rgb"] = torch.cat(samples_rgb)
            outputs[DEPTH_GT] = camera_depth.squeeze(0).unsqueeze(-1)

            if not self.config.rgb_only:
                decoded_depth = self.decode_depth(decoded[..., 3:]) * depth_scale_factor
                samples_depth.append(torch.cat([*decoded_depth], 1))
                outputs["samples_depth"] = torch.cat(samples_depth)
                outputs[NEIGHBOR_DEPTH] = neighbor_depth.view(-1, *outputs[DEPTH_GT].shape)
            outputs[RGB] = torch.cat(concat_rgb)
            outputs[DEPTH] = torch.cat(concat_depth)
            outputs[ACCUMULATION] = torch.cat(concat_acc)

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

        if VALID_ALIGNMENT in outputs and outputs[VALID_ALIGNMENT] == 0:
            CONSOLE.log("No valid outputs, ignoring training batch")
            loss = loss * 0  # Do this so that backprop doesn't complain

        return {"loss": loss}

    @profiler.time_function
    def get_image_metrics_and_images(
            self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        cond_rgb_gt = batch["image"].to(outputs[RGB].device)
        cond_depth_gt = outputs[DEPTH_GT]
        if cond_depth_gt.shape[0] != cond_rgb_gt.shape[1]:
            cond_depth_gt = F.interpolate(
                cond_depth_gt.squeeze(-1).unsqueeze(0).unsqueeze(0),
                cond_rgb_gt.shape[:2], mode="bicubic", antialias=True).view(*cond_rgb_gt.shape[:2], 1)
        cond_depth_gt = colormaps.apply_depth_colormap(cond_depth_gt)
        cond_gt = torch.cat([cond_rgb_gt, cond_depth_gt], 1)

        neighbor_images = batch[NEIGHBOR_IMAGES].squeeze(0)
        if neighbor_images.shape[-2] != self.config.image_dim:
            neighbor_images = F.interpolate(neighbor_images.permute(0, 3, 1, 2), self.config.image_dim,
                                            mode="bicubic", antialias=True).permute(0, 2, 3, 1)
        image_gt = torch.cat([*neighbor_images.to(outputs[RGB].device)], 1)
        cond_rgb = torch.cat([outputs[RGB], image_gt])
        samples_rgb = torch.cat([outputs["samples_rgb"], image_gt])

        images_dict = {
            "cond_gt": cond_gt,
            "concat_rgb": cond_rgb,
            "samples_rgb": samples_rgb
        }

        if not self.config.rgb_only:
            depth_gt = outputs[NEIGHBOR_DEPTH]
            if depth_gt.shape[1] != self.config.image_dim:
                depth_gt = F.interpolate(depth_gt.squeeze(-1).unsqueeze(0),
                                         self.config.image_dim, mode="bicubic", antialias=True).squeeze(0).unsqueeze(-1)
            depth_gt = torch.cat([*depth_gt], 1)
            near_plane = depth_gt.min()
            far_plane = depth_gt.max()
            images_dict["concat_depth"] = torch.cat(
                [colormaps.apply_depth_colormap(outputs[DEPTH], outputs[ACCUMULATION], near_plane=near_plane,
                                                far_plane=far_plane),
                 colormaps.apply_depth_colormap(depth_gt)])
            images_dict["samples_depth"] = colormaps.apply_depth_colormap(
                torch.cat([outputs["samples_depth"], depth_gt]), near_plane=near_plane, far_plane=far_plane)

        return {}, images_dict

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)

        self.ema_unet.to(self.device)
        for key, val in self.ema_unet.state_dict().items():
            state_dict[f"ema_unet.{key}"] = val

        return state_dict

    def load_state_dict(self, dict, **kwargs):  # type: ignore
        prefix = "ema_unet."
        prefix_len = len(prefix)
        ema_state_dict = {}
        other_dict = {}
        for key, val in dict.items():
            if key.startswith(prefix):
                ema_state_dict[key[prefix_len:]] = val
            else:
                other_dict[key] = val

        super().load_state_dict(other_dict, **kwargs)
        self.ema_unet.to(self.device)
        self.ema_unet.load_state_dict(ema_state_dict)
