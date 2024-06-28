from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union, Type, Optional, Literal, Any

import math
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, EMAModel, DDIMScheduler
from diffusers.utils.torch_utils import randn_tensor
from gsplat import fully_fused_projection, isect_tiles, isect_offset_encode, rasterize_to_pixels
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
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchvision.transforms import transforms
from transformers import CLIPVisionModelWithProjection

from generfstudio.fields.batched_pc_optimizer import fast_depthmap_to_pts3d
from generfstudio.fields.dino_v2_encoder import DinoV2Encoder
from generfstudio.fields.dust3r_field import Dust3rField
from generfstudio.fields.spatial_encoder import SpatialEncoder
from generfstudio.generfstudio_constants import NEIGHBOR_IMAGES, NEIGHBOR_CAMERAS, NEAR, FAR, \
    POSENC_SCALE, RGB, ACCUMULATION, DEPTH, ALIGNMENT_LOSS, PTS3D, VALID_ALIGNMENT, \
    NEIGHBOR_DEPTH, DEPTH_GT, NEIGHBOR_FG_MASK, FG_MASK, BG_COLOR


@dataclass
class RGBDDiffusionConfig(ModelConfig):
    _target: Type = field(default_factory=lambda: RGBDDiffusion)

    uncond: float = 0.05
    noisy_only: float = 0.1
    ddim_steps = 50

    cond_image_encoder_type: Literal["unet", "resnet", "dino"] = "resnet"
    cond_feature_out_dim: int = 0
    freeze_cond_image_encoder: bool = False

    unet_pretrained_path: str = "Intel/ldm3d-4c"
    vae_pretrained_path: str = "Intel/ldm3d-4c"
    image_encoder_pretrained_path: str = "lambdalabs/sd-image-variations-diffusers"
    scheduler_pretrained_path: str = "Intel/ldm3d-4c"

    beta_schedule: str = "scaled_linear"
    prediction_type: str = "epsilon"

    dust3r_model_name: str = "/data/hturki/dust3r/checkpoints/DUSt3R_ViTLarge_BaseDecoder_224_linear.pth"
    noisy_cond_views: int = 0
    depth_mapping: Literal["simple", "disparity", "sigmoid"] = "disparity"
    rgbd_concat_strategy: Literal["mode", "sample", "mean_std", "downsample"] = "sample"

    guidance_scale: float = 2.0

    use_ema: bool = True
    allow_tf32: bool = False
    rgb_only: bool = False


class RGBDDiffusion(Model):
    config: RGBDDiffusionConfig

    def __init__(self, config: RGBDDiffusionConfig, metadata: Dict[str, Any], **kwargs) -> None:
        if config.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

        self.depth_available = DEPTH in metadata
        CONSOLE.log(f"Depth available: {self.depth_available}")

        super().__init__(config=config, **kwargs)

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()
        self.collider = None
        assert self.config.cond_feature_out_dim == 0, self.config.cond_feature_out_dim

        self.vae = AutoencoderKL.from_pretrained(self.config.vae_pretrained_path, subfolder="vae")
        self.vae.requires_grad_(False)
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(self.config.image_encoder_pretrained_path,
                                                                           subfolder="image_encoder")
        self.image_encoder.requires_grad_(False)

        self.image_encoder_resize = transforms.Resize(
            (224, 224),
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=False,
        )
        self.image_encoder_normalize = transforms.Normalize(
            [0.48145466, 0.4578275, 0.40821073],
            [0.26862954, 0.26130258, 0.27577711])
        # 4 for latent + 4 for RGBD + 1 for accumulation + feature dim
        unet_base = UNet2DConditionModel.from_pretrained(self.config.unet_pretrained_path, subfolder="unet",
                                                         low_cpu_mem_usage=False,
                                                         ignore_mismatched_sizes=False)

        concat_channels = 5 + self.config.cond_feature_out_dim  # RGBD or VAE + acc mask + features = 4 + 4 + 1 = 9
        if self.config.rgbd_concat_strategy == "mean_std":
            concat_channels += 4
        if self.config.noisy_cond_views > 0:
            concat_channels *= 2
            concat_channels += 1

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
            ema_unet = UNet2DConditionModel.from_pretrained(self.config.unet_pretrained_path, subfolder="unet",
                                                            in_channels=4 + concat_channels,
                                                            low_cpu_mem_usage=False,
                                                            ignore_mismatched_sizes=True)
            if old_state[in_key].shape != new_state[in_key].shape:
                ema_unet.load_state_dict(new_state)

            self.ema_unet = EMAModel(ema_unet.parameters(), model_cls=UNet2DConditionModel,
                                     model_config=ema_unet.config)

        self.ddpm_scheduler = DDPMScheduler.from_pretrained(self.config.scheduler_pretrained_path,
                                                            subfolder="scheduler",
                                                            beta_schedule=self.config.beta_schedule,
                                                            prediction_type=self.config.prediction_type)
        self.ddim_scheduler = DDIMScheduler.from_pretrained(self.config.scheduler_pretrained_path,
                                                            subfolder="scheduler",
                                                            beta_schedule=self.config.beta_schedule,
                                                            prediction_type=self.config.prediction_type)

        if self.config.beta_schedule != self.ddpm_scheduler.config["beta_schedule"] \
                or self.config.prediction_type != self.ddpm_scheduler.config["prediction_type"]:
            self.ddpm_scheduler.config["beta_schedule"] = self.config.beta_schedule
            self.ddpm_scheduler.config["prediction_type"] = self.config.prediction_type
            self.ddpm_scheduler = DDPMScheduler.from_config(self.ddpm_scheduler.config)

            self.ddim_scheduler.config["beta_schedule"] = self.config.beta_schedule
            self.ddim_scheduler.config["prediction_type"] = self.config.prediction_type
            self.ddim_scheduler = DDIMScheduler.from_config(self.ddim_scheduler.config)

        assert self.config.noisy_cond_views >= 0, self.config.noisy_cond_views
        if self.config.noisy_cond_views > 0:
            assert not self.config.rgb_only
        if self.config.cond_feature_out_dim > 0:
            assert self.config.noisy_cond_views == 0, self.config.noisy_cond_views
            if self.config.cond_image_encoder_type == "resnet":
                self.cond_image_encoder = SpatialEncoder()
                encoder_dim = self.cond_image_encoder.latent_size
                # self.cond_image_encoder = torch.compile(self.cond_image_encoder)
            elif self.config.cond_image_encoder_type == "unet":
                assert not self.config.freeze_cond_image_encoder
                encoder_dim = 128
                # self.cond_image_encoder = torch.compile(UNet(in_channels=3, n_classes=encoder_dim))
            elif self.config.cond_image_encoder_type == "dino":
                assert not self.config.freeze_cond_image_encoder
                self.cond_image_encoder = DinoV2Encoder(out_feature_dim=self.config.cond_feature_out_dim)
                encoder_dim = self.config.cond_feature_out_dim
            else:
                raise Exception(self.config.cond_image_encoder_type)

        self.cond_feature_field = Dust3rField(model_name=self.config.dust3r_model_name,
                                              depth_precomputed=self.depth_available)

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        for p in self.lpips.parameters():
            p.requires_grad_(False)

        if get_world_size() > 1:
            CONSOLE.log("Using sync batchnorm for DDP")
            self.unet = nn.SyncBatchNorm.convert_sync_batchnorm(self.unet)
            self.cond_feature_field = nn.SyncBatchNorm.convert_sync_batchnorm(self.cond_feature_field)
            if self.config.cond_feature_out_dim > 0 and (not self.config.freeze_cond_image_encoder):
                self.cond_image_encoder = nn.SyncBatchNorm.convert_sync_batchnorm(self.cond_image_encoder)

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        param_groups["cond_encoder"] = list(self.cond_feature_field.parameters())

        if self.config.cond_feature_out_dim > 0 and (not self.config.freeze_cond_image_encoder):
            param_groups["cond_encoder"] += list(self.cond_image_encoder.parameters())

        param_groups["fields"] = list(self.unet.parameters())

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

    @profiler.time_function
    @torch.inference_mode()
    def encode_with_vae(self, to_encode: torch.Tensor, sample_override: bool = False,
                        autocast_enabled: bool = True) -> torch.Tensor:
        with torch.cuda.amp.autocast(enabled=autocast_enabled):
            latent_dist = self.vae.encode(to_encode * 2 - 1).latent_dist
            if sample_override or self.config.rgbd_concat_strategy == "sample":
                encoded = latent_dist.sample() * self.vae.config.scaling_factor
            elif self.config.rgbd_concat_strategy == "mode":
                encoded = latent_dist.mode() * self.vae.config.scaling_factor
            elif self.config.rgbd_concat_strategy == "mean_std":
                encoded = torch.cat([latent_dist.mean, latent_dist.std], 1) * self.vae.config.scaling_factor
            else:
                raise Exception(self.config.rgbd_concat_strategy)

        to_upcast = torch.logical_not(torch.isfinite(encoded.view(encoded.shape[0], -1)).all(dim=-1))
        if to_upcast.any():
            encoded[to_upcast] = self.encode_with_vae(to_encode[to_upcast], sample_override, False)

        return encoded

    @profiler.time_function
    def decode_with_vae(self, to_decode: torch.Tensor, autocast_enabled: bool = True) -> torch.Tensor:
        with torch.cuda.amp.autocast(enabled=autocast_enabled):
            decoded = (self.vae.decode(to_decode / self.vae.config.scaling_factor, return_dict=False)[0] + 1) / 2

        to_upcast = torch.logical_not(torch.isfinite(decoded.view(decoded.shape[0], -1)).all(dim=-1))
        if to_upcast.any():
            decoded[to_upcast] = self.decode_with_vae(decoded[to_upcast], False)

        return decoded

    def encode_depth(self, to_encode: torch.Tensor) -> torch.Tensor:
        scaled = to_encode.float()
        if self.config.depth_mapping == "simple":
            return scaled
        elif self.config.depth_mapping == "disparity":
            return 1 - 1 / (1 + scaled)
        elif self.config.depth_mapping == "sigmoid":
            return torch.sigmoid(scaled)
        else:
            raise Exception(self.config.depth_mapping)

    def decode_depth(self, to_decode: torch.Tensor) -> torch.Tensor:
        if self.config.depth_mapping == "simple":
            scaled = to_decode.float().clamp_min(0)
        elif self.config.depth_mapping == "disparity":
            scaled = 1 / (1 - to_decode.float().clamp(0, 0.999)) - 1
        elif self.config.depth_mapping == "sigmoid":
            scaled = torch.logit(to_decode.float().clamp(0, 1 - 1e-5))
        else:
            raise Exception(self.config.depth_mapping)

        return scaled

    @staticmethod
    @profiler.time_function
    @torch.cuda.amp.autocast(enabled=False)
    def splat_gaussians(w2cs: torch.Tensor, fx: torch.Tensor, fy: torch.Tensor, cx: torch.Tensor,
                        cy: torch.Tensor, xyz: torch.Tensor, scales: torch.Tensor, opacity: torch.Tensor,
                        rgbs: Optional[torch.Tensor], camera_dim: int, return_depth: bool, cameras_per_scene: int,
                        bg_colors: Optional[torch.Tensor], calc_compensations: bool = False,
                        normalize_rgb: bool = False):
        quats = torch.cat([torch.ones_like(xyz[..., :1]), torch.zeros_like(xyz)], -1)

        zeros = torch.zeros_like(fx)
        ones = torch.ones_like(fx)

        K = torch.cat([
            torch.cat([fx, zeros, cx], -1).unsqueeze(1),
            torch.cat([zeros, fy, cy], -1).unsqueeze(1),
            torch.cat([zeros, zeros, ones], -1).unsqueeze(1)
        ], 1)

        radii, means2d, depths, conics, compensations = fully_fused_projection(
            xyz, None, quats, scales, w2cs, K, camera_dim, camera_dim, cameras_per_scene=cameras_per_scene,
            calc_compensations=calc_compensations)

        tile_size = 16
        tile_width = math.ceil(camera_dim / tile_size)
        tile_height = math.ceil(camera_dim / tile_size)
        tiles_per_gauss, isect_ids, gauss_ids = isect_tiles(
            means2d, radii, depths, tile_size, tile_width, tile_height
        )

        # assert (tiles_per_gauss > 0).any()

        isect_offsets = isect_offset_encode(isect_ids, w2cs.shape[0], tile_width, tile_height)

        inputs = []
        if rgbs is not None:
            inputs.append(rgbs.expand(cameras_per_scene, -1, -1) if cameras_per_scene > 1 else rgbs)
        if return_depth:
            inputs.append(depths.unsqueeze(-1))

        inputs = torch.cat(inputs, -1)

        if bg_colors is not None:
            bg_inputs = []
            if rgbs is not None:
                bg_inputs.append(bg_colors)
            if return_depth:
                bg_inputs.append(torch.zeros_like(bg_colors[..., :1]))

            bg_colors = torch.cat(bg_inputs, -1)
            if cameras_per_scene > 1:
                bg_colors = bg_colors.expand(cameras_per_scene, -1)

        if cameras_per_scene > 1:
            opacity = opacity.expand(cameras_per_scene, -1)

        if compensations is not None:
            opacity = opacity * compensations

        splat_results, alpha = rasterize_to_pixels(
            means2d,
            conics,
            inputs,
            opacity,
            camera_dim,
            camera_dim,
            tile_size,
            isect_offsets,
            gauss_ids,
            backgrounds=bg_colors,
        )

        outputs = {ACCUMULATION: alpha}

        if rgbs is not None:
            outputs[RGB] = splat_results[..., :3]
            if normalize_rgb:
                outputs[RGB] = outputs[RGB] / alpha.clamp_min(1e-8)

        if return_depth:
            outputs[DEPTH] = splat_results[..., -1:] / alpha.clamp_min(1e-8)

        return outputs

    @profiler.time_function
    @torch.inference_mode()
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
            rgbs.permute(0, 1, 3, 4, 2).reshape(S, -1, 3),
            camera_dim,
            not self.config.rgb_only,
            cameras_per_scene,
            bg_colors,
        )

        rendered_rgb = rendering[RGB].permute(0, 3, 1, 2)
        rendered_accumulation = rendering[ACCUMULATION].permute(0, 3, 1, 2)

        if self.config.rgb_only:
            if self.config.rgbd_concat_strategy != "downsample":
                concat_list = self.encode_with_vae(rendered_rgb)
            else:
                concat_list = [rendered_rgb, rendered_accumulation]
        else:
            disparity = (self.encode_depth(rendering[DEPTH] / depth_scaling_factor)).permute(0, 3, 1, 2)
            if self.config.rgbd_concat_strategy != "none":
                encoded = self.encode_with_vae(torch.cat([rendered_rgb, disparity], 1))

                with torch.no_grad():
                    encoded_accumulation = F.interpolate(rendered_accumulation, encoded.shape[2:], mode="bicubic")

                concat_list = [encoded, encoded_accumulation]
            else:
                concat_list = [rendered_rgb, disparity, rendered_accumulation]

        return torch.cat(concat_list, 1), rendering

    def get_outputs(self, cameras: Cameras) -> Dict[str, Union[torch.Tensor, List]]:
        image_gt = cameras.metadata["image"]
        neighbor_cameras = cameras.metadata[NEIGHBOR_CAMERAS]
        bg_colors = cameras.metadata.get(BG_COLOR, None)

        noisy_cond_views = self.config.noisy_cond_views
        if noisy_cond_views > 0:
            neighbor_images = cameras.metadata[NEIGHBOR_IMAGES]

        if self.training:
            camera_w2cs_opencv = torch.eye(4, device=cameras.device).unsqueeze(0).repeat(cameras.shape[0], 1, 1)

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
            scale_factor = 1

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
                    neighbor_cameras.shape[0], noisy_cond_views, 1, 1)
                c2ws_for_noisy[:, :, :3] = neighbor_cameras.camera_to_worlds[:, :noisy_cond_views]
                c2ws_for_noisy[:, :, :, 1:3] *= -1  # opengl to opencv
        else:
            rgbs = torch.cat([cameras.metadata["image"].unsqueeze(1), cameras.metadata[NEIGHBOR_IMAGES]], 1) \
                .permute(0, 1, 4, 2, 3)
            with torch.inference_mode():
                rgbs = self.image_encoder_resize(rgbs.flatten(0, 1))
                rgbs = rgbs.view(cameras.shape[0], -1, *rgbs.shape[1:])

            if FG_MASK in cameras.metadata:
                fg_masks = torch.cat([cameras.metadata[FG_MASK].unsqueeze(1), cameras.metadata[NEIGHBOR_FG_MASK]], 1)
                with torch.inference_mode():
                    fg_masks = F.interpolate(fg_masks.unsqueeze(1).float(), 224).squeeze(1).bool()

                camera_fg_masks = fg_masks[:, 0]
                neighbor_fg_masks = fg_masks[:, 1:]
            else:
                fg_masks = None
                camera_fg_masks = None
                neighbor_fg_masks = None

            c2ws_opencv = torch.eye(4, device=cameras.device).view(1, 1, 4, 4).repeat(rgbs.shape[0], rgbs.shape[1], 1,
                                                                                      1)
            c2ws_opencv[:, 0, :3] = cameras.camera_to_worlds
            c2ws_opencv[:, 1:, :3] = neighbor_cameras.camera_to_worlds
            c2ws_opencv[:, :, :, 1:3] *= -1  # opengl to opencv

            duster_dim = 224
            camera_width = cameras.width[0].item()  # Assume camera dimensions are identical
            camera_height = cameras.height[0].item()
            scale_factor = duster_dim / camera_width
            fx = torch.cat([cameras.fx.unsqueeze(1), neighbor_cameras.fx], 1) * scale_factor
            fy = torch.cat([cameras.fy.unsqueeze(1), neighbor_cameras.fy], 1) * duster_dim / camera_height
            cx = torch.cat([cameras.cx.unsqueeze(1), neighbor_cameras.cx], 1) * scale_factor
            cy = torch.cat([cameras.cy.unsqueeze(1), neighbor_cameras.cy], 1) * duster_dim / camera_height

            all_pts3d, all_depth, alignment_loss, valid_alignment = self.cond_feature_field.get_pts3d_and_depth(
                rgbs, fg_masks, c2ws_opencv, fx, fy, cx, cy)

            camera_depth = all_depth.view(rgbs.shape[0], -1, *all_depth.shape[1:])[:, 0].view(-1, duster_dim,
                                                                                              duster_dim).clamp_min(0)
            neighbor_depth = all_depth.view(rgbs.shape[0], -1, *all_depth.shape[1:])[:, 1:].clamp_min(0)
            neighbor_depth = neighbor_depth.view(*neighbor_depth.shape[:2], duster_dim, duster_dim)

            if self.training:
                max_depth = camera_depth.flatten(1, 2).max(dim=-1)[0]
                valid_alignment = torch.logical_and(valid_alignment, max_depth > 0)
                pts3d = all_pts3d.view(rgbs.shape[0], -1, *all_pts3d.shape[1:])[:, 1:]
                cond_rgbs = rgbs[:, 1:]
                clip_rgbs = cond_rgbs[:, noisy_cond_views:]

                if noisy_cond_views > 0:
                    c2ws_for_noisy = c2ws_opencv[:, 1:noisy_cond_views + 1]

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
            outputs[VALID_ALIGNMENT] = valid_alignment.float().mean()

        with torch.no_grad():
            # Get CLIP embeddings for cross attention
            c_crossattn = self.image_encoder(self.image_encoder_normalize(clip_rgbs.flatten(0, 1))).image_embeds
            c_crossattn = c_crossattn.view(-1, clip_rgbs.shape[1], c_crossattn.shape[-1])

        depth_scaling_factor = torch.cat([camera_depth.unsqueeze(1), neighbor_depth], 1).view(cameras.shape[0],
                                                                                              -1).mean(dim=-1).view(-1,
                                                                                                                    1,
                                                                                                                    1,
                                                                                                                    1)

        if noisy_cond_views > 0:
            pixels_im = torch.stack(
                torch.meshgrid(
                    torch.arange(cameras.height[0].item(), device=cameras.device, dtype=torch.float32),
                    torch.arange(cameras.width[0].item(), device=cameras.device, dtype=torch.float32),
                    indexing="ij")).permute(2, 1, 0)

        if self.training:
            timesteps = torch.randint(0, self.ddpm_scheduler.config.num_train_timesteps, (cond_rgbs.shape[0],),
                                      device=self.device, dtype=torch.long)
            outputs["timesteps"] = timesteps

            opacity = neighbor_fg_masks.view(pts3d[..., :1].shape).float() if neighbor_fg_masks is not None \
                else torch.ones_like(pts3d[..., :1])

            if noisy_cond_views > 0:
                depth_to_noise = neighbor_depth[:, :noisy_cond_views] / depth_scaling_factor

                if depth_to_noise.shape[-1] != cameras.width[0].item():
                    with torch.inference_mode():
                        depth_to_noise = F.interpolate(depth_to_noise, cameras.width[0].item(), mode="bicubic")

                rgb_to_noise = neighbor_images[:, :noisy_cond_views].permute(0, 1, 4, 2, 3)

                rgbd_to_noise = torch.cat([rgb_to_noise.flatten(0, 1),
                                           self.encode_depth(depth_to_noise).flatten(0, 1).unsqueeze(1)], 1)

                latents = self.encode_with_vae(rgbd_to_noise, sample_override=True)
                noise = torch.randn_like(latents)
                noisy_latents = self.ddpm_scheduler.add_noise(latents, noise,
                                                              timesteps.repeat_interleave(noisy_cond_views))

                noisy_rgbd = self.decode_with_vae(noisy_latents)
                noisy_rgb = noisy_rgbd[:, :3].clamp(0, 1)
                noisy_depth = self.decode_depth(noisy_rgbd[:, 3:]) * depth_scaling_factor

                fx_for_noisy = neighbor_cameras.fx[:, :noisy_cond_views].view(-1, 1)
                fy_for_noisy = neighbor_cameras.fy[:, :noisy_cond_views].view(-1, 1)
                cx_for_noisy = neighbor_cameras.cx[:, :noisy_cond_views].view(-1, 1)
                cy_for_noisy = neighbor_cameras.cy[:, :noisy_cond_views].view(-1, 1)

                noisy_pts3d_cam = fast_depthmap_to_pts3d(
                    noisy_depth.view(noisy_depth.shape[0], -1),
                    pixels_im.reshape(-1, 2),
                    torch.cat([fx_for_noisy, fy_for_noisy], 1),
                    torch.cat([cx_for_noisy, cy_for_noisy], 1))

                noisy_pts3d = c2ws_for_noisy.view(-1, 4, 4) @ torch.cat(
                    [noisy_pts3d_cam, torch.ones_like(noisy_pts3d_cam[..., :1])], -1).transpose(1, 2)
                noisy_pts3d = noisy_pts3d.transpose(1, 2)[..., :3].view(cond_rgbs.shape[0], noisy_cond_views, -1, 3)
                noisy_depth = noisy_depth.view(cond_rgbs.shape[0], noisy_cond_views, -1)
                noisy_rgb = noisy_rgb.view(cond_rgbs.shape[0], noisy_cond_views, *noisy_rgb.shape[1:])
                noisy_scales = noisy_depth.flatten(0, 1) / fx_for_noisy
                noisy_opacity = neighbor_fg_masks[:, :noisy_cond_views] \
                    if neighbor_fg_masks is not None else torch.ones_like(noisy_pts3d[..., :1])

                noisy_concat, _ = self.get_concat(
                    camera_w2cs_opencv,
                    cameras,
                    noisy_pts3d,
                    noisy_scales,
                    noisy_opacity,
                    noisy_rgb,
                    1,
                    bg_colors,
                    depth_scaling_factor
                )

                pts3d = pts3d[:, noisy_cond_views:]
                neighbor_depth = neighbor_depth[:, noisy_cond_views:]
                cond_rgbs = cond_rgbs[:, noisy_cond_views:]
                opacity = opacity[:, noisy_cond_views:]

            c_concat, _ = self.get_concat(
                camera_w2cs_opencv,
                cameras,
                pts3d,
                neighbor_depth.flatten(-2, -1) / (neighbor_cameras.fx[:, noisy_cond_views:] * scale_factor),
                opacity,
                cond_rgbs,
                1,
                bg_colors,
                depth_scaling_factor
            )

            if noisy_cond_views > 0:
                noisy_only_mask = torch.rand((c_concat.shape[0], 1, 1, 1), device=c_concat.device)
                c_concat = torch.where(noisy_only_mask < self.config.noisy_only, 0, c_concat)
                c_concat = torch.cat([c_concat, noisy_concat,
                                      self.ddpm_scheduler.alphas_cumprod.to(self.device)[timesteps].view(-1, 1, 1, 1)
                                     .expand(-1, 1, *noisy_concat.shape[2:])], 1)

            random_mask = torch.rand((c_concat.shape[0], 1, 1), device=c_concat.device)
            prompt_mask = random_mask < 2 * self.config.uncond
            input_mask = torch.logical_not(
                torch.logical_and(random_mask >= self.config.uncond,
                                  random_mask < 3 * self.config.uncond)).float().unsqueeze(-1)

            c_crossattn = torch.where(prompt_mask, 0, c_crossattn)
            c_concat = input_mask * c_concat

            # Hack - we include the gt image in the forward loop, so that all model calculation is done in the
            # forward call as expected in DDP
            if self.config.rgb_only:
                input_gt = image_gt.permute(0, 3, 1, 2)
            else:
                depth_gt = camera_depth.unsqueeze(1)
                if depth_gt.shape[-1] != cameras.width[0].item():
                    with torch.no_grad():
                        depth_gt = F.interpolate(depth_gt, cameras.width[0].item(), mode="bicubic")

                input_gt = torch.cat([image_gt.permute(0, 3, 1, 2), self.encode_depth(depth_gt / depth_scaling_factor)],
                                     1)

            latents = self.encode_with_vae(input_gt, sample_override=True)
            noise = torch.randn_like(latents)
            noisy_latents = self.ddpm_scheduler.add_noise(latents, noise, timesteps)

            if self.ddpm_scheduler.config.prediction_type == "epsilon":
                outputs["target"] = noise
            elif self.ddpm_scheduler.config.prediction_type == "v_prediction":
                outputs["target"] = self.ddpm_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {self.ddpm_scheduler.config.prediction_type}")

            outputs["model_output"] = self.unet(torch.cat([noisy_latents, c_concat], 1), timesteps, c_crossattn).sample
        else:
            scales = camera_depth.flatten(1, 2) / (cameras.fx * scale_factor)
            opacity = camera_fg_masks.view(pts3d[..., :1].shape).float() if camera_fg_masks is not None \
                else torch.ones_like(pts3d[..., :1])

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

            c_concat, rendering = self.get_concat(
                neighbor_w2cs,
                neighbor_cameras,
                pts3d,
                scales,
                opacity,
                cond_rgbs,
                neighbor_cameras.shape[1],
                bg_colors,
                depth_scaling_factor
            )

            concat_rgb = [torch.cat([*rendering[RGB]], 1)]
            concat_depth = [torch.cat([*rendering[DEPTH]], 1)]
            concat_acc = [torch.cat([*rendering[ACCUMULATION]], 1)]

            self.ddim_scheduler.set_timesteps(self.config.ddim_steps, device=self.device)

            inference_latents = randn_tensor((neighbor_cameras.shape[1],
                                              self.vae.config.latent_channels,
                                              cameras.height[0].item() // self.vae_scale_factor,
                                              cameras.width[0].item() // self.vae_scale_factor),
                                             device=c_concat.device,
                                             dtype=c_concat.dtype)

            if noisy_cond_views > 0:
                neighbor_c2ws_opencv = torch.eye(4, device=cameras.device).view(1, 4, 4) \
                    .repeat(neighbor_cameras.shape[1], 1, 1)
                neighbor_c2ws_opencv[:, :3] = neighbor_cameras[0].camera_to_worlds
                neighbor_c2ws_opencv[:, :, 1:3] *= -1  # opengl to opencv

                focals_for_decode = torch.cat([neighbor_cameras.fx.view(-1, 1), neighbor_cameras.fy.view(-1, 1)],
                                              -1)
                pp_for_decode = torch.cat([neighbor_cameras.cx.view(-1, 1), neighbor_cameras.cy.view(-1, 1)], -1)
                pixels_for_decode = pixels_im.reshape(-1, 2)
                c_concat_inference = torch.cat(
                    [c_concat, torch.zeros_like(c_concat), torch.zeros_like(c_concat[:, :1])], 1)
            else:
                c_concat_inference = c_concat

            c_crossattn_inference = c_crossattn.expand(neighbor_cameras.shape[1], -1, -1)

            if self.config.guidance_scale > 1:
                c_concat_inference = torch.cat([torch.zeros_like(c_concat_inference), c_concat_inference])
                c_crossattn_inference = torch.cat([torch.zeros_like(c_crossattn_inference), c_crossattn_inference])

            samples_rgb = []
            samples_depth = []

            if self.config.use_ema:
                self.ema_unet.to(self.device)
                self.ema_unet.store(self.unet.parameters())
                self.ema_unet.copy_to(self.unet.parameters())
            try:
                for t_index, t in enumerate(self.ddim_scheduler.timesteps):
                    latent_model_input = torch.cat([inference_latents] * 2) \
                        if self.config.guidance_scale > 1 else inference_latents
                    latent_model_input = self.ddim_scheduler.scale_model_input(latent_model_input, t)
                    latent_model_input = torch.cat([latent_model_input, c_concat_inference], dim=1)
                    noise_pred = self.unet(latent_model_input, t, c_crossattn_inference, return_dict=False)[0]

                    if self.config.guidance_scale > 1:
                        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + self.config.guidance_scale * \
                                     (noise_pred_cond - noise_pred_uncond)

                    inference_latents = self.ddim_scheduler.step(noise_pred, t, inference_latents, return_dict=False)[0]
                    if noisy_cond_views > 0 and t_index < self.config.ddim_steps - 1:
                        decoded = self.decode_with_vae(inference_latents).permute(0, 2, 3, 1)
                        decoded_rgb = decoded[..., :3].clamp(0, 1)
                        decoded_depth = self.decode_depth(decoded[..., 3:]) * depth_scaling_factor
                        decoded_pts3d_cam = fast_depthmap_to_pts3d(
                            decoded_depth.view(decoded_depth.shape[0], -1),
                            pixels_for_decode,
                            focals_for_decode,
                            pp_for_decode)

                        decoded_pts3d = neighbor_c2ws_opencv @ torch.cat(
                            [decoded_pts3d_cam, torch.ones_like(decoded_pts3d_cam[..., :1])], -1).transpose(1, 2)
                        decoded_pts3d = decoded_pts3d.transpose(1, 2)[..., :3]

                        noisy_rgbs = []
                        noisy_scales = []
                        noisy_pts3d = []

                        for neighbor_index in range(neighbor_cameras.shape[1]):
                            for other_neighbor_index in range(neighbor_cameras.shape[1]):
                                if neighbor_index == other_neighbor_index:
                                    continue
                                noisy_rgbs.append(
                                    decoded_rgb[other_neighbor_index:other_neighbor_index + 1].permute(0, 3, 1, 2))
                                noisy_scales.append(
                                    decoded_depth[other_neighbor_index].view(-1, 1) / neighbor_cameras.fx.squeeze(0)[
                                        other_neighbor_index].squeeze(-1))
                                noisy_pts3d.append(decoded_pts3d[other_neighbor_index])

                        noisy_opacity = self.ddim_scheduler.alphas_cumprod.to(self.device)[t]

                        noisy_pts3d = torch.stack(noisy_pts3d)
                        noisy_concat, rendering = self.get_concat(
                            neighbor_w2cs,
                            neighbor_cameras,
                            noisy_pts3d,
                            torch.stack(noisy_scales),
                            torch.ones_like(noisy_pts3d[..., 0]),
                            torch.stack(noisy_rgbs),
                            1,
                            bg_colors,
                            depth_scaling_factor
                        )

                        c_concat_inference = torch.cat(
                            [c_concat, noisy_concat, torch.full_like(c_concat[:, :1], noisy_opacity)], 1)

                        if self.config.guidance_scale > 1:
                            c_concat_inference = torch.cat([torch.zeros_like(c_concat_inference), c_concat_inference])

                        if t_index > self.config.ddim_steps - 10 and t_index % 2 == 0:
                            concat_rgb.append(torch.cat([*rendering[RGB]], 1))
                            concat_depth.append(torch.cat([*rendering[DEPTH]], 1))
                            concat_acc.append(torch.cat([*rendering[ACCUMULATION]], 1))
                            samples_rgb.append(torch.cat([*decoded_rgb], 1))
                            samples_depth.append(torch.cat([*decoded_depth], 1))
            finally:
                if self.config.use_ema:
                    self.ema_unet.restore(self.unet.parameters())

            decoded = self.decode_with_vae(inference_latents).permute(0, 2, 3, 1)
            decoded_rgb = decoded[..., :3].clamp(0, 1)
            samples_rgb.append(torch.cat([*decoded_rgb], 1))
            outputs["samples_rgb"] = torch.cat(samples_rgb)
            outputs[DEPTH_GT] = camera_depth.squeeze(0).unsqueeze(-1)

            if not self.config.rgb_only:
                decoded_depth = self.decode_depth(decoded[..., 3:]) * depth_scaling_factor
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

        loss = F.mse_loss(model_output.float(), target.float(), reduction="mean")
        assert torch.isfinite(loss).all(), loss

        if VALID_ALIGNMENT in outputs and outputs[VALID_ALIGNMENT] == 0:
            CONSOLE.warn("No valid outputs, ignoring training batch")
            loss = loss * 0  # Do this so that backprop doesn't complain

        return {"loss": loss}

    @profiler.time_function
    def get_image_metrics_and_images(
            self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        cond_rgb_gt = batch["image"].to(outputs[RGB].device)
        cond_depth_gt = outputs[DEPTH_GT]
        if cond_depth_gt.shape[0] != cond_rgb_gt.shape[1]:
            cond_depth_gt = F.interpolate(cond_depth_gt.squeeze(-1).unsqueeze(0).unsqueeze(0),
                                          cond_rgb_gt.shape[:2], mode="bicubic").view(*cond_rgb_gt.shape[:2], 1)
        cond_depth_gt = colormaps.apply_depth_colormap(cond_depth_gt)
        cond_gt = torch.cat([cond_rgb_gt, cond_depth_gt], 1)

        image_gt = torch.cat([*batch[NEIGHBOR_IMAGES].squeeze(0).to(outputs[RGB].device)], 1)
        cond_rgb = torch.cat([outputs[RGB], image_gt])
        samples_rgb = torch.cat([outputs["samples_rgb"], image_gt])

        images_dict = {
            "cond_gt": cond_gt,
            "concat_rgb": cond_rgb,
            "samples_rgb": samples_rgb
        }

        if not self.config.rgb_only:
            depth_gt = outputs[NEIGHBOR_DEPTH]
            if depth_gt.shape[1] != cond_rgb_gt.shape[1]:
                depth_gt = F.interpolate(depth_gt.squeeze(-1).unsqueeze(0),
                                         cond_rgb_gt.shape[:2], mode="bicubic").squeeze(0).unsqueeze(-1)
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
