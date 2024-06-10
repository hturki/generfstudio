from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union, Type, Optional, Literal, Any

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, EMAModel, DDIMScheduler
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
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchvision.transforms import transforms
from transformers import CLIPVisionModelWithProjection

from generfstudio.fields.dino_v2_encoder import DinoV2Encoder
from generfstudio.fields.dust3r_field import Dust3rField
from generfstudio.fields.spatial_encoder import SpatialEncoder
from generfstudio.generfstudio_constants import NEIGHBOR_IMAGES, NEIGHBOR_CAMERAS, NEAR, FAR, \
    POSENC_SCALE, RGB, FEATURES, ACCUMULATION, DEPTH, NEIGHBOR_RESULTS, ALIGNMENT_LOSS, NEIGHBOR_PTS3D, VALID_ALIGNMENT, \
    NEIGHBOR_DEPTH, ACCUMULATION_FEATURES, DEPTH_GT, NEIGHBOR_FG_MASK, FG_MASK, BG_COLOR


@dataclass
class RGBDDiffusionConfig(ModelConfig):
    _target: Type = field(default_factory=lambda: RGBDDiffusion)

    uncond: float = 0.05
    ddim_steps = 50

    default_scene_view_count: int = 3

    num_ray_samples: int = 128

    background_color: Literal["random", "last_sample", "black", "white"] = "last_sample"

    cond_image_encoder_type: Literal["unet", "resnet", "dino"] = "resnet"
    cond_feature_out_dim: int = 32

    freeze_cond_image_encoder: bool = False

    unet_pretrained_path: str = "Intel/ldm3d-4c"
    vae_pretrained_path: str = "Intel/ldm3d-4c"
    image_encoder_pretrained_path: str = "lambdalabs/sd-image-variations-diffusers"
    scheduler_pretrained_path: str = "Intel/ldm3d-4c"

    dust3r_model_name: str = "/data/hturki/dust3r/checkpoints/DUSt3R_ViTLarge_BaseDecoder_224_linear.pth"
    intrinsics_from_dust3r: bool = False
    encode_rgbd_vae: bool = True

    guidance_scale: float = 2.0

    use_ema: bool = True
    allow_tf32: bool = False
    rgb_only: bool = False


class RGBDDiffusion(Model):
    config: RGBDDiffusionConfig

    def __init__(self, config: RGBDDiffusionConfig, metadata: Dict[str, Any], **kwargs) -> None:
        if config.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

        self.near = metadata.get(NEAR, None)
        self.far = metadata.get(FAR, None)
        self.posenc_scale = metadata[POSENC_SCALE]
        if self.near is not None or self.far is not None:
            CONSOLE.log(f"Using near and far bounds {self.near} {self.far} from metadata")
        self.depth_available = DEPTH in metadata
        CONSOLE.log(f"Depth available: {self.depth_available}")

        super().__init__(config=config, **kwargs)

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()
        self.collider = None

        self.vae = AutoencoderKL.from_pretrained(self.config.vae_pretrained_path, subfolder="vae")
        self.vae.requires_grad_(False)

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
                                                         in_channels=4,
                                                         low_cpu_mem_usage=False,
                                                         ignore_mismatched_sizes=False)

        self.unet = UNet2DConditionModel.from_pretrained(self.config.unet_pretrained_path, subfolder="unet",
                                                         in_channels=9 + self.config.cond_feature_out_dim,
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

        # self.unet.enable_xformers_memory_efficient_attention()

        if self.config.use_ema:
            ema_unet = UNet2DConditionModel.from_pretrained(self.config.unet_pretrained_path, subfolder="unet",
                                                            in_channels=9 + self.config.cond_feature_out_dim,
                                                            low_cpu_mem_usage=False,
                                                            ignore_mismatched_sizes=True)
            if old_state[in_key].shape != new_state[in_key].shape:
                ema_unet.load_state_dict(new_state)

            self.ema_unet = EMAModel(ema_unet.parameters(), model_cls=UNet2DConditionModel,
                                     model_config=ema_unet.config)

        self.noise_scheduler = DDPMScheduler.from_pretrained(self.config.scheduler_pretrained_path,
                                                             subfolder="scheduler")
        self.ddim_scheduler = DDIMScheduler.from_pretrained(self.config.scheduler_pretrained_path,
                                                            subfolder="scheduler")

        if self.config.cond_feature_out_dim > 0:
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
                                              in_feature_dim=encoder_dim if self.config.cond_feature_out_dim > 0 else 0,
                                              out_feature_dim=self.config.cond_feature_out_dim,
                                              intrinsics_from_dust3r=self.config.intrinsics_from_dust3r,
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
    def get_cond_features(self, cameras: Cameras, cond_rgbs: torch.Tensor):
        if self.config.cond_feature_out_dim > 0 and self.config.cond_image_encoder_type == "dino":
            # We're going to need to resize to 224 for both dino and dust3r - might as well only do it once
            if cond_rgbs.shape[-1] != self.cond_feature_field.image_dim:
                assert not DEPTH in cameras.metadata
                with profiler.time_function("interpolate_rgb_for_dust3r"), torch.no_grad():
                    cond_rgbs = F.interpolate(cond_rgbs.view(-1, *cond_rgbs.shape[2:]),
                                              self.cond_feature_field.image_dim, mode="bicubic")
                    cond_rgbs = cond_rgbs.view(len(cameras), -1, *cond_rgbs.shape[1:])

        if self.config.cond_feature_out_dim > 0:
            if self.config.freeze_cond_image_encoder:
                with torch.no_grad():
                    cond_features = self.cond_image_encoder(cond_rgbs.view(-1, *cond_rgbs.shape[2:]))
            else:
                cond_features = self.cond_image_encoder(cond_rgbs.view(-1, *cond_rgbs.shape[2:]))
        else:
            cond_features = None

        target_cameras = deepcopy(cameras)
        target_cameras.metadata = {}

        if self.config.encode_rgbd_vae:
            target_cameras_feature = deepcopy(target_cameras)
            target_cameras_feature.rescale_output_resolution(32 / cameras.height[0, 0].item())
        else:
            target_cameras.rescale_output_resolution(32 / cameras.height[0, 0].item())
            target_cameras_feature = None

        neighbor_cameras = cameras.metadata[NEIGHBOR_CAMERAS]

        return self.cond_feature_field(target_cameras, neighbor_cameras, cond_rgbs, cond_features,
                                       cameras.metadata.get(NEIGHBOR_PTS3D, None),
                                       cameras.metadata.get(NEIGHBOR_DEPTH, None),
                                       cameras.metadata["image"].permute(0, 3, 1, 2),
                                       cameras.metadata.get(FG_MASK, None),
                                       cameras.metadata.get(NEIGHBOR_FG_MASK, None),
                                       cameras.metadata.get(BG_COLOR, None),
                                       (not self.config.rgb_only or not self.training),
                                       target_cameras_feature)

    def get_outputs(self, cameras: Cameras) -> Dict[str, Union[torch.Tensor, List]]:
        cam_cond = cameras.metadata[NEIGHBOR_CAMERAS].camera_to_worlds
        neighbor_count = cam_cond.shape[1]

        image_cond = cameras.metadata[NEIGHBOR_IMAGES].permute(0, 1, 4, 2, 3)

        with torch.no_grad():
            image_cond_flat = self.image_encoder_resize(image_cond.view(-1, *image_cond.shape[2:]))
        image_cond = image_cond_flat.view(len(cameras), -1, *image_cond_flat.shape[1:])
        cond_features = self.get_cond_features(cameras, image_cond)

        valid_alignment = cond_features[VALID_ALIGNMENT]
        outputs = {
            ALIGNMENT_LOSS: cond_features[ALIGNMENT_LOSS],
            VALID_ALIGNMENT: valid_alignment.float().mean()
        }

        if not torch.any(valid_alignment):
            valid_alignment = torch.ones_like(valid_alignment)

        is_training = (not torch.is_inference_mode_enabled()) and self.training

        if is_training:
            image_cond = image_cond[valid_alignment]
            image_cond_flat = image_cond.view(-1, *image_cond.shape[2:])
        elif not self.config.intrinsics_from_dust3r:
            outputs[NEIGHBOR_RESULTS] = cond_features[NEIGHBOR_RESULTS]

        with torch.no_grad():
            # Get CLIP embeddings for cross attention
            c_crossattn = self.image_encoder(self.image_encoder_normalize(image_cond_flat)).image_embeds
            c_crossattn = c_crossattn.view(-1, neighbor_count, c_crossattn.shape[-1])

        if self.config.rgb_only:
            if self.config.encode_rgbd_vae:
                with torch.cuda.amp.autocast(enabled=False):
                    concat_list = [self.vae.encode(cond_features[RGB].permute(0, 3, 1, 2) * 2 - 1).latent_dist.mode()
                                   * self.vae.config.scaling_factor]
            else:
                concat_list = [cond_features[RGB].permute(0, 3, 1, 2) * 2 - 1]
        else:
            if NEIGHBOR_DEPTH in cameras.metadata:
                # depth_scaling_factor = \
                #     cameras.metadata[DEPTH if is_training else NEIGHBOR_DEPTH].view(len(cameras), -1).max(dim=-1)[0] \
                #         .view(-1, 1, 1, 1)
                depth_scaling_factor = cameras.metadata[NEIGHBOR_DEPTH].view(len(cameras), -1).mean(dim=-1) \
                                           .view(-1, 1, 1, 1) * 2
                if self.training:
                    depth_scaling_factor = depth_scaling_factor[valid_alignment]
            else:
                depth_scaling_factor = cond_features[DEPTH_GT].mean(dim=-1).view(-1, 1, 1, 1) * 2

            cond_depth = (cond_features[DEPTH] / depth_scaling_factor).permute(0, 3, 1, 2)
            if self.config.encode_rgbd_vae:
                with torch.cuda.amp.autocast(enabled=False):
                    concat_list = [self.vae.encode(torch.cat([
                        cond_features[RGB].permute(0, 3, 1, 2),
                        cond_depth,
                    ], 1) * 2 - 1).latent_dist.mode() * self.vae.config.scaling_factor]
            else:
                concat_list = [cond_features[RGB].permute(0, 3, 1, 2) * 2 - 1,
                               cond_depth * 2 - 1]

        concat_list.append(cond_features[ACCUMULATION_FEATURES].permute(0, 3, 1, 2))

        if self.config.cond_feature_out_dim > 0:
            concat_list.append(cond_features[FEATURES].permute(0, 3, 1, 2))

        c_concat = torch.cat(concat_list, 1)

        if is_training:
            # To support classifier-free guidance, randomly drop out only concat conditioning 5%, only cross_attn conditioning 5%, and both 5%.
            random_mask = torch.rand((image_cond.shape[0], 1, 1), device=image_cond.device)
            prompt_mask = random_mask < 2 * self.config.uncond
            input_mask = torch.logical_not(
                torch.logical_and(random_mask >= self.config.uncond,
                                  random_mask < 3 * self.config.uncond)).float().unsqueeze(-1)

            c_crossattn = torch.where(prompt_mask, 0, c_crossattn)
            c_concat = input_mask * c_concat

        if DEPTH_GT in cond_features:
            depth_gt = cond_features[DEPTH_GT]
            with torch.no_grad():
                depth_gt = F.interpolate(depth_gt.view(-1, self.cond_feature_field.image_dim,
                                                       self.cond_feature_field.image_dim).unsqueeze(1),
                                         cameras.metadata["image"].shape[1:3],
                                         mode="bilinear").squeeze(1).unsqueeze(-1)
                outputs[DEPTH_GT] = depth_gt

        if is_training:
            # Hack - we include the gt image in the forward loop, so that all model calculation is done in the
            # forward call as expected in DDP
            image_gt = cameras.metadata["image"][valid_alignment]
            if self.config.rgb_only:
                input_gt = image_gt.permute(0, 3, 1, 2) * 2 - 1
            else:
                if DEPTH in cameras.metadata:
                    depth_gt = cameras.metadata[DEPTH][valid_alignment] / depth_scaling_factor

                input_gt = torch.cat([image_gt.permute(0, 3, 1, 2), depth_gt.permute(0, 3, 1, 2)], 1) * 2 - 1

            # VAE operations should be upcasted
            with torch.cuda.amp.autocast(enabled=False):
                latents = self.vae.encode(input_gt).latent_dist.sample() * self.vae.config.scaling_factor

            timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (latents.shape[0],),
                                      device=self.device, dtype=torch.long)
            outputs["timesteps"] = timesteps
            noise = torch.randn_like(latents)

            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

            if self.noise_scheduler.config.prediction_type == "epsilon":
                outputs["target"] = noise
            elif self.noise_scheduler.config.prediction_type == "v_prediction":
                outputs["target"] = self.noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

            outputs["model_output"] = self.unet(torch.cat([noisy_latents, c_concat], 1), timesteps, c_crossattn).sample
        else:
            if self.config.use_ema:
                self.ema_unet.to(self.device)
                self.ema_unet.store(self.unet.parameters())
                self.ema_unet.copy_to(self.unet.parameters())
            try:
                self.ddim_scheduler.set_timesteps(self.config.ddim_steps, device=self.device)
                n_samples = cameras.metadata[NEIGHBOR_CAMERAS].shape[1] + 1
                vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
                inference_latents = randn_tensor((n_samples, self.vae.config.latent_channels,
                                                  cameras.height[0, 0].item() // vae_scale_factor,
                                                  cameras.width[0, 0].item() // vae_scale_factor),
                                                 device=c_concat.device,
                                                 dtype=c_concat.dtype)

                c_concat_inference = c_concat.expand(n_samples, -1, -1, -1)
                c_crossattn_inference = c_crossattn.expand(n_samples, -1, -1)

                if self.config.guidance_scale > 1:
                    c_concat_inference = torch.cat([torch.zeros_like(c_concat_inference), c_concat_inference])
                    c_crossattn_inference = torch.cat([torch.zeros_like(c_crossattn_inference), c_crossattn_inference])

                for t in self.ddim_scheduler.timesteps:
                    latent_model_input = torch.cat([inference_latents] * 2) \
                        if self.config.guidance_scale > 1 else inference_latents
                    latent_model_input = self.ddim_scheduler.scale_model_input(latent_model_input, t)
                    latent_model_input = torch.cat([latent_model_input, c_concat_inference], dim=1)
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        c_crossattn_inference,
                        return_dict=False,
                    )[0]

                    if self.config.guidance_scale > 1:
                        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + self.config.guidance_scale * (
                                noise_pred_cond - noise_pred_uncond)

                    inference_latents = self.ddim_scheduler.step(noise_pred, t, inference_latents, return_dict=False)[0]

                with torch.cuda.amp.autocast(enabled=False):
                    decoded = (self.vae.decode(inference_latents / self.vae.config.scaling_factor, return_dict=False)[
                                   0].permute(0, 2, 3, 1) + 1) / 2

                outputs["samples_rgb"] = decoded[..., :3].clamp(0, 1)
                if not self.config.rgb_only:
                    outputs["samples_depth"] = decoded[..., 3:].clamp_min(0) * depth_scaling_factor
                outputs[RGB] = cond_features[RGB]
                outputs[ACCUMULATION] = cond_features[ACCUMULATION]
                outputs[DEPTH] = cond_features[DEPTH]
            finally:
                if self.config.use_ema:
                    self.ema_unet.restore(self.unet.parameters())

        return outputs

    @torch.no_grad()
    def get_outputs_for_camera(self, camera: Cameras, obb_box: Optional[OrientedBox] = None) -> Dict[str, torch.Tensor]:
        return self.get_outputs(camera)

    def get_metrics_dict(self, outputs, batch):
        assert self.training

        return {ALIGNMENT_LOSS: outputs[ALIGNMENT_LOSS], VALID_ALIGNMENT: outputs[VALID_ALIGNMENT]}

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        model_output = outputs["model_output"]
        target = outputs["target"]

        loss = F.mse_loss(model_output.float(), target.float(), reduction="mean")
        assert torch.isfinite(loss).all(), loss

        if outputs[VALID_ALIGNMENT] == 0:
            CONSOLE.warn("No valid outputs, ignoring training batch")
            loss = loss * 0  # Do this so that backprop doesn't complain

        return {"loss": loss}

    @profiler.time_function
    def get_image_metrics_and_images(
            self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        image = batch["image"].to(outputs[RGB].device)

        cond_rgb = outputs[RGB]
        if self.config.encode_rgbd_vae:
            cond_rgb_gt = image
        else:
            cond_rgb_gt = F.interpolate(image.unsqueeze(0).permute(0, 3, 1, 2), cond_rgb.shape[1:3], mode="bicubic",
                                        antialias=True).permute(0, 2, 3, 1)
        metrics_dict = {
            "psnr_cond_image": self.psnr(cond_rgb_gt, cond_rgb)
        }

        cond_depth = colormaps.apply_depth_colormap(
            outputs[DEPTH].squeeze(0),
            accumulation=outputs[ACCUMULATION].squeeze(0),
        )

        images_dict = {"cond_img": torch.cat([cond_rgb_gt.squeeze(0), cond_rgb.squeeze(0), cond_depth], 1)}

        if not self.config.intrinsics_from_dust3r:
            neighbor_splat_depth = colormaps.apply_depth_colormap(
                outputs[NEIGHBOR_RESULTS][DEPTH].squeeze(0),
                accumulation=outputs[NEIGHBOR_RESULTS][ACCUMULATION].squeeze(0),
            )
            images_dict["neighbor_splats"] = torch.cat([outputs[NEIGHBOR_RESULTS][RGB].squeeze(0), neighbor_splat_depth],
                                                   0)

        rgb_row_1 = torch.cat([image, *batch[NEIGHBOR_IMAGES].squeeze(0)], 1)
        rgb_row_2 = torch.cat([*outputs["samples_rgb"]], 1)

        images_dict["img"] = torch.cat([rgb_row_1, rgb_row_2], dim=0)

        if not self.config.rgb_only:
            if NEIGHBOR_DEPTH in batch:
                # neighbor_depth = batch[NEIGHBOR_DEPTH].view(batch[NEIGHBOR_DEPTH].shape[0], 1, 224, 224)
                # neighbor_depth = F.interpolate(neighbor_depth, batch[DEPTH].shape[:2], mode="bilinear").squeeze(
                #     1).unsqueeze(-1)
                depth_row_1 = torch.cat([batch[DEPTH], *batch[NEIGHBOR_DEPTH]], 1)
                depth_row_2 = torch.cat([*outputs["samples_depth"]], 1)

                images_dict["depth"] = colormaps.apply_depth_colormap(torch.cat([depth_row_1, depth_row_2], dim=0))
            else:
                images_dict["depth"] = colormaps.apply_depth_colormap(
                    torch.cat([outputs[DEPTH_GT].squeeze(0), *outputs["samples_depth"]], 1))

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]

        avg_metrics_dict = defaultdict(list)
        for i, sample in enumerate(outputs["samples_rgb"]):
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

        return metrics_dict, images_dict

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
