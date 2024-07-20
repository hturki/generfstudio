from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Literal, Any

import math
import torch
from diffusers import DDPMScheduler, DDIMScheduler
from gsplat import fully_fused_projection, isect_tiles, isect_offset_encode, rasterize_to_pixels
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.engine.callbacks import TrainingCallbackAttributes, TrainingCallback, TrainingCallbackLocation
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import profiler
from nerfstudio.utils.rich_utils import CONSOLE
from torchvision.transforms import transforms

from generfstudio.fields.batched_pc_optimizer import fast_depthmap_to_pts3d
from generfstudio.fields.dust3r_field import Dust3rField
from generfstudio.generfstudio_constants import DEPTH, ACCUMULATION, RGB


@dataclass
class RGBDDiffusionBaseConfig(ModelConfig):
    # dust3r_model_name: str = "/data/hturki/dust3r/checkpoints/DUSt3R_ViTLarge_BaseDecoder_224_linear.pth"
    dust3r_model_name: str = "/data/hturki/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
    min_conf_thresh: float = 1.5
    uncond: float = 0.05

    unet_pretrained_path: str = "Intel/ldm3d-4c"
    use_ddim: bool = True
    inference_steps: int = 50
    beta_schedule: str = "scaled_linear"
    prediction_type: str = "epsilon"

    depth_mapping: Literal["scaled", "disparity", "log"] = "disparity"

    scale_with_pixel_area: bool = True

    guidance_scale: float = 2.0

    use_ema: bool = True
    allow_tf32: bool = False
    rgb_only: bool = False


class RGBDDiffusionBase(Model):
    config: RGBDDiffusionBaseConfig

    def __init__(self, config: RGBDDiffusionBaseConfig, metadata: Dict[str, Any], **kwargs) -> None:
        if config.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

        self.depth_available = DEPTH in metadata
        CONSOLE.log(f"Depth available: {self.depth_available}")

        super().__init__(config=config, **kwargs)

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()
        self.collider = None

        self.ddpm_scheduler = DDPMScheduler.from_pretrained(self.config.unet_pretrained_path,
                                                            subfolder="scheduler",
                                                            beta_schedule=self.config.beta_schedule,
                                                            prediction_type=self.config.prediction_type,
                                                            variance_type="fixed_small")
        if self.config.use_ddim:
            self.ddim_scheduler = DDIMScheduler.from_pretrained(self.config.unet_pretrained_path,
                                                                subfolder="scheduler",
                                                                beta_schedule=self.config.beta_schedule,
                                                                prediction_type=self.config.prediction_type)

        self.dust3r_field = Dust3rField(model_name=self.config.dust3r_model_name,
                                        depth_precomputed=self.depth_available)
        self.dust3r_resize = transforms.Resize(
            (224, 224),
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=True,
        )

        # https://huggingface.co/lambdalabs/sd-image-variations-diffusers - apparently the image encoder
        # was trained without anti-aliasing
        self.clip_resize = transforms.Resize(
            (224, 224),
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=False,
        )
        self.clip_normalize = transforms.Normalize(
            [0.48145466, 0.4578275, 0.40821073],
            [0.26862954, 0.26130258, 0.27577711])

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

    def get_depth_scaling_factor(self, depth_to_scale: torch.Tensor) -> torch.Tensor:
        if self.config.depth_mapping == "scaled":
            return torch.quantile(depth_to_scale, torch.FloatTensor([0.02, 0.98]).to(depth_to_scale.device),
                                  dim=-1).T.view(-1, 2, 1, 1)
        elif self.config.depth_mapping == "disparity" or self.config.depth_mapping == "log":
            return depth_to_scale.mean(dim=-1).view(-1, 1, 1, 1)
        else:
            raise Exception(self.config.depth_mapping)

    def encode_depth(self, to_encode: torch.Tensor, depth_scaling_factor: torch.Tensor) -> torch.Tensor:
        if self.config.depth_mapping == "scaled":
            p2 = depth_scaling_factor[:, :1]
            p98 = depth_scaling_factor[:, 1:]
            encoded_0_1 = (to_encode - p2) / (p98 - p2).clamp_min(1e-8)
            return encoded_0_1 * 2 - 1
        elif self.config.depth_mapping == "disparity":
            scaled_depth = to_encode / depth_scaling_factor
            encoded_0_1 = 1 - 1 / (1 + scaled_depth.float())
            return encoded_0_1 * 2 - 1
        elif self.config.depth_mapping == "log":
            scaled_depth = to_encode / depth_scaling_factor
            return (scaled_depth.float() + 1e-3).log()
        else:
            raise Exception(self.config.depth_mapping)

    def decode_depth(self, to_decode: torch.Tensor, depth_scaling_factor: torch.Tensor) -> torch.Tensor:
        if self.config.depth_mapping == "scaled":
            p2 = depth_scaling_factor[:, :1]
            p98 = depth_scaling_factor[:, 1:]
            to_decode_0_1 = 0.5 * (to_decode.float() + 1)
            return (to_decode_0_1 * (p98 - p2) + p2).clamp_min(0)
        elif self.config.depth_mapping == "disparity":
            to_decode_0_1 = 0.5 * (to_decode.float() + 1)
            scaled = 1 / (1 - to_decode_0_1.clamp(0, 0.999)) - 1
            return scaled * depth_scaling_factor
        elif self.config.depth_mapping == "log":
            scaled = (to_decode.float().exp() - 1e-3).clamp_min(0)
            return scaled * depth_scaling_factor
        else:
            raise Exception(self.config.depth_mapping)

    def get_pixel_area_scale_mult(self, cameras: Cameras, camera_dim: int):
        if cameras.width[0, 0].item() != camera_dim:
            cameras = copy.deepcopy(cameras)
            cameras.rescale_output_resolution(camera_dim / cameras.width[0, 0].item())

        ray_bundle = cameras.generate_rays(camera_indices=torch.arange(len(cameras)).view(-1, 1))

        pixel_area = ray_bundle.pixel_area.squeeze(-1).permute(2, 0, 1)
        directions_norm = ray_bundle.metadata["directions_norm"].squeeze(-1).permute(2, 0, 1)
        return pixel_area * directions_norm

    @torch.cuda.amp.autocast(enabled=False)
    def c2ws_opengl_to_w2cs_opencv(self, c2ws: torch.Tensor) -> torch.Tensor:
        w2cs = torch.eye(4, device=c2ws.device).view(1, 1, 4, 4).repeat(c2ws.shape[0], c2ws.shape[1], 1, 1)
        R = c2ws[:, :, :3, :3].clone()
        R[:, :, :, 1:3] *= -1  # opengl to opencv
        R_inv = R.transpose(2, 3)
        w2cs[:, :, :3, :3] = R_inv
        T = c2ws[:, :, :3, 3:]
        inv_T = -torch.bmm(R_inv.view(-1, 3, 3), T.view(-1, 3, 1))
        w2cs[:, :, :3, 3:] = inv_T.view(w2cs[:, :, :3, 3:].shape)
        return w2cs

    @torch.cuda.amp.autocast(enabled=False)
    def depth_to_pts3d(self, depth: torch.Tensor, c2ws_opencv: torch.Tensor, pixels: torch.Tensor, focals: torch.Tensor,
                       pp: torch.Tensor) -> torch.Tensor:
        pts3d_cam = fast_depthmap_to_pts3d(
            depth.view(depth.shape[0], -1),
            pixels,
            focals,
            pp)
        pts3d = (c2ws_opencv @ torch.cat([pts3d_cam, torch.ones_like(pts3d_cam[..., :1])], -1).transpose(1, 2))
        return pts3d.transpose(1, 2)[..., :3]

    @profiler.time_function
    @torch.cuda.amp.autocast(enabled=False)
    def splat_gaussians(self, w2cs: torch.Tensor, fx: torch.Tensor, fy: torch.Tensor, cx: torch.Tensor,
                        cy: torch.Tensor, xyz: torch.Tensor, scales: torch.Tensor, opacity: torch.Tensor,
                        rgbs: Optional[torch.Tensor], camera_dim: int, return_depth: bool, cameras_per_scene: int,
                        bg_colors: Optional[torch.Tensor]):
        quats = torch.cat([torch.ones_like(xyz[..., :1]), torch.zeros_like(xyz)], -1)

        zeros = torch.zeros_like(fx)
        ones = torch.ones_like(fx)

        K = torch.cat([
            torch.cat([fx, zeros, cx], -1).unsqueeze(1),
            torch.cat([zeros, fy, cy], -1).unsqueeze(1),
            torch.cat([zeros, zeros, ones], -1).unsqueeze(1)
        ], 1)

        radii, means2d, depths, conics, compensations = fully_fused_projection(
            xyz, None, quats, scales, w2cs, K, camera_dim, camera_dim, cameras_per_scene=cameras_per_scene)

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
            outputs[RGB] = splat_results[..., :3] / alpha.clamp_min(1e-8)

        if return_depth:
            outputs[DEPTH] = splat_results[..., -1:] / alpha.clamp_min(1e-8)

        return outputs

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)

        # state_dict = {}
        # for key, val in base_state_dict.items():
        #     if "dust3r_field." in key:
        #         print("SKIP", key)
        #         continue
        #     state_dict[key] = val

        if self.config.use_ema:
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
