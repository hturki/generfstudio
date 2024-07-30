# ruff: noqa: E741
# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
NeRF implementation that combines many recent advancements.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type, Union, Any

import math
import numpy as np
import torch
from diffusers import AutoencoderKL, UNet2DConditionModel, EMAModel, DDPMScheduler, DDIMScheduler
from diffusers.utils.torch_utils import randn_tensor
from gsplat.cuda_legacy._torch_impl import quat_to_rotmat
from transformers import CLIPVisionModelWithProjection

try:
    from gsplat.rendering import rasterization
except ImportError:
    print("install via: pip install git+https://github.com/nerfstudio-project/gsplat.git@v1.0")
from gsplat.cuda_legacy._wrapper import num_sh_bases
from pytorch_msssim import SSIM
from torch.nn import Parameter
from typing_extensions import Literal

from nerfstudio.cameras.camera_optimizers import CameraOptimizer, CameraOptimizerConfig
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.engine.optimizers import Optimizers

# need following import for background color override
from nerfstudio.model_components import renderers
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils.colors import get_color
from nerfstudio.utils.rich_utils import CONSOLE
from torchvision.transforms import transforms
from PIL import Image
from torch.nn import functional as F


def random_quat_tensor(N, device="cpu"):
    """
    Defines a random quaternion tensor of shape (N, 4)
    """
    u = torch.rand(N, device=device)
    v = torch.rand(N, device=device)
    w = torch.rand(N, device=device)
    return torch.stack(
        [
            torch.sqrt(1 - u) * torch.sin(2 * math.pi * v),
            torch.sqrt(1 - u) * torch.cos(2 * math.pi * v),
            torch.sqrt(u) * torch.sin(2 * math.pi * w),
            torch.sqrt(u) * torch.cos(2 * math.pi * w),
        ],
        dim=-1,
    )


def RGB2SH(rgb):
    """
    Converts from RGB values [0,1] to the 0th spherical harmonic coefficient
    """
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def SH2RGB(sh):
    """
    Converts from the 0th spherical harmonic coefficient to RGB values [0,1]
    """
    C0 = 0.28209479177387814
    return sh * C0 + 0.5


def resize_image(image: torch.Tensor, d: int):
    """
    Downscale images using the same 'area' method in opencv

    :param image shape [H, W, C]
    :param d downscale factor (must be 2, 4, 8, etc.)

    return downscaled image in shape [H//d, W//d, C]
    """
    import torch.nn.functional as tf

    weight = (1.0 / (d * d)) * torch.ones((1, 1, d, d), dtype=torch.float32, device=image.device)
    return tf.conv2d(image.permute(2, 0, 1)[:, None, ...], weight, stride=d).squeeze(1).permute(1, 2, 0)


@torch.compile()
def get_viewmat(optimized_camera_to_world):
    """
    function that converts c2w to gsplat world2camera matrix, using compile for some speed
    """
    R = optimized_camera_to_world[:, :3, :3]  # 3 x 3
    T = optimized_camera_to_world[:, :3, 3:4]  # 3 x 1
    # flip the z and y axes to align with gsplat conventions
    R = R * torch.tensor([[[1, -1, -1]]], device=R.device, dtype=R.dtype)
    # analytic matrix inverse to get world2camera matrix
    R_inv = R.transpose(1, 2)
    T_inv = -torch.bmm(R_inv, T)
    viewmat = torch.zeros(R.shape[0], 4, 4, device=R.device, dtype=R.dtype)
    viewmat[:, 3, 3] = 1.0  # homogenous
    viewmat[:, :3, :3] = R_inv
    viewmat[:, :3, 3:4] = T_inv
    return viewmat


@dataclass
class SplatfactoSDSModelConfig(ModelConfig):
    """SplatfactoSDS Model Config, nerfstudio's implementation of Gaussian Splatting"""

    _target: Type = field(default_factory=lambda: SplatfactoSDSModel)
    warmup_length: int = 500
    """period of steps where refinement is turned off"""
    refine_every: int = 100
    """period of steps where gaussians are culled and densified"""
    resolution_schedule: int = 3000
    """training starts at 1/d resolution, every n steps this is doubled"""
    background_color: Literal["random", "black", "white"] = "random"
    """Whether to randomize the background color."""
    num_downscales: int = 2
    """at the beginning, resolution is 1/2^d, where d is this number"""
    cull_alpha_thresh: float = 0.1
    """threshold of opacity for culling gaussians. One can set it to a lower value (e.g. 0.005) for higher quality."""
    cull_scale_thresh: float = 0.5
    """threshold of scale for culling huge gaussians"""
    continue_cull_post_densification: bool = True
    """If True, continue to cull gaussians post refinement"""
    reset_alpha_every: int = 30
    """Every this many refinement steps, reset the alpha"""
    densify_grad_thresh: float = 0.0008
    """threshold of positional gradient norm for densifying gaussians"""
    densify_size_thresh: float = 0.01
    """below this size, gaussians are *duplicated*, otherwise split"""
    n_split_samples: int = 2
    """number of samples to split gaussians into"""
    sh_degree_interval: int = 1000
    """every n intervals turn on another sh degree"""
    split_screen_size: float = 0.05
    """if a gaussian is more than this percent of screen space, split it"""
    stop_screen_size_at: int = 4000
    """stop culling/splitting at this step WRT screen size of gaussians"""
    random_init: bool = False
    """whether to initialize the positions uniformly randomly (not SFM points)"""
    num_random: int = 50000
    """Number of gaussians to initialize if random init is used"""
    random_scale: float = 10.0
    "Size of the cube to initialize random gaussians within"
    ssim_lambda: float = 0.2
    """weight of ssim loss"""
    stop_split_at: int = 15000
    """stop splitting at this step"""
    sh_degree: int = 0
    """maximum degree of spherical harmonics to use"""
    use_scale_regularization: bool = False
    """If enabled, a scale regularization introduced in PhysGauss (https://xpandora.github.io/PhysGaussian/) is used for reducing huge spikey gaussians."""
    max_gauss_ratio: float = 10.0
    """threshold of ratio of gaussian max to min scale before applying regularization
    loss from the PhysGaussian paper
    """
    output_depth_during_training: bool = False
    """If True, output depth during training. Otherwise, only output depth during evaluation."""
    rasterize_mode: Literal["classic", "antialiased"] = "classic"
    """
    Classic mode of rendering will use the EWA volume splatting with a [0.3, 0.3] screen space blurring kernel. This
    approach is however not suitable to render tiny gaussians at higher or lower resolution than the captured, which
    results "aliasing-like" artifacts. The antialiased mode overcomes this limitation by calculating compensation factors
    and apply them to the opacities of gaussians to preserve the total integrated density of splats.

    However, PLY exported with antialiased rasterize mode is not compatible with classic mode. Thus many web viewers that
    were implemented for classic mode can not render antialiased mode PLY properly without modifications.
    """
    camera_optimizer: CameraOptimizerConfig = field(default_factory=lambda: CameraOptimizerConfig(mode="off"))
    """Config of the camera optimizer to use"""

    # diffusion_checkpoint_path: Path = Path(
    #     "outputs/rgbd-diffusion-union-no-oxl-resnet-False-ldm3d-nv-1-DI-False-4-nofeat/rgbd-diffusion-union-ddp/2024-06-08_163402/nerfstudio_models/step-000049000.ckpt")
    cond_feature_out_dim: int = 0
    ddim_steps: int = 50
    guidance_scale: float = 3.0

    min_step_percent: Tuple[float, float] = (0.7, 0.3)
    max_step_percent: Tuple[float, float] = (0.98, 0.8)
    step_anneal_range: Tuple[int, int] = (50, 200)
    random_camera_batch_size: int = 12
    sds_loss_mult: float = 1e-4

class SplatfactoSDSModel(Model):
    """Nerfstudio's implementation of Gaussian Splatting

    Args:
        config: SplatfactoSDS configuration to instantiate model
    """

    config: SplatfactoSDSModelConfig

    def __init__(
            self,
            *args,
            seed_points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            metadata: Dict[str, Any],
            **kwargs,
    ):
        self.seed_points = seed_points
        self.points3D_scale = metadata.get("points3D_scale", None)
        self.train_image_filenames = metadata["train_image_filenames"]
        self.sds_cameras = deepcopy(metadata["train_image_cameras_all"]).to("cuda")

        cropped_dim = torch.minimum(self.sds_cameras.width, self.sds_cameras.height)
        self.sds_cameras.cx -= ((self.sds_cameras.width - cropped_dim) / 2)
        self.sds_cameras.cy -= ((self.sds_cameras.height - cropped_dim) / 2)
        self.sds_cameras.cx *= (256 / cropped_dim)
        self.sds_cameras.cy *= (256 / cropped_dim)
        self.sds_cameras.fx *= (256 / cropped_dim)
        self.sds_cameras.fy *= (256 / cropped_dim)
        self.sds_cameras.width.fill_(256)
        self.sds_cameras.height.fill_(256)

        super().__init__(*args, **kwargs)

    def populate_modules(self):
        if self.seed_points is not None and not self.config.random_init:
            means = torch.nn.Parameter(self.seed_points[0])  # (Location, Color)
        else:
            means = torch.nn.Parameter((torch.rand((self.config.num_random, 3)) - 0.5) * self.config.random_scale)
        self.xys_grad_norm = None
        if self.points3D_scale is not None:
            scales = torch.nn.Parameter(torch.log(self.points3D_scale.repeat(1, 3)))
        else:
            distances, _ = self.k_nearest_sklearn(means.data, 3)
            distances = torch.from_numpy(distances)
            # find the average of the three nearest neighbors for each point and use that as the scale
            avg_dist = distances.mean(dim=-1, keepdim=True)
            scales = torch.nn.Parameter(torch.log(avg_dist.repeat(1, 3)))

        num_points = means.shape[0]
        quats = torch.nn.Parameter(random_quat_tensor(num_points))
        dim_sh = num_sh_bases(self.config.sh_degree)

        if (
                self.seed_points is not None
                and not self.config.random_init
                # We can have colors without points.
                and self.seed_points[1].shape[0] > 0
        ):
            shs = torch.zeros((self.seed_points[1].shape[0], dim_sh, 3)).float().cuda()
            if self.config.sh_degree > 0:
                shs[:, 0, :3] = RGB2SH(self.seed_points[1] / 255)
                shs[:, 1:, 3:] = 0.0
            else:
                CONSOLE.log("use color only optimization with sigmoid activation")
                shs[:, 0, :3] = torch.logit(self.seed_points[1] / 255, eps=1e-10)
            features_dc = torch.nn.Parameter(shs[:, 0, :])
            features_rest = torch.nn.Parameter(shs[:, 1:, :])
        else:
            features_dc = torch.nn.Parameter(torch.rand(num_points, 3))
            features_rest = torch.nn.Parameter(torch.zeros((num_points, dim_sh - 1, 3)))

        if self.points3D_scale is not None:
            opacities = torch.nn.Parameter(torch.logit(torch.ones(num_points, 1) - 1e-5))
        else:
            opacities = torch.nn.Parameter(torch.logit(0.1 * torch.ones(num_points, 1)))

        self.gauss_params = torch.nn.ParameterDict(
            {
                "means": means,
                "scales": scales,
                "quats": quats,
                "features_dc": features_dc,
                "features_rest": features_rest,
                "opacities": opacities,
            }
        )

        self.register_buffer("means_orig", self.means.data.detach().clone(), persistent=False)
        self.register_buffer("scales_orig", self.scales.data.detach().clone(), persistent=False)
        self.register_buffer("quats_orig", self.quats.data.detach().clone(), persistent=False)
        self.register_buffer("features_dc_orig", self.features_dc.data.detach().clone(), persistent=False)
        self.register_buffer("features_rest_orig", self.features_rest.data.detach().clone(), persistent=False)
        self.register_buffer("opacities_orig", self.opacities.data.detach().clone(), persistent=False)

        self.camera_optimizer: CameraOptimizer = self.config.camera_optimizer.setup(
            num_cameras=self.num_train_data, device="cpu"
        )

        # metrics
        from torchmetrics.image import PeakSignalNoiseRatio
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        self.step = 0

        self.crop_box: Optional[OrientedBox] = None
        if self.config.background_color == "random":
            self.background_color = torch.tensor(
                [0, 0, 0]

                # [0.1490, 0.1647, 0.2157]
            )  # This color is the same as the default background color in Viser. This would only affect the background color when rendering.
        else:
            self.background_color = get_color(self.config.background_color)

        # our stuff
        # self.vae = AutoencoderKL.from_pretrained("Intel/ldm3d-4c", subfolder="vae")
        # self.vae.requires_grad_(False)
        #
        # # 4 for latent + 4 for RGBD + 1 for accumulation + feature dim
        # unet_base = UNet2DConditionModel.from_pretrained("Intel/ldm3d-4c", subfolder="unet",
        #                                                  in_channels=4,
        #                                                  low_cpu_mem_usage=False,
        #                                                  ignore_mismatched_sizes=False)
        #
        # self.unet = UNet2DConditionModel.from_pretrained("Intel/ldm3d-4c", subfolder="unet",
        #                                                  in_channels=9 + self.config.cond_feature_out_dim,
        #                                                  low_cpu_mem_usage=False,
        #                                                  ignore_mismatched_sizes=True)
        # old_state = unet_base.state_dict()
        # new_state = self.unet.state_dict()
        # in_key = "conv_in.weight"
        # # Check if we need to port weights from 4ch input
        # if old_state[in_key].shape != new_state[in_key].shape:
        #     CONSOLE.log(f"Manual init: {in_key}")
        #     new_state[in_key].zero_()
        #     new_state[in_key][:, :old_state[in_key].shape[1], :, :].copy_(old_state[in_key])
        #     self.unet.load_state_dict(new_state)
        #
        # # self.unet.enable_xformers_memory_efficient_attention()
        #
        # self.noise_scheduler = DDPMScheduler.from_pretrained("Intel/ldm3d-4c", subfolder="scheduler")
        # self.ddim_scheduler = DDIMScheduler.from_pretrained("Intel/ldm3d-4c", subfolder="scheduler")

        # ema_unet = EMAModel(self.unet.parameters(), model_cls=UNet2DConditionModel,
        #                     model_config=self.unet.config)
        # diffusion_state = torch.load(self.config.diffusion_checkpoint_path, map_location="cpu")["pipeline"]
        # prefix = "ema_unet."
        # prefix_len = len(prefix)
        # ema_state_dict = {}
        # for key, val in diffusion_state.items():
        #     if key.startswith(prefix):
        #         ema_state_dict[key[prefix_len:]] = val
        #
        # ema_unet.load_state_dict(ema_state_dict)
        # ema_unet.copy_to(self.unet.parameters())
        #
        # self.register_buffer("alphas", self.noise_scheduler.alphas_cumprod)

        self.clip_embeddings = None
        self.last_camera = None
        self.infer_index = 0

    @torch.inference_mode()
    def get_clip_embeddings(self):
        if self.clip_embeddings is not None:
            return self.clip_embeddings

        image_encoder = CLIPVisionModelWithProjection.from_pretrained("lambdalabs/sd-image-variations-diffusers",
                                                                      subfolder="image_encoder").to(self.device)
        image_encoder_resize = transforms.Resize(
            (224, 224),
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=False,
        )

        image_encoder_normalize = transforms.Normalize(
            [0.48145466, 0.4578275, 0.40821073],
            [0.26862954, 0.26130258, 0.27577711])

        images = torch.stack([torch.FloatTensor(np.asarray(Image.open(x))) for x in self.train_image_filenames]).to(
            self.device) / 255
        images = image_encoder_normalize(image_encoder_resize(images.permute(0, 3, 1, 2)))
        c_crossattn = image_encoder(images).image_embeds

        delattr(self, "clip_embeddings")
        self.register_buffer("clip_embeddings", c_crossattn, persistent=False)

        return self.clip_embeddings

    @torch.no_grad()
    def infer_views(self, optimizers: Optimizers, step: int):
        # trigger = Path("/scratch/hturki/render_view")
        # if not trigger.exists():
        #     return
        return
        if self.infer_index > len(self.sds_cameras) - 1:
            # print("DONE")
            return

        self.last_camera = self.sds_cameras[self.infer_index:self.infer_index+1]
        print("INFER", self.infer_index)
        self.infer_index += 1

        if self.last_camera is None:
            return

        c_crossattn = self.get_clip_embeddings().unsqueeze(0)

        cropped_dim = min(self.last_camera.width, self.last_camera.height)
        self.last_camera.cx -= ((self.last_camera.width - cropped_dim) / 2)
        self.last_camera.cy -= ((self.last_camera.height - cropped_dim) / 2)
        self.last_camera.cx *= (256 / cropped_dim)
        self.last_camera.cy *= (256 / cropped_dim)
        self.last_camera.fx *= (256 / cropped_dim)
        self.last_camera.fy *= (256 / cropped_dim)
        self.last_camera.width.fill_(256)
        self.last_camera.height.fill_(256)

        cond_outputs = self.get_outputs(self.last_camera, save_last_camera=False)
        cond_depth = cond_outputs["depth"]
        depth_scaling_factor = cond_depth.mean() * 2
        cond_depth = (cond_depth / depth_scaling_factor).permute(2, 0, 1)
        with torch.cuda.amp.autocast(enabled=False):
            encode = self.vae.encode(
                torch.cat([cond_outputs["rgb"].permute(2, 0, 1), cond_depth, ], 0).unsqueeze(0) * 2 - 1)
            concat_list = [encode.latent_dist.mode() * self.vae.config.scaling_factor]

        accumulation = cond_outputs["accumulation"].permute(2, 0, 1)
        concat_list.append(F.interpolate(accumulation.unsqueeze(0), concat_list[0].shape[2:], mode="bicubic"))
        c_concat = torch.cat(concat_list, 1)

        self.ddim_scheduler.set_timesteps(self.config.inference_steps, device=self.device)
        n_samples = 1
        vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        inference_latents = randn_tensor((n_samples, self.vae.config.latent_channels,
                                          self.last_camera.height.item() // vae_scale_factor,
                                          self.last_camera.height.item() // vae_scale_factor),
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
            noise_pred = self.unet_1(
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

        decoded_rgb = decoded[..., :3].clamp(0, 1).squeeze(0)
        decoded_depth = decoded[..., 3:].clamp_min(0).squeeze(0) * depth_scaling_factor

        ray_bundle = self.last_camera.generate_rays(0)

        xyz = ray_bundle.origins + decoded_depth * ray_bundle.directions * ray_bundle.metadata["directions_norm"]
        scales = decoded_depth.view(-1, 1) / self.last_camera.fx

        out = {
            "means": torch.cat([self.means, xyz.view(-1, 3)]),
            "features_dc": torch.cat([self.features_dc, torch.logit(decoded_rgb.view(-1, 3))]),
            "features_rest": torch.cat([self.features_rest, torch.zeros(scales.shape[0], *self.features_rest.shape[1:],
                                                                        device=self.features_rest.device)]),
            "opacities": torch.cat([self.opacities, torch.logit(torch.full_like(xyz.view(-1, 3)[..., :1], 1 - 1e-5))]),
            "scales": torch.cat([self.scales, torch.log(scales).expand(-1, 3)]),
            "quats": torch.cat([self.quats, random_quat_tensor(scales.shape[0], device=self.quats.device)]),
        }

        # TODO this will break if the size of self.means is smaller than what we're adding
        split_idcs = torch.zeros(self.means.shape[0], dtype=torch.bool, device=scales.device)
        split_idcs[-scales.shape[0]:] = True

        for name, param in self.gauss_params.items():
            self.gauss_params[name] = torch.nn.Parameter(out[name])
        self.dup_in_all_optim(optimizers, split_idcs, 1)
        # import pdb;
        # pdb.set_trace()

        self.xys_grad_norm = None
        self.vis_counts = None
        # trigger.unlink()

    def set_min_max_steps(self, step: int):
        weight = (step - self.config.step_anneal_range[0]) / (
                self.config.step_anneal_range[1] - self.config.step_anneal_range[0])
        weight = min(max(weight, 0), 1)

        min_step_percent = (1 - weight) * self.config.min_step_percent[0] + weight * self.config.min_step_percent[1]
        max_step_percent = (1 - weight) * self.config.max_step_percent[0] + weight * self.config.max_step_percent[1]
        self.min_step = int(1000 * min_step_percent)
        self.max_step = int(1000 * max_step_percent)

    def get_sds_loss(self):
        if len(self.sds_cameras) <= self.config.random_camera_batch_size:
            random_cameras = self.sds_cameras
        else:
            random_indices = np.random.choice(len(self.sds_cameras), self.config.random_camera_batch_size,
                                              replace=False)
            random_cameras = self.sds_cameras[torch.LongTensor(random_indices)].to(self.device)

        cond_outputs = self.get_outputs(random_cameras, save_last_camera=False, use_original=True)
        cond_depth = cond_outputs["depth"]
        depth_scaling_factor = cond_depth.mean(dim=-1, keepdim=True) * 2
        cond_depth = (cond_depth / depth_scaling_factor).permute(0, 3, 1, 2)
        with torch.cuda.amp.autocast(enabled=False):
            concat_list = [self.vae.encode(torch.cat([
                cond_outputs["rgb"].permute(0, 3, 1, 2),
                cond_depth,
            ], 1) * 2 - 1).latent_dist.mode() * self.vae.config.scaling_factor]

        accumulation = cond_outputs["accumulation"].permute(0, 3, 1, 2)
        concat_list.append(F.interpolate(accumulation, concat_list[0].shape[2:], mode="bicubic"))
        c_concat = torch.cat(concat_list, 1)
        c_crossattn = self.get_clip_embeddings().unsqueeze(0).expand(len(random_cameras), -1, -1)

        sds_outputs = self.get_outputs(random_cameras, save_last_camera=False, use_original=False)
        sds_depth = (cond_outputs["depth"] / depth_scaling_factor).permute(0, 3, 1, 2)

        latents = self.vae.encode(torch.cat([
            sds_outputs["rgb"].permute(0, 3, 1, 2),
            sds_depth,
        ], 1) * 2 - 1).latent_dist.sample() * self.vae.config.scaling_factor

        timesteps = torch.randint(
            self.min_step,
            self.max_step + 1,
            [self.config.random_camera_batch_size],
            dtype=torch.long,
            device=self.device,
        )

        c_concat = torch.cat([torch.zeros_like(c_concat), c_concat])
        c_crossattn = torch.cat([torch.zeros_like(c_crossattn), c_crossattn])

        with torch.inference_mode():
            # add noise
            noise = torch.randn_like(latents)
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)  # type: ignore
            # pred noise
            noisy_latents = torch.cat([noisy_latents] * 2)
            noise_pred = self.unet(torch.cat([noisy_latents, c_concat], 1), torch.cat([timesteps] * 2),
                                   c_crossattn).sample

        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        noise_pred = noise_pred_cond + self.config.guidance_scale * (noise_pred_cond - noise_pred_uncond)

        w = (1 - self.alphas[timesteps]).view(-1, 1, 1, 1)

        grad = w * (noise_pred - noise)
        grad = torch.nan_to_num(grad)

        target = (latents - grad).detach()
        return 0.5 * F.mse_loss(latents, target, reduction="sum") / latents.shape[0]

    @property
    def colors(self):
        if self.config.sh_degree > 0:
            return SH2RGB(self.features_dc)
        else:
            return torch.sigmoid(self.features_dc)

    @property
    def shs_0(self):
        return self.features_dc

    @property
    def shs_rest(self):
        return self.features_rest

    @property
    def num_points(self):
        return self.means.shape[0]

    @property
    def means(self):
        return self.gauss_params["means"]

    @property
    def scales(self):
        return self.gauss_params["scales"]

    @property
    def quats(self):
        return self.gauss_params["quats"]

    @property
    def features_dc(self):
        return self.gauss_params["features_dc"]

    @property
    def features_rest(self):
        return self.gauss_params["features_rest"]

    @property
    def opacities(self):
        return self.gauss_params["opacities"]

    def load_state_dict(self, dict, **kwargs):  # type: ignore
        # resize the parameters to match the new number of points
        self.step = 30000
        if "means" in dict:
            # For backwards compatibility, we remap the names of parameters from
            # means->gauss_params.means since old checkpoints have that format
            for p in ["means", "scales", "quats", "features_dc", "features_rest", "opacities"]:
                dict[f"gauss_params.{p}"] = dict[p]
        newp = dict["gauss_params.means"].shape[0]
        for name, param in self.gauss_params.items():
            old_shape = param.shape
            new_shape = (newp,) + old_shape[1:]
            self.gauss_params[name] = torch.nn.Parameter(torch.zeros(new_shape, device=self.device))
        super().load_state_dict(dict, **kwargs)

    def k_nearest_sklearn(self, x: torch.Tensor, k: int):
        """
            Find k-nearest neighbors using sklearn's NearestNeighbors.
        x: The data tensor of shape [num_samples, num_features]
        k: The number of neighbors to retrieve
        """
        # Convert tensor to numpy array
        x_np = x.cpu().numpy()

        # Build the nearest neighbors model
        from sklearn.neighbors import NearestNeighbors

        nn_model = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", metric="euclidean").fit(x_np)

        # Find the k-nearest neighbors
        distances, indices = nn_model.kneighbors(x_np)

        # Exclude the point itself from the result and return
        return distances[:, 1:].astype(np.float32), indices[:, 1:].astype(np.float32)

    def remove_from_optim(self, optimizer, deleted_mask, new_params):
        """removes the deleted_mask from the optimizer provided"""
        assert len(new_params) == 1
        # assert isinstance(optimizer, torch.optim.Adam), "Only works with Adam"

        param = optimizer.param_groups[0]["params"][0]
        param_state = optimizer.state[param]
        del optimizer.state[param]

        # Modify the state directly without deleting and reassigning.
        if "exp_avg" in param_state:
            param_state["exp_avg"] = param_state["exp_avg"][~deleted_mask]
            param_state["exp_avg_sq"] = param_state["exp_avg_sq"][~deleted_mask]

        # Update the parameter in the optimizer's param group.
        del optimizer.param_groups[0]["params"][0]
        del optimizer.param_groups[0]["params"]
        optimizer.param_groups[0]["params"] = new_params
        optimizer.state[new_params[0]] = param_state

    def remove_from_all_optim(self, optimizers, deleted_mask):
        param_groups = self.get_gaussian_param_groups()
        for group, param in param_groups.items():
            self.remove_from_optim(optimizers.optimizers[group], deleted_mask, param)
        torch.cuda.empty_cache()

    def dup_in_optim(self, optimizer, dup_mask, new_params, n=2):
        """adds the parameters to the optimizer"""
        param = optimizer.param_groups[0]["params"][0]
        param_state = optimizer.state[param]
        if "exp_avg" in param_state:
            repeat_dims = (n,) + tuple(1 for _ in range(param_state["exp_avg"].dim() - 1))
            param_state["exp_avg"] = torch.cat(
                [
                    param_state["exp_avg"],
                    torch.zeros_like(param_state["exp_avg"][dup_mask.squeeze()]).repeat(*repeat_dims),
                ],
                dim=0,
            )
            param_state["exp_avg_sq"] = torch.cat(
                [
                    param_state["exp_avg_sq"],
                    torch.zeros_like(param_state["exp_avg_sq"][dup_mask.squeeze()]).repeat(*repeat_dims),
                ],
                dim=0,
            )
        del optimizer.state[param]
        optimizer.state[new_params[0]] = param_state
        optimizer.param_groups[0]["params"] = new_params
        del param

    def dup_in_all_optim(self, optimizers, dup_mask, n):
        param_groups = self.get_gaussian_param_groups()
        for group, param in param_groups.items():
            self.dup_in_optim(optimizers.optimizers[group], dup_mask, param, n)

    def after_train(self, step: int):
        assert step == self.step
        # to save some training time, we no longer need to update those stats post refinement
        if self.step >= self.config.stop_split_at:
            return
        with torch.no_grad():
            # keep track of a moving average of grad norms
            visible_mask = (self.radii > 0).flatten()
            grads = self.xys.absgrad[0][visible_mask].norm(dim=-1)  # type: ignore
            # print(f"grad norm min {grads.min().item()} max {grads.max().item()} mean {grads.mean().item()} size {grads.shape}")
            if self.xys_grad_norm is None:
                self.xys_grad_norm = torch.zeros(self.num_points, device=self.device, dtype=torch.float32)
                self.vis_counts = torch.ones(self.num_points, device=self.device, dtype=torch.float32)
            assert self.vis_counts is not None
            self.vis_counts[visible_mask] += 1
            self.xys_grad_norm[visible_mask] += grads

    def set_crop(self, crop_box: Optional[OrientedBox]):
        self.crop_box = crop_box

    def set_background(self, background_color: torch.Tensor):
        assert background_color.shape == (3,)
        self.background_color = background_color

    def refinement_after(self, optimizers: Optimizers, step):
        assert step == self.step
        if self.step <= self.config.warmup_length:
            return
        with torch.no_grad():
            # Offset all the opacity reset logic by refine_every so that we don't
            # save checkpoints right when the opacity is reset (saves every 2k)
            # then cull
            # only split/cull if we've seen every image since opacity reset
            reset_interval = self.config.reset_alpha_every * self.config.refine_every
            do_densification = (
                    self.step < self.config.stop_split_at
                    and self.step % reset_interval > self.num_train_data + self.config.refine_every
            )
            if do_densification:
                # then we densify
                assert self.xys_grad_norm is not None and self.vis_counts is not None
                avg_grad_norm = (self.xys_grad_norm / self.vis_counts) * 0.5 * max(self.last_size[0],
                                                                                   self.last_size[1])
                high_grads = (avg_grad_norm > self.config.densify_grad_thresh).squeeze()
                splits = (self.scales.exp().max(dim=-1).values > self.config.densify_size_thresh).squeeze()
                splits &= high_grads
                nsamps = self.config.n_split_samples
                split_params = self.split_gaussians(splits, nsamps)

                dups = (self.scales.exp().max(dim=-1).values <= self.config.densify_size_thresh).squeeze()
                dups &= high_grads
                dup_params = self.dup_gaussians(dups)
                for name, param in self.gauss_params.items():
                    self.gauss_params[name] = torch.nn.Parameter(
                        torch.cat([param.detach(), split_params[name], dup_params[name]], dim=0)
                    )

                split_idcs = torch.where(splits)[0]
                self.dup_in_all_optim(optimizers, split_idcs, nsamps)

                dup_idcs = torch.where(dups)[0]
                self.dup_in_all_optim(optimizers, dup_idcs, 1)

                # After a guassian is split into two new gaussians, the original one should also be pruned.
                splits_mask = torch.cat(
                    (
                        splits,
                        torch.zeros(
                            nsamps * splits.sum() + dups.sum(),
                            device=self.device,
                            dtype=torch.bool,
                        ),
                    )
                )

                deleted_mask = self.cull_gaussians(splits_mask)
            elif self.step >= self.config.stop_split_at and self.config.continue_cull_post_densification:
                deleted_mask = self.cull_gaussians()
            else:
                # if we donot allow culling post refinement, no more gaussians will be pruned.
                deleted_mask = None

            if deleted_mask is not None:
                self.remove_from_all_optim(optimizers, deleted_mask)

            if self.step < self.config.stop_split_at and self.step % reset_interval == self.config.refine_every:
                # Reset value is set to be twice of the cull_alpha_thresh
                reset_value = self.config.cull_alpha_thresh * 2.0
                self.opacities.data = torch.clamp(
                    self.opacities.data,
                    max=torch.logit(torch.tensor(reset_value, device=self.device)).item(),
                )
                # reset the exp of optimizer
                optim = optimizers.optimizers["opacities"]
                param = optim.param_groups[0]["params"][0]
                param_state = optim.state[param]
                param_state["exp_avg"] = torch.zeros_like(param_state["exp_avg"])
                param_state["exp_avg_sq"] = torch.zeros_like(param_state["exp_avg_sq"])

            self.xys_grad_norm = None
            self.vis_counts = None

    def cull_gaussians(self, extra_cull_mask: Optional[torch.Tensor] = None):
        """
        This function deletes gaussians with under a certain opacity threshold
        extra_cull_mask: a mask indicates extra gaussians to cull besides existing culling criterion
        """
        n_bef = self.num_points
        # cull transparent ones
        culls = (torch.sigmoid(self.opacities) < self.config.cull_alpha_thresh).squeeze()
        below_alpha_count = torch.sum(culls).item()
        toobigs_count = 0
        if extra_cull_mask is not None:
            culls = culls | extra_cull_mask
        if self.step > self.config.refine_every * self.config.reset_alpha_every:
            # cull huge ones
            toobigs = (torch.exp(self.scales).max(dim=-1).values > self.config.cull_scale_thresh).squeeze()
            culls = culls | toobigs
            toobigs_count = torch.sum(toobigs).item()
        for name, param in self.gauss_params.items():
            self.gauss_params[name] = torch.nn.Parameter(param[~culls])

        CONSOLE.log(
            f"Culled {n_bef - self.num_points} gaussians "
            f"({below_alpha_count} below alpha thresh, {toobigs_count} too bigs, {self.num_points} remaining)"
        )

        return culls

    def split_gaussians(self, split_mask, samps):
        """
        This function splits gaussians that are too large
        """
        n_splits = split_mask.sum().item()
        CONSOLE.log(
            f"Splitting {split_mask.sum().item() / self.num_points} gaussians: {n_splits}/{self.num_points}")
        centered_samples = torch.randn((samps * n_splits, 3), device=self.device)  # Nx3 of axis-aligned scales
        scaled_samples = (
                torch.exp(self.scales[split_mask].repeat(samps, 1)) * centered_samples
        )  # how these scales are rotated
        quats = self.quats[split_mask] / self.quats[split_mask].norm(dim=-1, keepdim=True)  # normalize them first
        rots = quat_to_rotmat(quats.repeat(samps, 1))  # how these scales are rotated
        rotated_samples = torch.bmm(rots, scaled_samples[..., None]).squeeze()
        new_means = rotated_samples + self.means[split_mask].repeat(samps, 1)
        # step 2, sample new colors
        new_features_dc = self.features_dc[split_mask].repeat(samps, 1)
        new_features_rest = self.features_rest[split_mask].repeat(samps, 1, 1)
        # step 3, sample new opacities
        new_opacities = self.opacities[split_mask].repeat(samps, 1)
        # step 4, sample new scales
        size_fac = 1.6
        new_scales = torch.log(torch.exp(self.scales[split_mask]) / size_fac).repeat(samps, 1)
        self.scales[split_mask] = torch.log(torch.exp(self.scales[split_mask]) / size_fac)
        # step 5, sample new quats
        new_quats = self.quats[split_mask].repeat(samps, 1)
        out = {
            "means": new_means,
            "features_dc": new_features_dc,
            "features_rest": new_features_rest,
            "opacities": new_opacities,
            "scales": new_scales,
            "quats": new_quats,
        }
        for name, param in self.gauss_params.items():
            if name not in out:
                out[name] = param[split_mask].repeat(samps, 1)
        return out

    def dup_gaussians(self, dup_mask):
        """
        This function duplicates gaussians that are too small
        """
        n_dups = dup_mask.sum().item()
        CONSOLE.log(f"Duplicating {dup_mask.sum().item() / self.num_points} gaussians: {n_dups}/{self.num_points}")
        new_dups = {}
        for name, param in self.gauss_params.items():
            new_dups[name] = param[dup_mask]
        return new_dups

    def get_training_callbacks(
            self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        cbs = []
        cbs.append(TrainingCallback([TrainingCallbackLocation.BEFORE_TRAIN_ITERATION], self.step_cb))
        # The order of these matters
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                self.after_train,
            )
        )
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                self.refinement_after,
                update_every_num_iters=self.config.refine_every,
                args=[training_callback_attributes.optimizers],
            )
        )
        cbs.append(TrainingCallback([TrainingCallbackLocation.AFTER_TRAIN_ITERATION], self.infer_views,
                                    args=[training_callback_attributes.optimizers]))
        cbs.append(TrainingCallback(
            [TrainingCallbackLocation.BEFORE_TRAIN_ITERATION], self.set_min_max_steps,
        ))

        return cbs

    def step_cb(self, step):
        self.step = step

    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        # Here we explicitly use the means, scales as parameters so that the user can override this function and
        # specify more if they want to add more optimizable params to gaussians.
        return {
            name: [self.gauss_params[name]]
            for name in ["means", "scales", "quats", "features_dc", "features_rest", "opacities"]
        }

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Obtain the parameter groups for the optimizers

        Returns:
            Mapping of different parameter groups
        """
        gps = self.get_gaussian_param_groups()
        self.camera_optimizer.get_param_groups(param_groups=gps)
        return gps

    def _get_downscale_factor(self):
        if self.training:
            return 2 ** max(
                (self.config.num_downscales - self.step // self.config.resolution_schedule),
                0,
            )
        else:
            return 1

    def _downscale_if_required(self, image):
        d = self._get_downscale_factor()
        if d > 1:
            return resize_image(image, d)
        return image

    @staticmethod
    def get_empty_outputs(width: int, height: int, background: torch.Tensor) -> Dict[
        str, Union[torch.Tensor, List]]:
        rgb = background.repeat(height, width, 1)
        depth = background.new_ones(*rgb.shape[:2], 1) * 10
        accumulation = background.new_zeros(*rgb.shape[:2], 1)
        return {"rgb": rgb, "depth": depth, "accumulation": accumulation, "background": background}

    def get_outputs(self, camera: Cameras, save_last_camera: bool = True, use_original: bool = False) -> Dict[
        str, Union[torch.Tensor, List]]:
        """Takes in a Ray Bundle and returns a dictionary of outputs.

        Args:
            ray_bundle: Input bundle of rays. This raybundle should have all the
            needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """
        if not isinstance(camera, Cameras):
            print("Called get_outputs with not a camera")
            return {}

        if save_last_camera and not self.training:
            self.last_camera = camera

        # get the background color
        if not save_last_camera:
            background = torch.zeros(3, device=self.device)
            optimized_camera_to_world = camera.camera_to_worlds
        elif self.training and (not save_last_camera):
            assert camera.shape[0] == 1, "Only one camera at a time"
            optimized_camera_to_world = self.camera_optimizer.apply_to_camera(camera)

            if self.config.background_color == "random":
                background = torch.rand(3, device=self.device)
            elif self.config.background_color == "white":
                background = torch.ones(3, device=self.device)
            elif self.config.background_color == "black":
                background = torch.zeros(3, device=self.device)
            else:
                background = self.background_color.to(self.device)
        else:
            optimized_camera_to_world = camera.camera_to_worlds

            if renderers.BACKGROUND_COLOR_OVERRIDE is not None:
                background = renderers.BACKGROUND_COLOR_OVERRIDE.to(self.device)
            else:
                background = self.background_color.to(self.device)

        if self.crop_box is not None and not self.training:
            crop_ids = self.crop_box.within(self.means).squeeze()
            if crop_ids.sum() == 0:
                return self.get_empty_outputs(int(camera.width.item()), int(camera.height.item()), background)
        else:
            crop_ids = None
        camera_scale_fac = (1.0 / self._get_downscale_factor()) if save_last_camera else 1
        viewmat = get_viewmat(optimized_camera_to_world)
        W, H = int(camera.width[0] * camera_scale_fac), int(camera.height[0] * camera_scale_fac)
        self.last_size = (H, W)

        if use_original:
            if crop_ids is not None:
                opacities_crop = self.opacities_orig[crop_ids]
                means_crop = self.means_orig[crop_ids]
                features_dc_crop = self.features_dc_orig[crop_ids]
                features_rest_crop = self.features_rest_orig[crop_ids]
                scales_crop = self.scales_orig[crop_ids]
                quats_crop = self.quats_orig[crop_ids]
            else:
                opacities_crop = self.opacities_orig
                means_crop = self.means_orig
                features_dc_crop = self.features_dc_orig
                features_rest_crop = self.features_rest_orig
                scales_crop = self.scales_orig
                quats_crop = self.quats_orig
        else:
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
        BLOCK_WIDTH = 16  # this controls the tile size of rasterization, 16 is a good default
        K = camera.get_intrinsics_matrices().cuda()
        K[:, :2, :] *= camera_scale_fac
        # apply the compensation of screen space blurring to gaussians
        if self.config.rasterize_mode not in ["antialiased", "classic"]:
            raise ValueError("Unknown rasterize_mode: %s", self.config.rasterize_mode)

        if self.config.output_depth_during_training or (not self.training) or (not save_last_camera):
            render_mode = "RGB+ED"
        else:
            render_mode = "RGB"

        if self.config.sh_degree > 0:
            sh_degree_to_use = min(self.step // self.config.sh_degree_interval, self.config.sh_degree)
        else:
            sh_degree_to_use = None
            colors_crop = torch.sigmoid(colors_crop.squeeze(1))

        render, alpha, info = rasterization(
            means=means_crop,
            quats=quats_crop / quats_crop.norm(dim=-1, keepdim=True),
            scales=torch.exp(scales_crop),
            opacities=torch.sigmoid(opacities_crop).squeeze(-1),
            colors=colors_crop,
            viewmats=viewmat,
            Ks=K,
            width=W,
            height=H,
            tile_size=BLOCK_WIDTH,
            packed=False,
            near_plane=0.01,
            far_plane=1e10,
            render_mode=render_mode,
            sh_degree=sh_degree_to_use,
            sparse_grad=False,
            absgrad=True,
            rasterize_mode=self.config.rasterize_mode,
            radius_clip=0 if self.training else 3,
        )
        if self.training and info["means2d"].requires_grad:
            info["means2d"].retain_grad()
        self.xys = info["means2d"]  # [1, N, 2]
        self.radii = info["radii"][0]  # [N]

        alpha = alpha[:, ...]
        rgb = render[:, ..., :3] + (1 - alpha) * background
        rgb = torch.clamp(rgb, 0.0, 1.0)
        if render_mode == "RGB+ED":
            depth_im = render[:, ..., 3:4]
            depth_im = torch.where(alpha > 0, depth_im, depth_im.detach().max()).squeeze(0)
        else:
            depth_im = None
        return {"rgb": rgb.squeeze(0), "depth": depth_im, "accumulation": alpha.squeeze(0),
                "background": background}  # type: ignore

    def get_gt_img(self, image: torch.Tensor):
        """Compute groundtruth image with iteration dependent downscale factor for evaluation purpose

        Args:
            image: tensor.Tensor in type uint8 or float32
        """
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        gt_img = self._downscale_if_required(image)
        return gt_img.to(self.device)

    def composite_with_background(self, image, background) -> torch.Tensor:
        """Composite the ground truth image with a background color when it has an alpha channel.

        Args:
            image: the image to composite
            background: the background color
        """
        if image.shape[2] == 4:
            alpha = image[..., -1].unsqueeze(-1).repeat((1, 1, 3))
            return alpha * image[..., :3] + (1 - alpha) * background
        else:
            return image

    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        """Compute and returns metrics.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
        """
        gt_rgb = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        metrics_dict = {}
        predicted_rgb = outputs["rgb"]
        metrics_dict["psnr"] = self.psnr(predicted_rgb, gt_rgb)

        metrics_dict["gaussian_count"] = self.num_points

        self.camera_optimizer.get_metrics_dict(metrics_dict)
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        """Computes and returns the losses dict.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
            metrics_dict: dictionary of metrics, some of which we can use for loss
        """
        gt_img = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        pred_img = outputs["rgb"]

        # Set masked part of both ground-truth and rendered image to black.
        # This is a little bit sketchy for the SSIM loss.
        if "mask" in batch:
            # batch["mask"] : [H, W, 1]
            mask = self._downscale_if_required(batch["mask"])
            mask = mask.to(self.device)
            assert mask.shape[:2] == gt_img.shape[:2] == pred_img.shape[:2]
            gt_img = gt_img * mask
            pred_img = pred_img * mask

        Ll1 = torch.abs(gt_img - pred_img).mean()
        simloss = 1 - self.ssim(gt_img.permute(2, 0, 1)[None, ...], pred_img.permute(2, 0, 1)[None, ...])
        if self.config.use_scale_regularization and self.step % 10 == 0:
            scale_exp = torch.exp(self.scales)
            scale_reg = (
                    torch.maximum(
                        scale_exp.amax(dim=-1) / scale_exp.amin(dim=-1),
                        torch.tensor(self.config.max_gauss_ratio),
                    )
                    - self.config.max_gauss_ratio
            )
            scale_reg = 0.1 * scale_reg.mean()
        else:
            scale_reg = torch.tensor(0.0).to(self.device)

        loss_dict = {
            "main_loss": (1 - self.config.ssim_lambda) * Ll1 + self.config.ssim_lambda * simloss * 0,
            "scale_reg": scale_reg * 0,
        }

        # if self.training:
        #     # Add loss from camera optimizer
        #     self.camera_optimizer.get_loss_dict(loss_dict)
        #
        #     loss_dict["sds"] = self.config.sds_loss_mult * self.get_sds_loss()

        return loss_dict

    @torch.no_grad()
    def get_outputs_for_camera(self, camera: Cameras, obb_box: Optional[OrientedBox] = None) -> Dict[
        str, torch.Tensor]:
        """Takes in a camera, generates the raybundle, and computes the output of the model.
        Overridden for a camera-based gaussian model.

        Args:
            camera: generates raybundle
        """
        assert camera is not None, "must provide camera to gaussian model"
        self.set_crop(obb_box)
        outs = self.get_outputs(camera.to(self.device))
        return outs  # type: ignore

    def get_image_metrics_and_images(
            self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """Writes the test image outputs.

        Args:
            image_idx: Index of the image.
            step: Current step.
            batch: Batch of data.
            outputs: Outputs of the model.

        Returns:
            A dictionary of metrics.
        """
        gt_rgb = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        predicted_rgb = outputs["rgb"]

        combined_rgb = torch.cat([gt_rgb, predicted_rgb], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        gt_rgb = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
        predicted_rgb = torch.moveaxis(predicted_rgb, -1, 0)[None, ...]

        psnr = self.psnr(gt_rgb, predicted_rgb)
        ssim = self.ssim(gt_rgb, predicted_rgb)
        lpips = self.lpips(gt_rgb, predicted_rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        images_dict = {"img": combined_rgb}

        return metrics_dict, images_dict
