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
Gaussian Splatting implementation that combines many recent advancements.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Type, Union, Any

import torch
from diffusers import UNet2DConditionModel, EMAModel, DDPMScheduler, DDIMScheduler
from diffusers.utils.torch_utils import randn_tensor
from gsplat.cuda_legacy._torch_impl import quat_to_rotmat
from mast3r.model import AsymmetricMASt3R
from nerfstudio.models.splatfacto import random_quat_tensor, RGB2SH, SH2RGB, resize_image, get_viewmat
from torch_scatter import scatter_min
from torchvision import transforms
from transformers import CLIPVisionModelWithProjection

from generfstudio.fields.batched_pc_optimizer import fast_depthmap_to_pts3d
from generfstudio.generfstudio_constants import RGB, DEPTH, ACCUMULATION
from generfstudio.models.rgbd_diffusion_base import RGBDDiffusionBase

try:
    from gsplat.rendering import rasterization
except ImportError:
    print("Please install gsplat>=1.0.0")
from gsplat.cuda_legacy._wrapper import num_sh_bases
from pytorch_msssim import SSIM
from torch.nn import Parameter

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.engine.optimizers import Optimizers
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils.colors import get_color
from nerfstudio.utils.rich_utils import CONSOLE
from PIL import Image
from generfstudio.generfstudio_utils import central_crop_v2
import torch.nn.functional as F
import numpy as np
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.utils.image import load_images


@dataclass
class DepthGaussiansModelConfig(ModelConfig):
    _target: Type = field(default_factory=lambda: DepthGaussiansModel)
    warmup_length: int = 500
    """period of steps where refinement is turned off"""
    refine_every: int = 1000
    """period of steps where gaussians are culled and densified"""
    resolution_schedule: int = 3000
    """training starts at 1/d resolution, every n steps this is doubled"""
    background_color: Literal["random", "black", "white", "none"] = "none"
    """Whether to randomize the background color."""
    num_downscales: int = 0
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
    cull_screen_size: float = 0.15
    """if a gaussian is more than this percent of screen space, cull it"""
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
    output_depth_during_training: bool = True
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

    lpips_lambda: float = 0.25
    """weight of ssim loss"""

    checkpoint_path_1: Path = Path("/compute/trinity-2-29/hturki/stage-1.ckpt")
    checkpoint_path_2: Path = Path("/compute/trinity-2-29/hturki/stage-2.ckpt")

    unet_1_pretrained_path: str = "DeepFloyd/IF-I-L-v1.0"
    # unet_1_pretrained_path: str = "Intel/ldm3d-4c"  # "DeepFloyd/IF-I-L-v1.0"
    unet_2_pretrained_path: str = "DeepFloyd/IF-II-M-v1.0"
    image_encoder_pretrained_path: str = "lambdalabs/sd-image-variations-diffusers"

    # beta_schedule: str = "scaled_linear"  # "squaredcos_cap_v2"
    beta_schedule: str = "squaredcos_cap_v2"
    prediction_type: str = "epsilon"

    guidance_scale: float = 1.0
    use_ddim: bool = True
    inference_steps: int = 50
    render_batch_size: int = 1

    rgb_mapping: Literal["clamp", "sigmoid"] = "clamp"
    depth_mapping: Literal["scaled", "disparity", "log"] = "scaled"
    # rgbd_concat_strategy: Literal["mode", "sample", "mean_std", "downsample"] = "sample"

    depth_model_name: str = "/data/hturki/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"

    optimize: bool = True
    run_autoregression: bool = True
    autoregression_interval: int = 1000
    render_with_diffusion: bool = True
    run_dust3r_global_ba: bool = False
    blend_with_concat: bool = True
    mask_with_existing_depth: bool = True


class DepthGaussiansModel(Model):
    """Nerfstudio's implementation of Gaussian Splatting

    Args:
        config: DepthGaussians configuration to instantiate model
    """

    config: DepthGaussiansModelConfig

    def __init__(
            self,
            *args,
            seed_points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            metadata: Dict[str, Any],
            **kwargs,
    ):
        self.seed_points = seed_points
        self.points3D_scale = metadata["points3D_scale"]

        self.image_filenames = [x for x in metadata["train_image_filenames"]]
        self.ellipse_cameras = metadata["ellipse_cameras"].to("cuda")
        padding_w = (64 - self.ellipse_cameras.width % 64) % 64
        self.ellipse_cameras.width += padding_w

        padding_h = (64 - self.ellipse_cameras.height % 64) % 64
        self.ellipse_cameras.height += padding_h

        self.train_cameras = metadata["train_cameras"].to("cuda")
        self.train_depths = metadata["train_depths"]
        self.add_image_fn = metadata["add_image_fn"]
        self.min_conf_thresh = metadata["min_conf_thresh"]
        self.remaining_indices = set(range(len(self.ellipse_cameras)))

        self.depth_outputs = metadata["depth_outputs"]
        self.depth_focals = self.train_cameras.fx
        self.depth_pp = torch.cat([self.train_cameras.cx, self.train_cameras.cy], -1)

        self.depth_c2ws_opencv, self.train_w2cs_opencv = self.c2ws_opengl_to_opencv(self.train_cameras.camera_to_worlds)

        ray_bundle = self.train_cameras.generate_rays(
            camera_indices=torch.arange(self.train_cameras.shape[0]).view(-1, 1))
        pixel_area = ray_bundle.pixel_area.squeeze(-1).permute(2, 0, 1).view(self.train_cameras.shape[0], -1)
        directions_norm = ray_bundle.metadata["directions_norm"].squeeze(-1).permute(2, 0, 1).view(
            self.train_cameras.shape[0], -1)
        self.depth_scale_mult = (pixel_area * directions_norm).reshape(-1, 1)

        self.prev_depths = None
        self.prev_indices = None
        self.prev_w2cs = None

        super().__init__(*args, **kwargs)
        self.register_buffer("pred_rgbs", metadata["train_rgbs"], persistent=False)

    def populate_modules(self):
        if self.seed_points is not None and not self.config.random_init:
            means = torch.nn.Parameter(self.seed_points[0])  # (Location, Color)
        else:
            means = torch.nn.Parameter((torch.rand((self.config.num_random, 3)) - 0.5) * self.config.random_scale)
        self.xys_grad_norm = None
        num_points = means.shape[0]

        scales = torch.nn.Parameter(torch.log(self.points3D_scale.repeat(1, 3)))
        delattr(self, "points3D_scale")
        opacities = torch.nn.Parameter(torch.logit(1 - 1e-5 * torch.ones(num_points, 1)))

        self.max_2Dsize = None
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

        delattr(self, "seed_points")

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
                [0.1490, 0.1647, 0.2157]
            )  # This color is the same as the default background color in Viser. This would only affect the background color when rendering.
        elif self.config.background_color != "none":
            self.background_color = get_color(self.config.background_color)

        # our stuff
        # self.vae = AutoencoderKL.from_pretrained(self.config.unet_1_pretrained_path, subfolder="vae")
        # self.vae.requires_grad_(False)
        # self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        self.unet_1 = UNet2DConditionModel.from_pretrained(self.config.unet_1_pretrained_path, subfolder="unet",
                                                           in_channels=4 + 5,
                                                           out_channels=4,
                                                           low_cpu_mem_usage=False,
                                                           ignore_mismatched_sizes=True)
        self.unet_1.enable_xformers_memory_efficient_attention()
        self.unet_1.requires_grad_(False)

        ema_unet = EMAModel(self.unet_1.parameters(), model_cls=UNet2DConditionModel, model_config=self.unet_1.config)
        diffusion_state = torch.load(self.config.checkpoint_path_1, map_location="cpu")["pipeline"]

        ema_state_dict = {}
        ema_prefix = "ema_unet."
        ema_prefix_len = len(ema_prefix)

        self.image_encoder_state_dict_1 = {}
        image_encoder_prefixes = {"_model.image_encoder.", "_model.module.image_encoder."}

        for key, val in diffusion_state.items():
            if key.startswith(ema_prefix):
                ema_state_dict[key[ema_prefix_len:]] = val
            for image_encoder_prefix in image_encoder_prefixes:
                if key.startswith(image_encoder_prefix):
                    self.image_encoder_state_dict_1[key[len(image_encoder_prefix):]] = val

        ema_unet.load_state_dict(ema_state_dict)
        ema_unet.copy_to(self.unet_1.parameters())

        self.unet_2 = UNet2DConditionModel.from_pretrained(self.config.unet_2_pretrained_path, subfolder="unet",
                                                           in_channels=4 + 4 + 5,
                                                           out_channels=4,
                                                           low_cpu_mem_usage=False,
                                                           ignore_mismatched_sizes=True)
        self.unet_2.enable_xformers_memory_efficient_attention()
        self.unet_2.requires_grad_(False)

        ema_unet = EMAModel(self.unet_2.parameters(), model_cls=UNet2DConditionModel, model_config=self.unet_2.config)
        diffusion_state = torch.load(self.config.checkpoint_path_2, map_location="cpu")["pipeline"]

        ema_state_dict = {}
        self.image_encoder_state_dict_2 = {}

        for key, val in diffusion_state.items():
            if key.startswith(ema_prefix):
                ema_state_dict[key[ema_prefix_len:]] = val
            for image_encoder_prefix in image_encoder_prefixes:
                if key.startswith(image_encoder_prefix):
                    self.image_encoder_state_dict_2[key[len(image_encoder_prefix):]] = val

        ema_unet.load_state_dict(ema_state_dict)
        ema_unet.copy_to(self.unet_2.parameters())

        if self.config.use_ddim:
            self.scheduler = DDIMScheduler.from_pretrained(self.config.unet_1_pretrained_path,
                                                           subfolder="scheduler",
                                                           beta_schedule=self.config.beta_schedule,
                                                           prediction_type=self.config.prediction_type)
            self.scheduler.config["variance_type"] = "fixed_small"

            if self.config.blend_with_concat:
                self.scheduler.config["prediction_type"] = "sample"

            self.scheduler = DDIMScheduler.from_config(self.scheduler.config)
        else:
            self.scheduler = DDPMScheduler.from_pretrained(self.config.unet_1_pretrained_path,
                                                           subfolder="scheduler",
                                                           beta_schedule=self.config.beta_schedule,
                                                           prediction_type=self.config.prediction_type)
            self.scheduler.config["variance_type"] = "fixed_small"

            if self.config.blend_with_concat:
                self.scheduler.config["prediction_type"] = "sample"

            self.scheduler = DDPMScheduler.from_config(self.scheduler.config)

        self.scheduler.set_timesteps(self.config.inference_steps)

        self.depth_model = AsymmetricMASt3R.from_pretrained(self.config.depth_model_name)
        self.depth_model.requires_grad_(False)

        self.clip_embeddings_1 = None
        self.has_rendered = False

    @torch.inference_mode()
    def get_clip_embeddings(self):
        if self.clip_embeddings_1 is not None:
            return self.clip_embeddings_1, self.clip_embeddings_2

        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            self.config.image_encoder_pretrained_path, subfolder="image_encoder", projection_dim=4096,
            ignore_mismatched_sizes=True)
        image_encoder.load_state_dict(self.image_encoder_state_dict_1)
        delattr(self, "image_encoder_state_dict_1")
        image_encoder.requires_grad_(False)

        image_encoder_resize = transforms.Resize(
            (224, 224),
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=False,
        )

        image_encoder_normalize = transforms.Normalize(
            [0.48145466, 0.4578275, 0.40821073],
            [0.26862954, 0.26130258, 0.27577711])

        images = torch.stack([torch.FloatTensor(np.asarray(central_crop_v2(Image.open(x))))
                              for x in self.image_filenames]).to(self.device) / 255
        images = image_encoder_normalize(image_encoder_resize(images.permute(0, 3, 1, 2)))
        c_crossattn = image_encoder.to(self.device)(images).image_embeds

        delattr(self, "clip_embeddings_1")
        self.register_buffer("clip_embeddings_1", c_crossattn.unsqueeze(0), persistent=False)

        image_encoder.load_state_dict(self.image_encoder_state_dict_2)
        delattr(self, "image_encoder_state_dict_2")
        c_crossattn = image_encoder.to(self.device)(images).image_embeds
        self.register_buffer("clip_embeddings_2", c_crossattn.unsqueeze(0), persistent=False)

        return self.clip_embeddings_1, self.clip_embeddings_2

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
            # update the max screen size, as a ratio of number of pixels
            if self.max_2Dsize is None:
                self.max_2Dsize = torch.zeros_like(self.radii, dtype=torch.float32)
            newradii = self.radii.detach()[visible_mask]
            self.max_2Dsize[visible_mask] = torch.maximum(
                self.max_2Dsize[visible_mask],
                newradii / float(max(self.last_size[0], self.last_size[1])),
            )

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
                assert self.xys_grad_norm is not None and self.vis_counts is not None and self.max_2Dsize is not None
                avg_grad_norm = (self.xys_grad_norm / self.vis_counts) * 0.5 * max(self.last_size[0], self.last_size[1])
                high_grads = (avg_grad_norm > self.config.densify_grad_thresh).squeeze()
                splits = (self.scales.exp().max(dim=-1).values > self.config.densify_size_thresh).squeeze()
                if self.step < self.config.stop_screen_size_at:
                    splits |= (self.max_2Dsize > self.config.split_screen_size).squeeze()
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
                # append zeros to the max_2Dsize tensor
                self.max_2Dsize = torch.cat(
                    [
                        self.max_2Dsize,
                        torch.zeros_like(split_params["scales"][:, 0]),
                        torch.zeros_like(dup_params["scales"][:, 0]),
                    ],
                    dim=0,
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
            self.max_2Dsize = None

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
            if self.step < self.config.stop_screen_size_at:
                # cull big screen space
                if self.max_2Dsize is not None:
                    toobigs = toobigs | (self.max_2Dsize > self.config.cull_screen_size).squeeze()
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
        CONSOLE.log(f"Splitting {split_mask.sum().item() / self.num_points} gaussians: {n_splits}/{self.num_points}")
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

        if self.config.run_autoregression:
            cbs.append(TrainingCallback([TrainingCallbackLocation.AFTER_TRAIN_ITERATION], self.generate_view,
                                        update_every_num_iters=self.config.autoregression_interval,
                                        args=[training_callback_attributes.optimizers]))

        return cbs

    @torch.no_grad()
    def generate_view(self, optimizers: Optimizers, step: int):
        if step == 0 and self.config.optimize:
            return

        # trigger = Path("/scratch/hturki/render_view")
        # if not trigger.exists():
        #     return

        if len(self.remaining_indices) == 0:
            if not self.has_rendered:
                result_dir = Path("/scratch/hturki/result")
                result_dir.mkdir(parents=True, exist_ok=True)
                for idx in range(len(self.ellipse_cameras)):
                    rendering = self.get_outputs(self.ellipse_cameras[idx:idx + 1])
                    Image.fromarray((rendering[RGB] * 255).byte().cpu().numpy()).save(result_dir / f"{idx}.png")
                CONSOLE.log("DONE")
                self.has_rendered = True
            return

        remaining_renders = []
        for remaining_index in self.remaining_indices:
            remaining_renders.append(
                (remaining_index, self.get_outputs(self.ellipse_cameras[remaining_index:remaining_index + 1])))

        if len(remaining_renders) == len(self.ellipse_cameras):
            baseline_dir = Path("/scratch/hturki/baseline")
            baseline_dir.mkdir(parents=True, exist_ok=True)
            for remaining_index, rendering in remaining_renders:
                Image.fromarray((rendering[RGB] * 255).byte().cpu().numpy()).save(
                    baseline_dir / f"{remaining_index}.png")

        sorted_indices = sorted(remaining_renders, key=lambda x: x[1][ACCUMULATION].mean(), reverse=True)
        to_render_indices = []
        renderings = defaultdict(list)
        for sorted_index, sorted_rendering in sorted_indices:
            self.remaining_indices.remove(sorted_index)
            # acc_mean = sorted_rendering[ACCUMULATION].mean()
            # if acc_mean <= 0.75:
            to_render_indices.append(sorted_index)
            for key in [RGB, DEPTH, ACCUMULATION]:
                renderings[key].append(sorted_rendering[key])
            if len(to_render_indices) == self.config.render_batch_size:
                break

        CONSOLE.log(f"Rendering {to_render_indices}")
        to_render_indices = torch.LongTensor(to_render_indices).to(self.ellipse_cameras.device)
        renderings = {k: torch.stack(v) for k, v in renderings.items()}
        cameras = self.ellipse_cameras[to_render_indices]

        pred_rgb, pred_depth = self.get_diffusion_outputs(cameras, renderings)

        focals_for_decode = torch.cat([cameras.fx, cameras.fy], -1)
        pp_for_decode = torch.cat([cameras.cx, cameras.cy], -1)
        ray_bundle = cameras.generate_rays(camera_indices=torch.arange(len(cameras)).view(-1, 1))
        pixel_area_2 = ray_bundle.pixel_area.squeeze(-1).permute(2, 0, 1)
        directions_norm_2 = ray_bundle.metadata["directions_norm"].squeeze(-1).permute(2, 0, 1)
        decode_scale_mult = (pixel_area_2 * directions_norm_2).view(cameras.shape[0], -1)

        c2ws_opencv, w2cs_opencv = self.c2ws_opengl_to_opencv(cameras.camera_to_worlds)

        image_width = cameras.width[0].item()  # Assume that all camera dims are equal
        image_height = cameras.height[0].item()
        pixels_im = torch.stack(
            torch.meshgrid(
                torch.arange(image_height, dtype=torch.float32, device=cameras.device),
                torch.arange(image_width, dtype=torch.float32, device=cameras.device),
                indexing="ij")).permute(2, 1, 0)
        pixels = pixels_im.reshape(-1, 2)
        pts3d = self.depth_to_pts3d(pred_depth, c2ws_opencv, pixels, focals_for_decode, pp_for_decode).reshape(-1, 3)
        train_width = self.train_cameras.width[0].item()
        train_height = self.train_cameras.height[0].item()

        should_run_ga = False
        pred_rgb_chw_m1_1 = pred_rgb.permute(0, 3, 1, 2).unsqueeze(0) * 2 - 1
        overlapping_indices = self.overlapping_images(self.train_w2cs_opencv,
                                                      pts3d.unsqueeze(0).expand(self.train_w2cs_opencv.shape[0], -1,
                                                                                -1),
                                                      self.train_cameras.fx,
                                                      self.train_cameras.fy,
                                                      self.train_cameras.cx,
                                                      self.train_cameras.cy,
                                                      self.train_depths.view(self.train_w2cs_opencv.shape[0],
                                                                             train_height,
                                                                             train_width, 1),
                                                      train_width,
                                                      train_height)
        if overlapping_indices.shape[0] > 0:
            loaded_images = self.update_depth_outputs(overlapping_indices, pred_rgb_chw_m1_1, True, False, None)
            self.update_depth_outputs(overlapping_indices, pred_rgb_chw_m1_1, False, True, loaded_images)
            should_run_ga = True

        if self.prev_depths is not None:
            prev_cameras = self.ellipse_cameras[self.prev_indices]
            overlapping_indices = self.train_cameras.shape[0] + self.overlapping_images(self.prev_w2cs,
                                                                                        pts3d.unsqueeze(0).expand(
                                                                                            self.prev_w2cs.shape[0],
                                                                                            -1, -1),
                                                                                        prev_cameras.fx,
                                                                                        prev_cameras.fy,
                                                                                        prev_cameras.cx,
                                                                                        prev_cameras.cy,
                                                                                        self.prev_depths,
                                                                                        image_width,
                                                                                        image_height)
            if overlapping_indices.shape[0] > 0:
                self.update_depth_outputs(overlapping_indices, pred_rgb_chw_m1_1, True, True, None)
                should_run_ga = True

        self.depth_c2ws_opencv = torch.cat([self.depth_c2ws_opencv, c2ws_opencv])
        self.depth_focals = torch.cat([self.depth_focals, cameras.fx])
        self.depth_pp = torch.cat([self.depth_pp, pp_for_decode])
        self.depth_scale_mult = torch.cat([self.depth_scale_mult, decode_scale_mult.view(-1, 1)])

        self.prev_indices = torch.cat([self.prev_indices, to_render_indices]) \
            if self.prev_indices is not None else to_render_indices
        self.prev_w2cs = torch.cat([self.prev_w2cs, w2cs_opencv]) if self.prev_w2cs is not None else w2cs_opencv

        self.pred_rgbs = torch.cat([self.pred_rgbs, pred_rgb.view(-1, 3)])

        if should_run_ga:
            with torch.enable_grad():
                scene = global_aligner(self.depth_outputs, device=pred_rgb.device,
                                       mode=GlobalAlignerMode.PointCloudOptimizer, optimize_pp=True)
                scene.preset_pose(self.depth_c2ws_opencv)
                scene.preset_focal(self.depth_focals.cpu())  # Assume fx and fy are almost equal
                scene.preset_principal_point(self.depth_pp.cpu())
                scene.compute_global_alignment(init="known_poses", niter=300, schedule="cosine", lr=0.1)

            scene.clean_pointcloud()

            depth_maps = scene.get_depthmaps(raw=False)
            im_conf = [x for x in scene.im_conf]
            num_train = self.train_depths.shape[0]
            train_mask = torch.stack(im_conf[:num_train]) > self.min_conf_thresh
            self.train_depths = torch.where(train_mask, torch.stack(depth_maps[:num_train]), 0).unsqueeze(-1)
            prev_mask = torch.stack(im_conf[num_train:]) > 1e-5  # self.min_conf_thresh
            self.prev_depths = torch.where(prev_mask, torch.stack(depth_maps[num_train:]), 0).unsqueeze(-1)

            mask = torch.cat([train_mask.view(-1), prev_mask.view(-1)])
            means = torch.cat([x.view(-1, 3) for x in scene.get_pts3d(raw=False)])[mask]
            rgb = self.pred_rgbs[mask]
            scales = torch.cat([self.train_depths.view(-1, 1), self.prev_depths.view(-1, 1)])[mask] * \
                     self.depth_scale_mult[mask]

            out = {
                "means": means,
                "features_dc": torch.logit(rgb),
                "features_rest": torch.zeros(rgb.shape[0], *self.features_rest.shape[1:],
                                             device=self.features_rest.device),
                "opacities": torch.logit(torch.full_like(means[:, :1], 1 - 1e-5)),
                "scales": torch.log(scales.expand(-1, 3)),
                "quats": random_quat_tensor(scales.shape[0]).to(self.quats.device),
            }
        else:
            self.prev_depths = torch.cat([self.prev_depths, pred_depth]) if self.pred_depths is not None else pred_depth
            scales = pred_depth.view(-1, 1) * decode_scale_mult.view(-1, 1)
            out = {
                "means": torch.cat([self.means, pts3d]),
                "features_dc": torch.cat([self.features_dc, torch.logit(pred_rgb.view(-1, 3))]),
                "features_rest": torch.cat(
                    [self.features_rest, torch.zeros(scales.shape[0], *self.features_rest.shape[1:],
                                                     device=self.features_rest.device)]),
                "opacities": torch.cat([self.opacities, torch.logit(torch.full_like(pts3d[:, :1], 1 - 1e-5))]),
                "scales": torch.cat([self.scales, torch.log(scales).expand(-1, 3)]),
                "quats": torch.cat([self.quats, random_quat_tensor(scales.shape[0]).to(self.quats.device)]),
            }

        for name, param in self.gauss_params.items():
            self.gauss_params[name] = torch.nn.Parameter(out[name])

        param_groups = self.get_gaussian_param_groups()
        for group, new_params in param_groups.items():
            optimizer = optimizers.optimizers[group]
            param = optimizer.param_groups[0]["params"][0]
            param_state = optimizer.state[param]

            if "exp_avg" in param_state:
                param_state["exp_avg"] = torch.zeros(out["means"].shape[0], *param_state["exp_avg"].shape[1:],
                                                     device=param_state["exp_avg"].device,
                                                     dtype=param_state["exp_avg"].dtype)
                param_state["exp_avg_sq"] = torch.zeros(out["means"].shape[0], *param_state["exp_avg_sq"].shape[1:],
                                                        device=param_state["exp_avg_sq"].device,
                                                        dtype=param_state["exp_avg_sq"].dtype)
            del optimizer.state[param]
            optimizer.state[new_params[0]] = param_state
            optimizer.param_groups[0]["params"] = new_params
            del param

        for i, rgb in enumerate(pred_rgb):
            self.image_filenames.append(self.add_image_fn(rgb, cameras[i:i + 1]))

        # chunk_size = self.means.shape[0]
        # for i in range(0, pts3d.shape[0], chunk_size):
        #     end = i + chunk_size
        #
        #     if i == 0:
        #         new_opacities = torch.cat([torch.logit(torch.zeros_like(self.opacities) + 1e-5),
        #                                    torch.logit(torch.full_like(pts3d[i:end, :1], 1 - 1e-5))])
        #     else:
        #         new_opacities = torch.cat([self.opacities, torch.logit(torch.full_like(pts3d[i:end, :1], 1 - 1e-5))])
        #
        #     out = {
        #         "means": torch.cat([self.means, pts3d[i:end]]),
        #         "features_dc": torch.cat([self.features_dc, torch.logit(pred_rgb[i:end])]),
        #         "features_rest": torch.cat(
        #             [self.features_rest, torch.zeros(scales[i:end].shape[0], *self.features_rest.shape[1:],
        #                                              device=self.features_rest.device)]),
        #         # "opacities": torch.cat([self.opacities, torch.logit(torch.full_like(pts3d[i:end, :1], 1 - 1e-5))]),
        #         "opacities": new_opacities,
        #         "scales": torch.cat([self.scales, torch.log(scales[i:end]).expand(-1, 3)]),
        #         "quats": torch.cat([self.quats, random_quat_tensor(scales[i:end].shape[0]).to(self.quats.device)]),
        #     }
        #
        #     # TODO this will break if the size of self.means is smaller than what we're adding
        #     split_idcs = torch.zeros(self.means.shape[0], dtype=torch.bool, device=scales.device)
        #     split_idcs[-scales[i:end].shape[0]:] = True
        #
        #     for name, param in self.gauss_params.items():
        #         self.gauss_params[name] = torch.nn.Parameter(out[name])
        #     self.dup_in_all_optim(optimizers, split_idcs, 1)
        # # import pdb;
        # # pdb.set_trace()

        self.xys_grad_norm = None
        self.vis_counts = None
        self.max_2Dsize = None
        # trigger.unlink()
        CONSOLE.log(f"Finished rendering {to_render_indices}")

    @torch.inference_mode
    def get_diffusion_outputs(self, cameras: Cameras, renderings: Dict[str, torch.Tensor]):
        c_crossattn_1, c_crossattn_2 = self.get_clip_embeddings()
        c_crossattn_1 = c_crossattn_1.expand(cameras.shape[0], -1, -1)
        c_crossattn_2 = c_crossattn_2.expand(cameras.shape[0], -1, -1)

        depth = renderings[DEPTH]
        depth_scaling_factor = self.get_depth_scaling_factor(depth.view(cameras.shape[0], -1))

        cameras_1 = deepcopy(cameras)
        cameras_1.rescale_output_resolution(0.25)
        image_width_1 = cameras_1.width[0].item()  # Assume that all camera dims are equal
        image_height_1 = cameras_1.height[0].item()

        rgb_1 = F.interpolate(renderings[RGB].permute(0, 3, 1, 2), (image_height_1, image_width_1),
                              mode="bicubic", antialias=True).permute(0, 2, 3, 1)
        depth_1 = F.interpolate(depth.permute(0, 3, 1, 2), (image_height_1, image_width_1), mode="bicubic",
                                antialias=True).permute(0, 2, 3, 1)
        encoded_1 = self.encode_rgbd(rgb_1, depth_1, depth_scaling_factor)
        acc_1 = F.interpolate(renderings[ACCUMULATION].permute(0, 3, 1, 2), (image_height_1, image_width_1),
                              mode="bicubic", antialias=True)
        c_concat_1 = torch.cat([encoded_1, acc_1], 1)
        acc_1 = acc_1.permute(0, 2, 3, 1)

        if self.config.guidance_scale > 1:
            c_concat_1 = torch.cat([torch.zeros_like(c_concat_1), c_concat_1])
            c_crossattn_1 = torch.cat([torch.zeros_like(c_crossattn_1), c_crossattn_1])
            c_crossattn_2 = torch.cat([torch.zeros_like(c_crossattn_2), c_crossattn_2])

        # image_dim_1 = cameras.width[0].item()
        # encoded_1 = self.encode_with_vae(renderings[RGB], depth, depth_scaling_factor)
        # acc_1 = F.interpolate(renderings[ACCUMULATION].permute(0, 3, 1, 2),
        #                       (image_height_1 // self.vae_scale_factor, image_width_1 // self.vae_scale_factor),
        #                       mode="bicubic", antialias=True)
        # c_concat_1 = torch.cat([encoded_1, acc_1], 1)

        focals_for_decode_1 = torch.cat([cameras_1.fx, cameras_1.fy], -1)
        pp_for_decode_1 = torch.cat([cameras_1.cx, cameras_1.cy], -1)

        pixels_im = torch.stack(
            torch.meshgrid(
                torch.arange(image_height_1, dtype=torch.float32, device=cameras.device),
                torch.arange(image_width_1, dtype=torch.float32, device=cameras.device),
                indexing="ij")).permute(2, 1, 0)
        pixels_1 = pixels_im.reshape(-1, 2)

        train_width = self.train_cameras.width[0].item()
        train_height = self.train_cameras.height[0].item()
        if self.prev_indices is not None:
            prev_width = self.ellipse_cameras[self.prev_indices].width[0].item()
            prev_height = self.ellipse_cameras[self.prev_indices].height[0].item()

        c2ws_opencv, w2cs_opencv = self.c2ws_opengl_to_opencv(cameras.camera_to_worlds)
        sample_1 = self.scheduler.init_noise_sigma * randn_tensor((cameras.shape[0], 4, image_height_1, image_width_1),
                                                                  device=c_concat_1.device, dtype=c_concat_1.dtype)
        # sample_1 = scheduler.init_noise_sigma * randn_tensor(
        #     (to_render_indices.shape[0], 4, image_height_1 // self.vae_scale_factor, image_width_1 // self.vae_scale_factor),
        #     device=c_concat_1.device, dtype=c_concat_1.dtype)

        for t_index, t in enumerate(self.scheduler.timesteps):
            model_input = torch.cat([sample_1] * 2) if self.config.guidance_scale > 1 else sample_1
            model_input = self.scheduler.scale_model_input(model_input, t)
            model_input = torch.cat([model_input, c_concat_1], dim=1)

            model_output = self.unet_1(model_input, t, c_crossattn_1, return_dict=False)[0]

            if self.config.guidance_scale > 1:
                model_output_uncond, model_output_cond = model_output.chunk(2)
                model_output = model_output_uncond + self.config.guidance_scale * \
                               (model_output_cond - model_output_uncond)

            if self.config.blend_with_concat:
                alpha_prod_t = self.scheduler.alphas_cumprod.to(self.device)[t].view(-1, 1, 1, 1)
                beta_prod_t = 1 - alpha_prod_t
                if self.config.prediction_type == "epsilon":
                    pred_original_sample = (sample_1 - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
                elif self.config.prediction_type == "sample":
                    pred_original_sample = model_output
                elif self.config.prediction_type == "v_prediction":
                    pred_original_sample = (alpha_prod_t ** 0.5) * sample_1 - (beta_prod_t ** 0.5) * model_output
                else:
                    raise ValueError(f"Unknown prediction type {self.config.prediction_type}")

                pred_rgb, pred_depth = self.decode_rgbd(pred_original_sample, depth_scaling_factor)

                pts3d = self.depth_to_pts3d(pred_depth, c2ws_opencv, pixels_1, focals_for_decode_1,
                                            pp_for_decode_1).reshape(-1, 3)

                mask = depth_1 > pred_depth
                if self.config.mask_with_existing_depth:
                    depth_mask = self.not_in_existing_predictions(self.train_w2cs_opencv,
                                                                  pts3d.unsqueeze(0).expand(
                                                                      self.train_w2cs_opencv.shape[0], -1, -1),
                                                                  self.train_cameras.fx,
                                                                  self.train_cameras.fy,
                                                                  self.train_cameras.cx,
                                                                  self.train_cameras.cy,
                                                                  self.train_depths,
                                                                  train_width,
                                                                  train_height)
                    if self.prev_depths is not None:
                        prev_cameras = self.ellipse_cameras[self.prev_indices]
                        depth_mask = torch.logical_and(self.not_in_existing_predictions(self.prev_w2cs,
                                                                                        pts3d.unsqueeze(0).expand(
                                                                                            self.prev_w2cs.shape[0], -1,
                                                                                            -1),
                                                                                        prev_cameras.fx,
                                                                                        prev_cameras.fy,
                                                                                        prev_cameras.cx,
                                                                                        prev_cameras.cy,
                                                                                        self.prev_depths,
                                                                                        prev_width,
                                                                                        prev_height), depth_mask)

                    mask = torch.logical_and(mask, depth_mask)

                blend_weight = torch.where(mask, 0, acc_1)

                model_output = self.encode_rgbd(
                    (1 - blend_weight) * pred_rgb + blend_weight * rgb_1,
                    (1 - blend_weight) * pred_depth + blend_weight * depth_1,
                    depth_scaling_factor)

            sample_1 = self.scheduler.step(model_output, t, sample_1, return_dict=False)[0]

        image_width_2 = cameras.width[0].item()  # Assume that all camera dims are equal
        image_height_2 = cameras.height[0].item()
        encoded_2 = self.encode_rgbd(renderings[RGB], depth, depth_scaling_factor)
        acc_2 = renderings[ACCUMULATION]

        focals_for_decode_2 = torch.cat([cameras.fx, cameras.fy], -1)
        pp_for_decode_2 = torch.cat([cameras.cx, cameras.cy], -1)
        pixels_im = torch.stack(
            torch.meshgrid(
                torch.arange(image_height_2, dtype=torch.float32, device=cameras.device),
                torch.arange(image_width_2, dtype=torch.float32, device=cameras.device),
                indexing="ij")).permute(2, 1, 0)
        pixels_2 = pixels_im.reshape(-1, 2)

        c_concat_2 = torch.cat(
            [F.interpolate(sample_1, (image_height_2, image_width_2), mode="bicubic", antialias=True), encoded_2,
             acc_2.permute(0, 3, 1, 2)], 1)

        if self.config.guidance_scale > 1:
            c_concat_2 = torch.cat([torch.zeros_like(c_concat_2), c_concat_2])

        sample_2 = self.scheduler.init_noise_sigma * randn_tensor(
            (cameras.shape[0], 4, image_height_2, image_width_2),
            device=c_concat_2.device, dtype=c_concat_2.dtype)

        for t_index, t in enumerate(self.scheduler.timesteps):
            model_input = torch.cat([sample_2] * 2) if self.config.guidance_scale > 1 else sample_2
            model_input = self.scheduler.scale_model_input(model_input, t)
            model_input = torch.cat([model_input, c_concat_2], dim=1)

            model_output = self.unet_2(model_input, t, c_crossattn_2,
                                       class_labels=torch.tensor([t], device=model_input.device),
                                       return_dict=False)[0]

            if self.config.guidance_scale > 1:
                model_output_uncond, model_output_cond = model_output.chunk(2)
                model_output = model_output_uncond + self.config.guidance_scale * \
                               (model_output_cond - model_output_uncond)


            # if self.config.blend_with_concat:
            alpha_prod_t = self.scheduler.alphas_cumprod.to(self.device)[t].view(-1, 1, 1, 1)
            beta_prod_t = 1 - alpha_prod_t
            if self.config.prediction_type == "epsilon":
                pred_original_sample = (sample_2 - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
            elif self.config.prediction_type == "sample":
                pred_original_sample = model_output
            elif self.config.prediction_type == "v_prediction":
                pred_original_sample = (alpha_prod_t ** 0.5) * sample_2 - (beta_prod_t ** 0.5) * model_output
            else:
                raise ValueError(f"Unknown prediction type {self.config.prediction_type}")

            pred_rgb, pred_depth = self.decode_rgbd(pred_original_sample, depth_scaling_factor)

            pts3d = (self.depth_to_pts3d(pred_depth, c2ws_opencv, pixels_2, focals_for_decode_2, pp_for_decode_2)
                     .reshape(-1, 3))

            depth_mask = self.not_in_existing_predictions(self.train_w2cs_opencv,
                                                          pts3d.unsqueeze(0).expand(self.train_w2cs_opencv.shape[0], -1,
                                                                                    -1),
                                                          self.train_cameras.fx,
                                                          self.train_cameras.fy,
                                                          self.train_cameras.cx,
                                                          self.train_cameras.cy,
                                                          self.train_depths,
                                                          train_width,
                                                          train_height)
            if self.prev_depths is not None:
                depth_mask = torch.logical_and(self.not_in_existing_predictions(self.prev_w2cs,
                                                                                pts3d.unsqueeze(0).expand(
                                                                                    self.prev_w2cs.shape[0], -1, -1),
                                                                                prev_cameras.fx,
                                                                                prev_cameras.fy,
                                                                                prev_cameras.cx,
                                                                                prev_cameras.cy,
                                                                                self.prev_depths,
                                                                                prev_width,
                                                                                prev_height), depth_mask)

            blend_weight = torch.where(torch.logical_and(depth > pred_depth, depth_mask.view(depth.shape)), 0, acc_2)

            model_output = self.encode_rgbd(
                (1 - blend_weight) * pred_rgb + blend_weight * renderings[RGB],
                (1 - blend_weight) * pred_depth + blend_weight * depth,
                depth_scaling_factor)

            sample_2 = self.scheduler.step(model_output, t, sample_2, return_dict=False)[0]

        return self.decode_rgbd(sample_2, depth_scaling_factor)

    def update_depth_outputs(self, overlapping_indices: torch.Tensor, pred_rgb_chw_m1_1: torch.Tensor,
                             include_first: bool, include_second: bool, loaded_images: Optional[List]) -> None:
        view1 = {"img": [], "idx": [], "instance": []}
        view2 = {"img": [], "idx": [], "instance": []}
        if loaded_images is None:
            loaded_images = load_images([str(self.image_filenames[x]) for x in overlapping_indices],
                                        size=512 if (not include_first or not include_second) else max(
                                            pred_rgb_chw_m1_1.shape[-2:]))
        for first, rgb in enumerate(pred_rgb_chw_m1_1):
            for second, overlapping_index in enumerate(overlapping_indices):
                first_idx = len(self.image_filenames) + first
                second_img = loaded_images[second]["img"].to(pred_rgb_chw_m1_1.device)

                if include_first:
                    view1["img"].append(rgb)
                    view2["img"].append(second_img)
                    view1["idx"].append(first_idx)
                    view1["instance"].append(str(first_idx))
                    view2["idx"].append(overlapping_index)
                    view2["instance"].append(str(overlapping_index))

                if include_second:
                    view2["img"].append(rgb)
                    view1["img"].append(second_img)
                    view2["idx"].append(first_idx)
                    view2["instance"].append(str(first_idx))
                    view1["idx"].append(overlapping_index)
                    view1["instance"].append(str(overlapping_index))

        view1["img"] = torch.cat(view1["img"])
        view2["img"] = torch.cat(view2["img"])

        pred1, pred2 = self.depth_model(view1, view2)
        new = {"view1": view1, "view2": view2, "pred1": pred1, "pred2": pred2}
        for key in self.depth_outputs:
            for subkey in self.depth_outputs[key]:
                self.depth_outputs[key][subkey] += [x for x in new[key][subkey]]

        return loaded_images

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
    def get_empty_outputs(width: int, height: int, background: torch.Tensor) -> Dict[str, Union[torch.Tensor, List]]:
        rgb = background.repeat(height, width, 1)
        depth = background.new_ones(*rgb.shape[:2], 1) * 10
        accumulation = background.new_zeros(*rgb.shape[:2], 1)
        return {"rgb": rgb, "depth": depth, "accumulation": accumulation, "background": background}

    def _get_background_color(self):
        if self.config.background_color == "none":
            background = None
        elif self.config.background_color == "random":
            if self.training:
                background = torch.rand(3, device=self.device)
            else:
                background = self.background_color.to(self.device)
        elif self.config.background_color == "white":
            background = torch.ones(3, device=self.device)
        elif self.config.background_color == "black":
            background = torch.zeros(3, device=self.device)
        else:
            raise ValueError(f"Unknown background color {self.config.background_color}")
        return background

    def get_outputs(self, camera: Cameras) -> Dict[str, Union[torch.Tensor, List]]:
        """Takes in a camera and returns a dictionary of outputs.

        Args:
            camera: The camera(s) for which output images are rendered. It should have
            all the needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """
        if not isinstance(camera, Cameras):
            print("Called get_outputs with not a camera")
            return {}

        if not self.training and self.config.render_with_diffusion:
            camera = deepcopy(camera)
            padding_w = (64 - camera.width % 64) % 64
            camera.width += padding_w

            padding_h = (64 - camera.height % 64) % 64
            camera.height += padding_h

        optimized_camera_to_world = camera.camera_to_worlds

        # cropping
        if self.crop_box is not None and not self.training:
            crop_ids = self.crop_box.within(self.means).squeeze()
            if crop_ids.sum() == 0:
                return self.get_empty_outputs(
                    int(camera.width.item()), int(camera.height.item()), self.background_color
                )
        else:
            crop_ids = None

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
        camera_scale_fac = self._get_downscale_factor()
        camera.rescale_output_resolution(1 / camera_scale_fac)
        viewmat = get_viewmat(optimized_camera_to_world)
        K = camera.get_intrinsics_matrices().cuda()
        W, H = int(camera.width.item()), int(camera.height.item())
        self.last_size = (H, W)
        camera.rescale_output_resolution(camera_scale_fac)  # type: ignore

        # apply the compensation of screen space blurring to gaussians
        if self.config.rasterize_mode not in ["antialiased", "classic"]:
            raise ValueError("Unknown rasterize_mode: %s", self.config.rasterize_mode)

        if self.config.output_depth_during_training or not self.training:
            render_mode = "RGB+ED"
        else:
            render_mode = "RGB"

        if self.config.sh_degree > 0:
            sh_degree_to_use = min(self.step // self.config.sh_degree_interval, self.config.sh_degree)
        else:
            colors_crop = torch.sigmoid(colors_crop).squeeze(1)  # [N, 1, 3] -> [N, 3]
            sh_degree_to_use = None

        render, alpha, info = rasterization(
            means=means_crop,
            quats=quats_crop / quats_crop.norm(dim=-1, keepdim=True),
            scales=torch.exp(scales_crop),
            opacities=torch.sigmoid(opacities_crop).squeeze(-1),
            colors=colors_crop,
            viewmats=viewmat,  # [1, 4, 4]
            Ks=K,  # [1, 3, 3]
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
            # set some threshold to disregrad small gaussians for faster rendering.
            # radius_clip=3.0,
        )
        if self.training and info["means2d"].requires_grad:
            info["means2d"].retain_grad()
        self.xys = info["means2d"]  # [1, N, 2]
        self.radii = info["radii"][0]  # [N]
        alpha = alpha[:, ...]

        background = self._get_background_color()
        if background is None:
            rgb = render[:, ..., :3] / alpha.clamp_min(1e-8)
        else:
            rgb = render[:, ..., :3] + (1 - alpha) * background

        rgb = torch.clamp(rgb, 0.0, 1.0)

        if render_mode == "RGB+ED":
            depth_im = render[:, ..., 3:4]
            depth_im = torch.where(alpha > 0, depth_im, depth_im.detach().max()).squeeze(0)
        else:
            depth_im = None

        if background is not None and background.shape[0] == 3 and not self.training:
            background = background.expand(H, W, 3)

        result = {
            "rgb": rgb.squeeze(0),  # type: ignore
            "depth": depth_im,  # type: ignore
            "accumulation": alpha.squeeze(0),  # type: ignore
            "background": background,  # type: ignore
        }

        if not self.training and self.config.render_with_diffusion:
            pred_rgb, pred_depth = self.get_diffusion_outputs(camera, {RGB: rgb, DEPTH: depth_im.unsqueeze(0),
                                                                       ACCUMULATION: alpha})
            result["diffusion_rgb"] = pred_rgb.squeeze(0)
            result["diffusion_depth"] = pred_depth.squeeze(0)
            for key, val in result.items():
                if val is not None:
                    result[key] = val[:-padding_h, :-padding_w]

        return result  # type: ignore

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
        gt_chw = gt_img.permute(2, 0, 1)[None, ...]
        pred_chw = pred_img.permute(2, 0, 1)[None, ...]
        ssim_loss = 1 - self.ssim(gt_chw, pred_chw)
        lpips_loss = self.lpips(gt_chw, pred_chw)

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
            "main_loss": Ll1 + self.config.ssim_lambda * ssim_loss + self.config.lpips_lambda * lpips_loss,
            "scale_reg": scale_reg,
        }

        loss_mult = 1 if self.config.optimize else 0

        return {k: loss_mult * v for k, v in loss_dict.items()}

    @torch.no_grad()
    def get_outputs_for_camera(self, camera: Cameras, obb_box: Optional[OrientedBox] = None) -> Dict[str, torch.Tensor]:
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

        if self.config.render_with_diffusion:
            gt_rgb = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])

            predicted_rgb = outputs["diffusion_rgb"]
            combined_rgb = torch.cat([gt_rgb, predicted_rgb], dim=1)
            images_dict["diffusion_img"] = combined_rgb

            # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
            gt_rgb = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
            predicted_rgb = torch.moveaxis(predicted_rgb, -1, 0)[None, ...]

            psnr = self.psnr(gt_rgb, predicted_rgb)
            ssim = self.ssim(gt_rgb, predicted_rgb)
            lpips = self.lpips(gt_rgb, predicted_rgb)

            # all of these metrics will be logged as scalars
            metrics_dict["diffusion_psnr"] = float(psnr.item())
            metrics_dict["diffusion_ssim"] = float(ssim)
            metrics_dict["diffusion_lpips"] = float(lpips)

        return metrics_dict, images_dict

    def encode_rgbd(self, rgb: torch.Tensor, depth: Optional[torch.Tensor],
                    depth_scaling_factor: torch.Tensor) -> torch.Tensor:
        if self.config.rgb_mapping == "clamp":
            to_encode = (2 * rgb - 1).clamp(-1, 1).permute(0, 3, 1, 2)
        elif self.config.rgb_mapping == "sigmoid":
            to_encode = torch.logit((rgb.float().clamp(0, 1) * 0.99998) + 0.00001).permute(0, 3, 1, 2)
        else:
            raise Exception(self.config.rgb_mapping)

        encoded_depth = self.encode_depth(depth, depth_scaling_factor).permute(0, 3, 1, 2)
        to_encode = torch.cat([to_encode, encoded_depth], 1)

        return to_encode

    def decode_rgbd(self, to_decode: torch.Tensor, depth_scaling_factor: torch.Tensor) -> Tuple[
        torch.Tensor, Optional[torch.Tensor]]:
        to_decode = to_decode.permute(0, 2, 3, 1)

        rgb = self.decode_rgb(to_decode[..., :3])

        return rgb, self.decode_depth(to_decode[..., 3:], depth_scaling_factor)

    def decode_rgb(self, encoded_rgb: torch.Tensor) -> torch.Tensor:
        if self.config.rgb_mapping == "clamp":
            return (0.5 * (encoded_rgb + 1)).clamp(0, 1)
        elif self.config.rgb_mapping == "sigmoid":
            return ((torch.sigmoid(encoded_rgb) - 0.00001) / 0.99998).clamp(0, 1)
        else:
            raise Exception(self.config.rgb_mapping)

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

    def get_depth_scaling_factor(self, depth_to_scale: torch.Tensor) -> torch.Tensor:
        if self.config.depth_mapping == "scaled":
            return torch.quantile(depth_to_scale, torch.FloatTensor([0.02, 0.98]).to(depth_to_scale.device),
                                  dim=-1).T.view(-1, 2, 1, 1)
        elif self.config.depth_mapping == "disparity" or self.config.depth_mapping == "log":
            return depth_to_scale.mean(dim=-1).view(-1, 1, 1, 1)
        else:
            raise Exception(self.config.depth_mapping)

    def encode_with_vae(self, rgb: torch.Tensor, depth: Optional[torch.Tensor], depth_scaling_factor: torch.Tensor,
                        sample_override: bool = False) -> torch.Tensor:
        to_encode = rgb.permute(0, 3, 1, 2) * 2 - 1
        encoded_depth = self.encode_depth(depth, depth_scaling_factor).permute(0, 3, 1, 2)
        to_encode = torch.cat([to_encode, encoded_depth], 1)

        return self.encode_with_vae_inner(to_encode, sample_override)

    def encode_with_vae_inner(self, to_encode: torch.Tensor, sample_override: bool = False,
                              autocast_enabled: bool = False) -> torch.Tensor:
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

    def decode_with_vae(self, to_decode: torch.Tensor, depth_scaling_factor: torch.Tensor) -> Tuple[
        torch.Tensor, Optional[torch.Tensor]]:
        decoded = self.decode_with_vae_inner(to_decode).permute(0, 2, 3, 1)
        rgb = (0.5 * (decoded[..., :3] + 1)).clamp(0, 1)

        return rgb, self.decode_depth(decoded[..., 3:], depth_scaling_factor)

    def decode_with_vae_inner(self, to_decode: torch.Tensor, autocast_enabled: bool = False) -> torch.Tensor:
        with torch.cuda.amp.autocast(enabled=autocast_enabled):
            decoded = self.vae.decode(to_decode / self.vae.config.scaling_factor, return_dict=False)[0]

        if autocast_enabled:
            to_upcast = torch.logical_not(torch.isfinite(decoded.view(decoded.shape[0], -1)).all(dim=-1))
            if to_upcast.any():
                decoded[to_upcast] = self.decode_with_vae_inner(decoded[to_upcast].float(), False)

        return decoded

    def project_predictions(self, w2cs: torch.Tensor, c2ws: torch.Tensor, predictions: torch.Tensor,
                            pixels: torch.Tensor, focals: torch.Tensor, pp: torch.Tensor,
                            bg_colors: Optional[torch.Tensor], depth_scaling_factor: torch.Tensor,
                            decode_scale_mult: torch.Tensor, image_dim: int) -> torch.Tensor:
        rgb, depth = self.decode_rgbd(predictions, depth_scaling_factor)

        # rgb, depth = self.decode_with_vae(predictions, depth_scaling_factor)
        pts3d = self.depth_to_pts3d(depth, c2ws, pixels, focals, pp)

        base_rgbs = []
        base_depth = []
        other_rgbs = []
        other_pts3d = []
        for i in range(w2cs.shape[0]):
            cur_other_rgbs = []
            cur_other_pts3d = []
            for j in range(w2cs.shape[0]):
                if i == j:
                    base_rgbs.append(rgb[j])
                    base_depth.append(depth[j])
                else:
                    cur_other_rgbs.append(rgb[j])
                    cur_other_pts3d.append(pts3d[j])
            other_rgbs.append(torch.cat(cur_other_rgbs))
            other_pts3d.append(torch.cat(cur_other_pts3d))

        base_rgbs = torch.stack(base_rgbs)
        base_depth = torch.stack(base_depth)
        other_rgbs = torch.stack(other_rgbs).view(w2cs.shape[0], -1, 3)
        other_pts3d = torch.stack(other_pts3d).view(w2cs.shape[0], -1, 3)

        scales = []
        opacities = []
        ordering = torch.randperm(w2cs.shape[0])
        for i in range(w2cs.shape[0]):
            cur_scales = []
            cur_opacities = []
            for j in range(w2cs.shape[0]):
                if i != j:
                    cur_scales.append(depth[j].view(-1) * decode_scale_mult[j])
                    if ordering[i] < ordering[j]:
                        cur_opacities.append(torch.zeros_like(cur_scales[-1]))
                    else:
                        cur_opacities.append(torch.ones_like(cur_scales[-1]))
            scales.append(torch.cat(cur_scales))
            opacities.append(torch.cat(cur_opacities))
        scales = torch.stack(scales).unsqueeze(-1)
        opacities = torch.stack(opacities)

        rendering = RGBDDiffusionBase.splat_gaussians(
            w2cs,
            focals[..., :1],
            focals[..., 1:],
            pp[..., :1],
            pp[..., 1:],
            torch.cat([self.means.unsqueeze(0).expand(other_pts3d.shape[0], -1, -1), other_pts3d], 1),
            torch.cat([self.scales.exp().unsqueeze(0).expand(other_pts3d.shape[0], -1, -1), scales.expand(-1, -1, 3)],
                      1),
            torch.cat([torch.sigmoid(self.opacities).T.expand(other_pts3d.shape[0], -1),
                       opacities], 1),
            torch.cat([torch.sigmoid(self.features_dc).unsqueeze(0).expand(other_pts3d.shape[0], -1, -1), other_rgbs],
                      1),
            image_dim,
            True,
            1,
            bg_colors,
        )

        blend_weight = rendering[ACCUMULATION]
        other_depth = rendering[DEPTH]
        blend_weight[other_depth > base_depth] = 0

        return self.encode_rgbd((1 - blend_weight) * base_rgbs + blend_weight * rendering[RGB],
                                (1 - blend_weight) * base_depth + blend_weight * other_depth, depth_scaling_factor)
        # return self.encode_with_vae((1 - blend_weight) * base_rgbs + blend_weight * rendering[RGB],
        #                             (1 - blend_weight) * base_depth + blend_weight * other_depth, depth_scaling_factor)

    def not_in_existing_predictions(self, w2cs: torch.Tensor, pts3d: torch.Tensor, fx: torch.Tensor,
                                    fy: torch.Tensor, cx: torch.Tensor, cy: torch.Tensor,
                                    existing_depth: torch.Tensor, image_width: int, image_height: int) -> torch.Tensor:
        _, min_indices = self.project_depth_to_other_views(w2cs, pts3d, fx, fy, cx, cy, existing_depth, image_width,
                                                           image_height)
        invalid_indices = torch.where(min_indices < pts3d.shape[0] * pts3d.shape[1], min_indices % pts3d.shape[1], -1)
        mask = torch.ones_like(pts3d[0, :, 0], dtype=torch.bool)
        mask[invalid_indices[invalid_indices > 0]] = False
        return mask

    def overlapping_images(self, w2cs: torch.Tensor, pts3d: torch.Tensor, fx: torch.Tensor,
                           fy: torch.Tensor, cx: torch.Tensor, cy: torch.Tensor,
                           existing_depth: torch.Tensor, image_width: int, image_height: int) -> torch.Tensor:
        depth_map, _ = self.project_depth_to_other_views(w2cs, pts3d, fx, fy, cx, cy, existing_depth, image_width,
                                                         image_height)
        return torch.arange(existing_depth.shape[0], device=existing_depth.device)[
            (depth_map <= existing_depth).view(existing_depth.shape[0], -1).any(dim=-1)]

    def project_depth_to_other_views(self, w2cs: torch.Tensor, pts3d: torch.Tensor, fx: torch.Tensor,
                                     fy: torch.Tensor, cx: torch.Tensor, cy: torch.Tensor,
                                     existing_depth: torch.Tensor, image_width: int, image_height: int) -> Tuple[
        torch.Tensor, torch.Tensor]:
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

            is_valid_u = torch.logical_and(0 <= uv[..., 0], uv[..., 0] < image_width)
            is_valid_v = torch.logical_and(0 <= uv[..., 1], uv[..., 1] < image_height)
            is_valid_z = z_pos > 0
            is_valid_point = torch.logical_and(torch.logical_and(is_valid_u, is_valid_v), is_valid_z)

            max_val = torch.finfo(torch.float32).max
            u = torch.round(uv[..., 0]).long().clamp(0, image_width - 1)
            v = torch.round(uv[..., 1]).long().clamp(0, image_height - 1)
            z = torch.where(is_valid_point, z_pos, max_val)
            camera_index = torch.arange(w2cs.shape[0], device=z.device).view(-1, 1).expand(-1, z.shape[-1])
            indices = (camera_index * image_height * image_width + v * image_width + u).view(-1)
            _, min_indices = scatter_min(z.view(-1), indices, out=depth_map)
            return depth_map.view(existing_depth.shape), min_indices

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

    @staticmethod
    @torch.compile
    @torch.cuda.amp.autocast(enabled=False)
    def c2ws_opengl_to_opencv(c2ws_opengl: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        c2ws_opencv = torch.eye(4, device=c2ws_opengl.device).view(1, 4, 4).repeat(c2ws_opengl.shape[0], 1, 1)
        c2ws_opencv[:, :3] = c2ws_opengl
        c2ws_opencv[:, :, 1:3] *= -1  # opengl to opencv

        w2cs_opencv = torch.eye(4, device=c2ws_opengl.device).view(1, 4, 4).repeat(c2ws_opengl.shape[0], 1, 1)
        R_inv = c2ws_opencv[:, :3, :3].transpose(1, 2)
        w2cs_opencv[:, :3, :3] = R_inv
        T = c2ws_opengl[:, :3, 3:]
        inv_T = -torch.bmm(R_inv, T)
        w2cs_opencv[:, :3, 3:] = inv_T

        return c2ws_opencv, w2cs_opencv
