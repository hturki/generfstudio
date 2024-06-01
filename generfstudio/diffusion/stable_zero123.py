from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDIMScheduler
from jaxtyping import Float
from nerfstudio.utils.rich_utils import CONSOLE
try:
    from omegaconf import OmegaConf
except:
    pass
from torch import Tensor

from extern.ldm_zero123.util import instantiate_from_config


def load_model_from_config(config, ckpt, vram_O=True, verbose=True):
    pl_sd = torch.load(ckpt, map_location="cpu")

    if "global_step" in pl_sd and verbose:
        CONSOLE.print(f'Global Step: {pl_sd["global_step"]}')

    sd = pl_sd["state_dict"]

    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)

    if len(m) > 0 and verbose:
        CONSOLE.print("missing keys: \n", m)
    if len(u) > 0 and verbose:
        CONSOLE.print("unexpected keys: \n", u)

    # manually load ema and delete it to save GPU memory
    if model.use_ema:
        if verbose:
            CONSOLE.print("loading EMA...")
        model.model_ema.copy_to(model.model)
        del model.model_ema

    if vram_O:
        # we don't need decoder
        del model.first_stage_model.decoder

    torch.cuda.empty_cache()

    return model


class StableZero123(nn.Module):

    def __init__(self,
                 pretrained_model_name_or_path: str,
                 pretrained_config: str,
                 guidance_scale: float,
                 vram_O: bool = True,
                 freeze: bool = True,
                 img_dim=(256, 256),
                 num_train_timesteps: int = 1000,
                 ddim_linear_start=0.00085,
                 ddim_linear_end=0.012) -> None:
        super().__init__()

        config = OmegaConf.load(pretrained_config)
        self.model = load_model_from_config(
            config,
            pretrained_model_name_or_path,
            vram_O=vram_O,
        )

        if freeze:
            # Freeze the model
            for p in self.model.parameters():
                p.requires_grad_(False)
            self.model.eval()

        self.num_train_timesteps = num_train_timesteps

        self.scheduler = DDIMScheduler(
            self.num_train_timesteps,
            ddim_linear_start,
            ddim_linear_end,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )

        self.guidance_scale = guidance_scale
        self.img_dim = img_dim
        self.register_buffer("alphas_cumprod", self.scheduler.alphas_cumprod, persistent=False)

        CONSOLE.print("Loaded Stable Zero123!")

        self.device_indicator_param = nn.Parameter(torch.empty(0))

    @property
    def device(self):
        """Returns the device that the model is on."""
        return self.device_indicator_param.device

    # @torch.cuda.amp.autocast(enabled=False)
    def prepare_embeddings(self, image_path: Path, cond_elevation_deg: float, cond_azimuth_deg: float):
        # load cond image for zero123
        assert image_path.exists()
        rgba = cv2.cvtColor(
            cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA
        )
        rgba = (
                cv2.resize(rgba, self.img_dim, interpolation=cv2.INTER_AREA).astype(
                    np.float32
                )
                / 255.0
        )
        rgb = rgba[..., :3] * rgba[..., 3:] + (1 - rgba[..., 3:])
        rgb_tensor: Float[Tensor, "1 3 H W"] = (
            torch.from_numpy(rgb)
            .unsqueeze(0)
            .permute(0, 3, 1, 2)
            .contiguous()
            .to(self.device)
        )
        self.c_crossattn, self.c_concat = self.get_img_embeds(rgb_tensor)
        self.cond_elevation_deg = cond_elevation_deg
        self.cond_azimuth_deg = cond_azimuth_deg

    # @torch.cuda.amp.autocast(enabled=False)
    @torch.inference_mode()
    def get_img_embeds(
            self,
            img: Float[Tensor, "B 3 H W"],
    ) -> Tuple[Float[Tensor, "B 1 768"], Float[Tensor, "B 4 L L"]]:  # L = 32 when H = 256
        img = img * 2.0 - 1.0
        c_crossattn = self.model.get_learned_conditioning(img)
        c_concat = self.model.encode_first_stage(img).mode()
        return c_crossattn, c_concat

    def forward(self,
                rgb: Float[Tensor, "B H W C"],
                elevation: Float[Tensor, "B"],
                azimuth: Float[Tensor, "B"],
                # camera_distances: Float[Tensor, "B"]
                ):
        batch_size = rgb.shape[0]
        rgb_BCHW = rgb.permute(0, 3, 1, 2)
        rgb_BCHW_resized = F.interpolate(
            rgb_BCHW, self.img_dim, mode="bilinear", align_corners=False
        )
        latents: Float[Tensor, "B 4 L L"] = self.encode_images(rgb_BCHW_resized)

        cond = self.get_cond(elevation, azimuth)  # camera_distances)

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(
            self.min_step,
            self.max_step + 1,
            [batch_size],
            dtype=torch.long,
            device=self.device,
        )

        # predict the noise residual with unet, NO grad!
        with torch.inference_mode():
            # add noise
            noise = torch.randn_like(latents)  # TODO: use torch generator
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            x_in = torch.cat([latents_noisy] * 2)
            t_in = torch.cat([t] * 2)
            noise_pred = self.model.apply_model(x_in, t_in, cond)

        # perform guidance
        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + self.guidance_scale * (
                noise_pred_cond - noise_pred_uncond
        )

        w = (1 - self.alphas_cumprod[t]).reshape(-1, 1, 1, 1)
        grad = w * (noise_pred - noise)
        grad = torch.nan_to_num(grad)

        # # clip grad for stable training?
        # if self.grad_clip_val is not None:
        #     grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)

        # loss = SpecifyGradient.apply(latents, grad)
        # SpecifyGradient is not straightforward, use a reparameterization trick instead
        target = (latents - grad).detach()
        # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
        loss_sds = 0.5 * F.mse_loss(latents, target, reduction="sum") / batch_size

        return loss_sds

    # @torch.cuda.amp.autocast(enabled=False)
    def encode_images(
            self, imgs: Float[Tensor, "B 3 H W"]
    ) -> Float[Tensor, "B 4 32 32"]:
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        latents = self.model.get_first_stage_encoding(
            self.model.encode_first_stage(imgs)
        )
        return latents.to(input_dtype)  # [B, 4, 32, 32] Latent space image

    # @torch.cuda.amp.autocast(enabled=False)
    @torch.inference_mode()
    def get_cond(
            self,
            elevation_deg: Float[Tensor, "B"],
            azimuth_deg: Float[Tensor, "B"],
            # camera_distances: Float[Tensor, "B"],
            # c_crossattn=None,
            # c_concat=None,
            **kwargs,
    ) -> dict:
        T = torch.stack(
            [
                torch.deg2rad(
                    (90 - elevation_deg) - (90 - self.cond_elevation_deg)
                ),  # Zero123 polar is 90-elevation
                torch.sin(torch.deg2rad(azimuth_deg - self.cond_azimuth_deg)),
                torch.cos(torch.deg2rad(azimuth_deg - self.cond_azimuth_deg)),
                torch.deg2rad(
                    90 - torch.full_like(elevation_deg, self.cond_elevation_deg)
                ),
            ],
            dim=-1,
        )[:, None, :].to(self.device)
        cond = {}
        clip_emb = self.model.cc_projection(
            torch.cat([self.c_crossattn.repeat(len(T), 1, 1), T, ], dim=-1)
        )
        cond["c_crossattn"] = [
            torch.cat([torch.zeros_like(clip_emb), clip_emb], dim=0)
        ]
        cond["c_concat"] = [
            torch.cat(
                [
                    torch.zeros_like(self.c_concat).repeat(len(T), 1, 1, 1),
                    self.c_concat.repeat(len(T), 1, 1, 1),
                ],
                dim=0,
            )
        ]
        return cond

    def set_min_max_steps(self, min_step_percent: float, max_step_percent: float):
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)
