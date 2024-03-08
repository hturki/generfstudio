from __future__ import annotations

import random
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Literal

import math
import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Float
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils.rich_utils import CONSOLE

from generfstudio.diffusion.stable_zero123 import StableZero123
from generfstudio.generfstudio_constants import FG_MASK


@dataclass
class SDSModelBaseConfig(ModelConfig):
    guidance_scale: float = 3.0
    """guidance scale for sds loss"""
    diffusion_device: Optional[str] = None  # "cuda:1"
    """device for diffusion model"""

    diffusion_config: str = "/data/hturki/threestudio/load/zero123/sd-objaverse-finetune-c_concat-256.yaml"
    diffusion_checkpoint_path: str = "/data/hturki/threestudio/load/zero123/stable_zero123.ckpt"

    min_step_percent: Tuple[float, float] = (0.7, 0.3)
    max_step_percent: Tuple[float, float] = (0.98, 0.8)
    step_anneal_range: Tuple[int, int] = (50, 200)

    random_camera_strategy: Literal["orbit", "test_cameras"] = "orbit"
    random_elevation_range_deg: Tuple[int, int] = (-10, 80)
    random_azimuth_range_deg: Tuple[int, int] = (-180, 180)
    random_batch_uniform_azimuth: bool = False

    random_camera_dims: List[int] = field(default_factory=lambda: [64, 128, 256])
    random_camera_batch_sizes: List[int] = field(default_factory=lambda: [12, 8, 4])
    random_camera_resolution_milestones: List[int] = field(default_factory=lambda: [200, 300])

    sds_loss_mult: float = 1e-4
    """SDS loss multiplier."""

    mask_loss_mult: float = 0.1

    opacity_loss_mult: float = 5e-4


class SDSModelBase(Model):
    config: SDSModelBaseConfig

    def __init__(
            self,
            config: SDSModelBaseConfig,
            metadata: Dict[str, Any],
            **kwargs,
    ) -> None:
        self.diffusion_device = (
            torch.device(kwargs["device"]) if config.diffusion_device is None else torch.device(config.diffusion_device)
        )

        self.test_cameras = metadata["test_cameras"]
        if config.random_camera_strategy == "orbit":
            self.random_camera_distance = self.test_cameras.camera_to_worlds[:, :, 3].norm(dim=-1).mean()

        self.train_image_filenames = metadata["train_image_filenames"]
        self.train_image_cameras = metadata["train_image_cameras"]

        super().__init__(config=config, **kwargs)

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        # Need to prefix with underscore or else the nerfstudio viewer will throw an exception
        self._diffusion_model = StableZero123(self.config.diffusion_checkpoint_path, self.config.diffusion_config,
                                              self.config.guidance_scale).to(self.diffusion_device)

        c2w = self.train_image_cameras[-1].camera_to_worlds
        cam_dist = c2w[:, 3].norm()
        elevation_rad = torch.arcsin((c2w[2, 3] / cam_dist))
        azimuth_rad = torch.arctan2(c2w[1, 3], c2w[0, 3])
        elevation_deg = torch.rad2deg(elevation_rad)
        azimuth_deg = torch.rad2deg(azimuth_rad)

        CONSOLE.log(f"Cond image elevation {elevation_deg} azimuth {azimuth_deg}")
        self._diffusion_model.prepare_embeddings(self.train_image_filenames[-1], 5, 0)
                                                 # elevation_deg.item(),
                                                 # azimuth_deg.item())

        self.random_camera_dim = self.config.random_camera_dims[0]
        self.random_camera_batch_size = self.config.random_camera_batch_sizes[0]

    def get_training_callbacks(
            self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        def set_min_max_steps(step: int):
            weight = (step - self.config.step_anneal_range[0]) / (
                    self.config.step_anneal_range[1] - self.config.step_anneal_range[0])
            weight = min(max(weight, 0), 1)
            self._diffusion_model.set_min_max_steps((1 - weight) * self.config.min_step_percent[0] +
                                                    weight * self.config.min_step_percent[1],
                                                    (1 - weight) * self.config.max_step_percent[0] +
                                                    weight * self.config.max_step_percent[1])

        def update_random_camera_params(step: int):
            index = self.config.random_camera_resolution_milestones.index(step)
            self.random_camera_dim = self.config.random_camera_dims[index + 1]
            self.random_camera_batch_size = self.config.random_camera_batch_sizes[index + 1]
            CONSOLE.log(
                f"New camera resolution {self.random_camera_dim} and batch size {self.random_camera_batch_size} at step {step}")

        return [
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                func=set_min_max_steps,
                update_every_num_iters=1,
            ),
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                iters=tuple(self.config.random_camera_resolution_milestones),
                func=update_random_camera_params,
            ),
        ]

    @abstractmethod
    def get_random_camera_outputs(self, random_cameras: Cameras) -> Dict[str, torch.Tensor]:
        pass

    def get_sds_loss_dict(self, outputs, batch):
        loss_dict = {}

        if self.config.random_camera_strategy == "orbit":
            # sample elevation angles
            elevation_deg: torch.Tensor
            elevation_rad: torch.Tensor
            if random.random() < 0.5:
                # sample elevation angles uniformly with a probability 0.5 (biased towards poles)
                elevation_deg = (
                        torch.rand(self.random_camera_batch_size)
                        * (self.config.random_elevation_range_deg[1] - self.config.random_elevation_range_deg[0])
                        + self.config.random_elevation_range_deg[0]
                )
                elevation_rad = elevation_deg * math.pi / 180
            else:
                # otherwise sample uniformly on sphere
                elevation_range_percent = [
                    self.config.random_elevation_range_deg[0] / 180.0 * math.pi,
                    self.config.random_elevation_range_deg[1] / 180.0 * math.pi,
                ]
                # inverse transform sampling
                elevation_rad = torch.asin(
                    (
                            torch.rand(self.random_camera_batch_size)
                            * (
                                    math.sin(elevation_range_percent[1])
                                    - math.sin(elevation_range_percent[0])
                            )
                            + math.sin(elevation_range_percent[0])
                    )
                )

            # sample azimuth angles from a uniform distribution bounded by azimuth_range
            azimuth_deg: torch.Tensor
            if self.config.random_batch_uniform_azimuth:
                # ensures sampled azimuth angles in a batch cover the whole range
                azimuth_deg = (torch.rand(self.random_camera_batch_size) + torch.arange(self.random_camera_batch_size)) \
                              / self.random_camera_batch_size * (
                                      self.config.random_azimuth_range_deg[1] - self.config.random_azimuth_range_deg[0]
                              ) + self.config.random_azimuth_range_deg[0]
            else:
                # simple random sampling
                azimuth_deg = (
                        torch.rand(self.random_camera_batch_size)
                        * (self.config.random_azimuth_range_deg[1] - self.config.random_azimuth_range_deg[0])
                        + self.config.random_azimuth_range_deg[0]
                )

            azimuth_rad = azimuth_deg * math.pi / 180

            # convert spherical coordinates to cartesian coordinates
            # right hand coordinate system, x back, y right, z up
            # elevation in (-90, 90), azimuth from +x to +y in (-180, 180)
            camera_positions = torch.stack(
                [
                    self.random_camera_distance * torch.cos(elevation_rad) * torch.cos(azimuth_rad),
                    self.random_camera_distance * torch.cos(elevation_rad) * torch.sin(azimuth_rad),
                    self.random_camera_distance * torch.sin(elevation_rad),
                ],
                dim=-1,
            )

            # default scene center at origin
            center = torch.zeros_like(camera_positions)
            # default camera up direction as +z
            up = torch.as_tensor([0, 0, 1], dtype=torch.float32)[None, :].repeat(self.random_camera_batch_size, 1)

            lookat = F.normalize(center - camera_positions, dim=-1)
            right = F.normalize(torch.cross(lookat, up), dim=-1)
            up = F.normalize(torch.cross(right, lookat), dim=-1)
            c2ws = torch.cat([torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]], dim=-1)

            random_cameras = Cameras(
                camera_to_worlds=c2ws,
                fx=self.test_cameras[0].fx.item(),
                fy=self.test_cameras[0].fy.item(),
                cx=self.test_cameras[0].cx.item(),
                cy=self.test_cameras[0].cy.item(),
                width=self.test_cameras[0].width.item(),
                height=self.test_cameras[0].height.item(),
            ).to(self.device)
        else:
            # Take random camera test indices
            if len(self.test_cameras) <= self.random_camera_batch_size:
                random_cameras = self.test_cameras
            else:
                random_indices = np.random.choice(len(self.test_cameras), self.random_camera_batch_size, replace=False)
                random_cameras = self.test_cameras[torch.LongTensor(random_indices)].to(self.device)

        # TODO: Might need adjustment for non-square cameras
        random_cameras.rescale_output_resolution(scaling_factor=self.random_camera_dim / random_cameras.width)

        sds_outputs = self.get_random_camera_outputs(random_cameras)
        rgb = sds_outputs["rgb"]

        c2w = random_cameras.camera_to_worlds
        cam_dist = c2w[:, :, 3].norm(dim=-1)
        elevation_rad = torch.arcsin((c2w[:, 2, 3] / cam_dist))
        azimuth_rad = torch.arctan2(c2w[:, 1, 3], c2w[:, 0, 3])

        loss_dict["sds_loss"] = self.config.sds_loss_mult * self._diffusion_model(rgb, torch.rad2deg(elevation_rad),
                                                                                  torch.rad2deg(azimuth_rad))
        opacity = sds_outputs["accumulation"]
        rev_opacity = 1 - opacity
        loss_dict["opacity_loss"] = self.config.opacity_loss_mult * -(
                opacity * torch.log(opacity + 1e-5) + rev_opacity * torch.log(rev_opacity + 1e-5)).mean()

        if FG_MASK in batch:
            loss_dict["fg_mask_loss"] = self.config.mask_loss_mult * F.mse_loss(outputs["accumulation"],
                                                                                batch[FG_MASK].to(self.device))

        return loss_dict
