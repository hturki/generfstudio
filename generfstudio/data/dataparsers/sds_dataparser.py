from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Type, Optional

import numpy as np
import torch
from PIL import Image
from diffusers import DiffusionPipeline
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig, DataparserOutputs,
)
from nerfstudio.data.dataparsers.blender_dataparser import BlenderDataParserConfig


@dataclass
class SDSDataParserConfig(DataParserConfig):
    _target: Type = field(default_factory=lambda: SDS)
    """target class to instantiate"""
    inner: DataParserConfig = field(default_factory=lambda: BlenderDataParserConfig())
    """inner dataparser"""

    start: int = 2
    end: int = 2
    image_cond_override: Optional[Path] = None

    init_with_depth: bool = True


@dataclass
class SDS(DataParser):
    config: SDSDataParserConfig

    def __init__(self, config: SDSDataParserConfig):
        super().__init__(config=config)
        if config.data != Path():
            config.inner.data = self.config.data
        self.inner: DataParser = config.inner.setup()
        self.start = config.start
        self.end = config.end

    def _generate_dataparser_outputs(self, split='train') -> DataparserOutputs:
        inner_outputs = self.inner.get_dataparser_outputs(split)
        if split == 'train':
            train_image_filenames = inner_outputs.image_filenames[self.start:self.end + 1]
            train_image_cameras = inner_outputs.cameras[self.start:self.end + 1]
            image_cond_override = self.config.image_cond_override

            if image_cond_override is not None:
                train_image_filenames.pop()
                resized = image_cond_override.parent / f"{image_cond_override.stem}-resized{image_cond_override.suffix}"
                if not resized.exists():
                    Image.open(image_cond_override).resize(
                        (train_image_cameras.width[-1], train_image_cameras.height[-1]), Image.LANCZOS).save(resized)
                train_image_filenames.append(resized)

            metadata = {
                "train_image_filenames": train_image_filenames,
                "train_image_cameras": train_image_cameras,
                "test_cameras": self.inner.get_dataparser_outputs("test").cameras
            }

            if self.config.init_with_depth:
                estimator = DiffusionPipeline.from_pretrained(
                    "Bingxin/Marigold",
                    custom_pipeline="marigold_depth_estimation",
                    torch_dtype=torch.float16,
                ).to("cuda")
                points_rgb = []
                points_xyz = []
                for image_path, camera in zip(train_image_filenames, train_image_cameras):
                    image = Image.open(image_path)
                    rgb_vals = torch.ByteTensor(np.asarray(image))
                    if rgb_vals.shape[-1] == 4:
                        mask = torch.BoolTensor(rgb_vals[..., 3] > 0).view(-1)
                        rgb_vals = rgb_vals[..., :3]
                    else:
                        mask = torch.ones_like(rgb_vals[..., :1], dtype=torch.bool).view(-1)

                    points_rgb.append(rgb_vals.view(-1, 3)[mask])

                    pipeline_output = estimator(image)
                    depth = torch.FloatTensor(pipeline_output.depth_np).view(-1, 1)[mask]
                    rays = camera.generate_rays(camera_indices=0)
                    image_points = camera.camera_to_worlds[:, 3].unsqueeze(0) + rays.directions.view(-1, 3)[
                        mask] * depth * rays.metadata["directions_norm"].view(-1, 1)[mask]
                    points_xyz.append(image_points)

                metadata["points3D_rgb"] = torch.cat(points_rgb)
                metadata["points3D_xyz"] = torch.cat(points_xyz)

            return DataparserOutputs(
                image_filenames=train_image_filenames,
                cameras=train_image_cameras,
                alpha_color=inner_outputs.alpha_color,
                scene_box=inner_outputs.scene_box,
                dataparser_scale=inner_outputs.dataparser_scale,
                metadata=metadata
            )
        else:
            return inner_outputs
