from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Type, Optional

from PIL import Image
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
            return DataparserOutputs(
                image_filenames=train_image_filenames,
                cameras=train_image_cameras,
                alpha_color=inner_outputs.alpha_color,
                scene_box=inner_outputs.scene_box,
                dataparser_scale=inner_outputs.dataparser_scale,
                metadata={
                    "train_image_filenames": train_image_filenames,
                    "train_image_cameras": train_image_cameras,
                    "test_cameras": self.inner.get_dataparser_outputs("test").cameras
                }
            )
        else:
            return inner_outputs
