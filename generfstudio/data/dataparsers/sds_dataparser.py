from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Type

from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig, DataparserOutputs,
)
from nerfstudio.data.dataparsers.blender_dataparser import BlenderDataParserConfig

from generfstudio.data.dataparsers.dtu_dataparser import DTUDataParserConfig


@dataclass
class SDSDataParserConfig(DataParserConfig):
    _target: Type = field(default_factory=lambda: SDS)
    """target class to instantiate"""
    inner: DataParserConfig = field(default_factory=lambda: DTUDataParserConfig())
    """inner dataparser"""

    start: int = 2
    end: int = 2

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
            train_image_filenames = inner_outputs.image_filenames[self.start:self.end+1]
            train_image_cameras = inner_outputs.cameras[self.start:self.end+1]
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
