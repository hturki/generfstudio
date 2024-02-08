from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Type

import torch

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import DataParser, DataParserConfig, DataparserOutputs
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.io import load_from_json


@dataclass
class DTUDataParserConfig(DataParserConfig):
    """Scene dataset parser config"""

    _target: Type = field(default_factory=lambda: DTU)
    """target class to instantiate"""
    data: Path = Path("data/DTU")
    """Directory specifying location of data."""
    """sub sampling validation images"""
    auto_orient: bool = True


@dataclass
class DTU(DataParser):
    """DTU Dataset"""

    config: DTUDataParserConfig

    def _generate_dataparser_outputs(self, split="train"):

        # load meta data
        meta = load_from_json(self.config.data / "meta_data.json")

        indices = list(range(len(meta["frames"])))
        # subsample to avoid out-of-memory for validation set
        if split != "train" and self.config.skip_every_for_val_split >= 1:
            indices = indices[:: self.config.skip_every_for_val_split]

        image_filenames = []
        depth_filenames = []
        normal_filenames = []
        transform = None
        fx = []
        fy = []
        cx = []
        cy = []
        camera_to_worlds = []
        for i, frame in enumerate(meta["frames"]):
            if i not in indices:
                continue

            image_filename = self.config.data / frame["rgb_path"]
            depth_filename = frame.get("mono_depth_path")
            normal_filename = frame.get("mono_normal_path")

            intrinsics = torch.tensor(frame["intrinsics"])
            camtoworld = torch.tensor(frame["camtoworld"])

            # append data
            image_filenames.append(image_filename)
            if depth_filename is not None and normal_filename is not None:
                depth_filenames.append(self.config.data / depth_filename)
                normal_filenames.append(self.config.data / normal_filename)
            fx.append(intrinsics[0, 0])
            fy.append(intrinsics[1, 1])
            cx.append(intrinsics[0, 2])
            cy.append(intrinsics[1, 2])
            camera_to_worlds.append(camtoworld)

        fx = torch.stack(fx)
        fy = torch.stack(fy)
        cx = torch.stack(cx)
        cy = torch.stack(cy)
        c2w_colmap = torch.stack(camera_to_worlds)
        camera_to_worlds = torch.stack(camera_to_worlds)

        # Convert from COLMAP's/OPENCV's camera coordinate system to nerfstudio
        camera_to_worlds[:, 0:3, 1:3] *= -1

        if self.config.auto_orient:
            camera_to_worlds, transform = camera_utils.auto_orient_and_center_poses(
                camera_to_worlds,
                method="up",
                center_method="none",
            )

        # scene box from meta data
        meta_scene_box = meta["scene_box"]
        aabb = torch.tensor(meta_scene_box["aabb"], dtype=torch.float32)
        scene_box = SceneBox(
            aabb=aabb,
        )

        height, width = meta["height"], meta["width"]
        cameras = Cameras(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            height=height,
            width=width,
            camera_to_worlds=camera_to_worlds[:, :3, :4],
            camera_type=CameraType.PERSPECTIVE,
        )

        # TODO supports downsample
        # cameras.rescale_output_resolution(scaling_factor=1.0 / self.config.downscale_factor)
        if self.config.include_mono_prior:
            assert meta["has_mono_prior"], f"no mono prior in {self.config.data}"

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            metadata={
                "depth_filenames": depth_filenames if len(depth_filenames) > 0 else None,
                "normal_filenames": normal_filenames if len(normal_filenames) > 0 else None,
                "transform": transform,
                # required for normal maps, these are in colmap format so they require c2w before conversion
                "camera_to_worlds": c2w_colmap if len(c2w_colmap) > 0 else None,
                "include_mono_prior": self.config.include_mono_prior,
                "depth_unit_scale_factor": self.config.depth_unit_scale_factor,
            },
        )
        return dataparser_outputs
