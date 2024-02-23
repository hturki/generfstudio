from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Type, Optional

import cv2
import numpy as np
import torch
from PIL import Image
from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import DataParser, DataParserConfig, DataparserOutputs
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.data.utils.dataparsers_utils import get_train_eval_split_fraction

from generfstudio.generfstudio_constants import NEIGHBORING_VIEW_INDICES, NEAR, FAR, DEFAULT_SCENE_METADATA

# OpenCV to OpenGL
COORD_TRANS_WORLD = np.array(
    [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
)

COORD_TRANS_CAM = np.array(
    [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
)


@dataclass
class DTUDataParserConfig(DataParserConfig):
    """Scene dataset parser config. Assumes Pixel-NeRF format"""

    _target: Type = field(default_factory=lambda: DTU)
    """target class to instantiate"""
    data: Path = Path("data/DTU")
    """Directory specifying location of data."""

    scene_id: Optional[str] = "scan65"

    train_split_fraction: float = 0.9
    """The percentage of the dataset to use for training. Only used when using a single scene."""

    auto_orient: bool = False

    default_scene_id: str = "scan8"


@dataclass
class DTU(DataParser):
    """DTU Dataset"""

    config: DTUDataParserConfig

    def _generate_dataparser_outputs(self, split="train", get_default_scene=False):

        if get_default_scene:
            scenes = [self.config.default_scene_id]
        elif self.config.scene_id is not None:
            scenes = [self.config.scene_id]
        else:
            with (self.config.data / f"new_{split}.lst").open() as f:
                scenes = [line.strip() for line in f]

        image_filenames = []
        c2ws = []
        fx = []
        fy = []
        cx = []
        cy = []
        neighboring_views = []
        for scene in scenes:
            scene_image_filenames = sorted(list((self.config.data / scene / "image").iterdir()))
            scene_index_start = len(image_filenames)
            scene_index_end = scene_index_start + len(scene_image_filenames)
            image_filenames += scene_image_filenames

            all_cam = np.load(str(self.config.data / scene / 'cameras.npz'))
            scene_poses = []
            for scene_image_index, scene_image_filename in enumerate(scene_image_filenames):
                index = int(scene_image_filename.stem)
                P = all_cam["world_mat_" + str(index)][:3]
                K, R, t = cv2.decomposeProjectionMatrix(P)[:3]
                K = K / K[2, 2]
                pose = np.eye(4, dtype=np.float32)
                pose[:3, :3] = R.transpose()
                pose[:3, 3] = (t[:3] / t[3])[:, 0]

                scale_mtx = all_cam.get("scale_mat_" + str(index))
                if scale_mtx is not None:
                    norm_trans = scale_mtx[:3, 3:]
                    norm_scale = np.diagonal(scale_mtx[:3, :3])[..., None]

                    pose[:3, 3:] -= norm_trans
                    pose[:3, 3:] /= norm_scale

                pose = (
                        COORD_TRANS_WORLD
                        @ pose
                        @ COORD_TRANS_CAM
                )

                neighboring_views.append(
                    list(range(scene_index_start, scene_index_start + scene_image_index))
                    + list(range(scene_index_start + scene_image_index + 1, scene_index_end)))

                scene_poses.append(pose)

                fx.append(K[0, 0])
                fy.append(K[1, 1])
                cx.append(K[0, 2])
                cy.append(K[1, 2])

            scene_poses = torch.FloatTensor(scene_poses)
            if self.config.auto_orient:
                scene_poses, transform = camera_utils.auto_orient_and_center_poses(
                    scene_poses,
                    method="up",
                    center_method="none",
                )
            c2ws.append(scene_poses)

        c2ws = torch.cat(c2ws)
        fx = torch.FloatTensor(fx)
        fy = torch.FloatTensor(fy)
        cx = torch.FloatTensor(cx)
        cy = torch.FloatTensor(cy)

        min_bounds = c2ws[:, :3, 3].min(dim=0)[0]
        max_bounds = c2ws[:, :3, 3].max(dim=0)[0]
        scene_box = SceneBox(aabb=torch.stack([min_bounds, max_bounds]))

        if self.config.scene_id is not None:
            i_train, i_eval = get_train_eval_split_fraction(image_filenames, self.config.train_split_fraction)
            if split == "train":
                indices = i_train
            elif split in ["val", "test"]:
                indices = i_eval
            else:
                raise ValueError(f"Unknown dataparser split {split}")

            image_filenames = [image_filenames[i] for i in indices]

            #  Need to deal with indices changing post filtering if we actually want this
            neighboring_views = None  # [neighboring_views[i] for i in indices]

            idx_tensor = torch.LongTensor(indices)
            c2ws = c2ws[idx_tensor]
            fx = fx[idx_tensor]
            fy = fy[idx_tensor]
            cx = cx[idx_tensor]
            cy = cy[idx_tensor]

        image_dims = [Image.open(x).size for x in image_filenames]

        cameras = Cameras(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            height=torch.LongTensor([x[1] for x in image_dims]),
            width=torch.LongTensor([x[0] for x in image_dims]),
            camera_to_worlds=c2ws[:, :3, :4],
            camera_type=CameraType.PERSPECTIVE,
        )

        metadata = {
            NEAR: 0.1,
            FAR: 5,
        }

        if neighboring_views is not None:
            metadata[NEIGHBORING_VIEW_INDICES] = neighboring_views

        if (not get_default_scene) and self.config.scene_id is None:
            metadata[DEFAULT_SCENE_METADATA] = self._generate_dataparser_outputs(split, get_default_scene=True)

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            metadata=metadata
        )

        return dataparser_outputs
