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

    auto_orient: bool = True


@dataclass
class DTU(DataParser):
    """DTU Dataset"""

    config: DTUDataParserConfig

    def _generate_dataparser_outputs(self, split="train"):

        if self.config.scene_id is not None:
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
        for scene in scenes:
            scene_image_filenames = sorted(list((self.config.data / scene / "image").iterdir()))
            image_filenames += scene_image_filenames

            all_cam = np.load(str(self.config.data / scene / 'cameras.npz'))
            for scene_image_filename in scene_image_filenames:
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

                c2ws.append(pose)

                fx.append(K[0, 0])
                fy.append(K[1, 1])
                cx.append(K[0, 2])
                cy.append(K[1, 2])

        c2ws = torch.FloatTensor(c2ws)
        fx = torch.FloatTensor(fx)
        fy = torch.FloatTensor(fy)
        cx = torch.FloatTensor(cx)
        cy = torch.FloatTensor(cy)

        if self.config.auto_orient:
            c2ws, transform = camera_utils.auto_orient_and_center_poses(
                c2ws,
                method="up",
                center_method="none",
            )

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

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
        )
        return dataparser_outputs
