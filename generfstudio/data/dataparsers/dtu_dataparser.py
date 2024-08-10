from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Type, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import DataParser, DataParserConfig, DataparserOutputs
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.data.utils.dataparsers_utils import get_train_eval_split_fraction
from nerfstudio.utils.rich_utils import CONSOLE

from generfstudio.generfstudio_constants import NEIGHBOR_INDICES, NEAR, FAR, DEFAULT_SCENE_METADATA
from generfstudio.generfstudio_utils import central_crop_v2

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
    data: Path = Path("data/DTU/scan8")
    """Directory specifying location of data."""

    scene_id: Optional[str] = "scan65"

    train_images: int = 3


@dataclass
class DTU(DataParser):
    """DTU Dataset"""

    config: DTUDataParserConfig

    def _generate_dataparser_outputs(self, split="train", get_default_scene=False):

        image_filenames = []
        c2ws = []
        fx = []
        fy = []
        cx = []
        cy = []

        scene_image_filenames = sorted(list((self.config.data / "image").iterdir()))
        all_cam = np.load(str(self.config.data / "cameras.npz"))
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

            image_filenames.append(scene_image_filename)
            c2ws.append(torch.FloatTensor(pose))

            fx.append(K[0, 0])
            fy.append(K[1, 1])
            cx.append(K[0, 2])
            cy.append(K[1, 2])


        c2ws = torch.stack(c2ws)
        fx = torch.FloatTensor(fx)
        fy = torch.FloatTensor(fy)
        cx = torch.FloatTensor(cx)
        cy = torch.FloatTensor(cy)

        min_bounds = c2ws[:, :3, 3].min(dim=0)[0]
        max_bounds = c2ws[:, :3, 3].max(dim=0)[0]
        scene_box = SceneBox(aabb=torch.stack([min_bounds, max_bounds]))

        train_indices = [25, 22, 28, 40, 44, 48, 0, 8, 13]
        if split == "train":
            indices =  torch.LongTensor(train_indices)[:self.config.train_images]
        else:
            exclude_test_indices = set(train_indices + [3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 36, 37, 38, 39])
            indices = []
            for i in range(c2ws.shape[0]):
                if i not in exclude_test_indices:
                    indices.append(i)
            indices = torch.LongTensor(indices)

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
