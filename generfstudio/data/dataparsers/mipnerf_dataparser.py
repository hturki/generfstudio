from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Type, List, Optional, Dict

import imageio
import numpy as np
import torch
from PIL import Image
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
)
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.data.utils.colmap_parsing_utils import read_cameras_binary
from rich.console import Console

CONSOLE = Console(width=120)


@dataclass
class MipNerf360DataParserConfig(DataParserConfig):
    """Mipnerf 360 dataset parser config"""

    _target: Type = field(default_factory=lambda: Mipnerf360)
    """target class to instantiate"""
    data: Path = Path("data/mipnerf360/garden")

    """Directory specifying location of data."""
    auto_scale: bool = True

    """Scale based on pose bounds."""
    aabb_scale: float = 1

    """Scene scale."""
    train_images: int = 2


@dataclass
class Mipnerf360(DataParser):
    """MipNeRF 360 Dataset"""

    config: MipNerf360DataParserConfig

    @classmethod
    def normalize_orientation(cls, poses: np.ndarray):
        """Set the _up_ direction to be in the positive Y direction.
        Args:
            poses: Numpy array of poses.
        """
        poses_orig = poses.copy()
        bottom = np.reshape([0, 0, 0, 1.0], [1, 4])
        center = poses[:, :3, 3].mean(0)
        vec2 = poses[:, :3, 2].sum(0) / np.linalg.norm(poses[:, :3, 2].sum(0))
        up = poses[:, :3, 1].sum(0)
        vec0 = np.cross(up, vec2) / np.linalg.norm(np.cross(up, vec2))
        vec1 = np.cross(vec2, vec0) / np.linalg.norm(np.cross(vec2, vec0))
        c2w = np.stack([vec0, vec1, vec2, center], -1)  # [3, 4]
        c2w = np.concatenate([c2w[:3, :4], bottom], -2)  # [4, 4]
        bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])  # [BS, 1, 4]
        poses = np.concatenate([poses[:, :3, :4], bottom], -2)  # [BS, 4, 4]
        poses = np.linalg.inv(c2w) @ poses
        poses_orig[:, :3, :4] = poses[:, :3, :4]
        return poses_orig

    def get_dataparser_outputs(self, split: str = "train", **kwargs: Optional[Dict]) -> DataparserOutputs:
        fx = []
        fy = []
        cx = []
        cy = []
        c2ws = []
        width = []
        height = []
        image_filenames = []

        camera_params = read_cameras_binary(self.config.data / 'sparse/0/cameras.bin')
        assert camera_params[1].model == 'PINHOLE'
        camera_fx, camera_fy, camera_cx, camera_cy = camera_params[1].params

        image_dir = self.config.data / "images"
        if not image_dir.exists():
            raise ValueError(f"Image directory {image_dir} doesn't exist")

        valid_formats = ['.jpg', '.png']
        num_images = 0
        for f in sorted(image_dir.iterdir()):
            ext = f.suffix
            if ext.lower() not in valid_formats:
                continue
            image_filenames.append(f)
            num_images += 1

        poses_data = np.load(self.config.data / 'poses_bounds.npy')
        poses = poses_data[:, :-2].reshape([-1, 3, 5]).astype(np.float32)
        bounds = poses_data[:, -2:].transpose([1, 0])

        if num_images != poses.shape[0]:
            raise RuntimeError(f'Different number of images ({num_images}), and poses ({poses.shape[0]})')

        img_0 = imageio.imread(image_filenames[-1])
        image_height, image_width = img_0.shape[:2]

        width.append(torch.full((num_images, 1), image_width, dtype=torch.long))
        height.append(torch.full((num_images, 1), image_height, dtype=torch.long))
        fx.append(torch.full((num_images, 1), camera_fx))
        fy.append(torch.full((num_images, 1), camera_fy))
        cx.append(torch.full((num_images, 1), camera_cx))
        cy.append(torch.full((num_images, 1), camera_cy))

        # Reorder pose to match nerfstudio convention
        poses = np.concatenate([poses[:, :, 1:2], -poses[:, :, 0:1], poses[:, :, 2:]], axis=-1)

        # Center poses and rotate. (Compute up from average of all poses)
        poses = self.normalize_orientation(poses)

        # Scale factor used in mipnerf
        if self.config.auto_scale:
            scale_factor = 1 / (np.min(bounds) * 0.75)
            poses[:, :3, 3] *= scale_factor
            bounds *= scale_factor

        # Center poses
        poses[:, :3, 3] = poses[:, :3, 3] - np.mean(poses[:, :3, :], axis=0)[:, 3]
        c2ws.append(torch.from_numpy(poses[:, :3, :4]))

        c2ws = torch.cat(c2ws)
        min_bounds = c2ws[:, :, 3].min(dim=0)[0]
        max_bounds = c2ws[:, :, 3].max(dim=0)[0]

        origin = (max_bounds + min_bounds) * 0.5
        CONSOLE.log('Calculated origin: {} {} {}'.format(origin, min_bounds, max_bounds))

        pose_scale_factor = ((max_bounds - min_bounds) * 0.5).norm().item()
        CONSOLE.log('Calculated pose scale factor: {}'.format(pose_scale_factor))

        for c2w in c2ws:
            c2w[:, 3] = (c2w[:, 3] - origin) / pose_scale_factor
            assert torch.logical_and(c2w >= -1, c2w <= 1).all(), c2w

        # in x,y,z order
        scene_box = SceneBox(aabb=((torch.stack([min_bounds, max_bounds]) - origin) / pose_scale_factor).float())

        train_indices = np.linspace(0, num_images, self.config.train_images, endpoint=False, dtype=np.int32)

        if split.casefold() == 'train':
            indices = torch.LongTensor(train_indices)
        else:
            val_indices = []
            train_indices = set(train_indices)
            for i in range(len(image_filenames)):
                if i not in train_indices:
                    val_indices.append(i)

            indices = torch.LongTensor(val_indices)

        cameras = Cameras(
            camera_to_worlds=c2ws[indices].float(),
            fx=torch.cat(fx)[indices],
            fy=torch.cat(fy)[indices],
            cx=torch.cat(cx)[indices],
            cy=torch.cat(cy)[indices],
            width=torch.cat(width)[indices],
            height=torch.cat(height)[indices],
            camera_type=CameraType.PERSPECTIVE,
        )

        CONSOLE.log('Num images in split {}: {}'.format(split, len(indices)))

        dataparser_outputs = DataparserOutputs(
            image_filenames=[image_filenames[i] for i in indices],
            cameras=cameras,
            scene_box=scene_box,
            dataparser_scale=1,
            metadata={
                'pose_scale_factor': pose_scale_factor,
            }
        )

        return dataparser_outputs
