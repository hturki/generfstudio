from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Type, Optional, Dict

import imageio
import numpy as np
import torch
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

def pad_poses(p):
  """Pad [..., 3, 4] pose matrices with a homogeneous bottom row [0,0,0,1]."""
  bottom = np.broadcast_to([0, 0, 0, 1.0], p[Ellipsis, :1, :4].shape)
  return np.concatenate([p[Ellipsis, :3, :4], bottom], axis=-2)


def unpad_poses(p):
  """Remove the homogeneous bottom row from [..., 4, 4] pose matrices."""
  return p[Ellipsis, :3, :4]

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
    def transform_poses_pca(cls, poses):
        """Transforms poses so principal components lie on XYZ axes.
  
        Args:
          poses: a (N, 3, 4) array containing the cameras' camera to world transforms.
  
        Returns:
          A tuple (poses, transform), with the transformed poses and the applied
          camera_to_world transforms.
        """
        t = poses[:, :3, 3]
        t_mean = t.mean(axis=0)
        t = t - t_mean

        eigval, eigvec = np.linalg.eig(t.T @ t)
        # Sort eigenvectors in order of largest to smallest eigenvalue.
        inds = np.argsort(eigval)[::-1]
        eigvec = eigvec[:, inds]
        rot = eigvec.T
        if np.linalg.det(rot) < 0:
            rot = np.diag(np.array([1, 1, -1])) @ rot

        transform = np.concatenate([rot, rot @ -t_mean[:, None]], -1)
        poses_recentered = unpad_poses(transform @ pad_poses(poses))
        transform = np.concatenate([transform, np.eye(4)[3:]], axis=0)

        # Flip coordinate system if z component of y-axis is negative
        if poses_recentered.mean(axis=0)[2, 1] < 0:
            poses_recentered = np.diag(np.array([1, -1, -1])) @ poses_recentered
            transform = np.diag(np.array([1, -1, -1, 1])) @ transform

        # Just make sure it's it in the [-1, 1]^3 cube
        scale_factor = 1.0 / np.max(np.abs(poses_recentered[:, :3, 3]))
        poses_recentered[:, :3, 3] *= scale_factor
        transform = np.diag(np.array([scale_factor] * 3 + [1])) @ transform

        return poses_recentered, transform

    def get_dataparser_outputs(self, split: str = "train", **kwargs: Optional[Dict]) -> DataparserOutputs:
        fx = []
        fy = []
        cx = []
        cy = []
        # c2ws = []
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

        poses, _ = self.transform_poses_pca(poses[..., :4])
        c2ws = torch.FloatTensor(poses)

        min_bounds = c2ws[:, :, 3].min(dim=0)[0]
        max_bounds = c2ws[:, :, 3].max(dim=0)[0]

        # origin = (max_bounds + min_bounds) * 0.5
        # CONSOLE.log('Calculated origin: {} {} {}'.format(origin, min_bounds, max_bounds))
        #
        # pose_scale_factor = ((max_bounds - min_bounds) * 0.5).norm().item()
        # CONSOLE.log('Calculated pose scale factor: {}'.format(pose_scale_factor))
        #
        # for c2w in c2ws:
        #     c2w[:, 3] = (c2w[:, 3] - origin) / pose_scale_factor
        #     assert torch.logical_and(c2w >= -1, c2w <= 1).all(), c2w

        # in x,y,z order
        scene_box = SceneBox(aabb=torch.stack([min_bounds, max_bounds]))
        # scene_box = SceneBox(aabb=((torch.stack([min_bounds, max_bounds]) - origin) / pose_scale_factor).float())

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
            # metadata={
            #     'pose_scale_factor': pose_scale_factor,
            # }
        )

        return dataparser_outputs
