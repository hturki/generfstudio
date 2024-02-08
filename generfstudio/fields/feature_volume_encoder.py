"""
Generates the feature volume
"""
from enum import Enum, auto
from typing import Dict, NamedTuple

import torch
from rich.console import Console
from torch import nn

from generfstudio.fields.image_encoder import ImageEncoder
from generfstudio.fields.spatial_encoder import SpatialEncoder

CONSOLE = Console(width=120)


class OutputInterpolation(Enum):
    COLOR = auto()
    EMBEDDING = auto()


class LevelInterpolation(Enum):
    NONE = auto()
    LINEAR = auto()


class FeatureVolumeEncoder(nn.Module):

    def __init__(
            self,
            use_global_encoder: bool = False,
    ) -> None:
        super().__init__()

        self.encoder = SpatialEncoder()
        self.global_encoder = ImageEncoder() if use_global_encoder else None

    def forward(self, images, poses, focal, c=None) -> Dict:
        """
        :param images (B, NS, 3, H, W)
        B is the batch size
        NS is number of input (aka source or reference) views
        :param poses (B, NS, 4, 4)
        :param focal focal length () or (2) or (NS) or (NS, 2) [fx, fy]
        :param c principal point None or () or (2) or (NS) or (NS, 2) [cx, cy],
        default is center of image
        """
        self.num_objs = images.shape[0]
        self.num_views_per_obj = images.shape[1]
        assert len(poses.shape) == 4
        assert poses.shape[1] == images.shape[1]  # Be consistent with NS = num input views

        # if len(images.shape) == 5:
        #     assert len(poses.shape) == 4
        #     assert poses.size(1) == images.size(
        #         1
        #     )  # Be consistent with NS = num input views
        #     self.num_views_per_obj = images.size(1)
        #     images = images.reshape(-1, *images.shape[2:])
        #     poses = poses.reshape(-1, 4, 4)
        # else:
        #     self.num_views_per_obj = 1

        self.encoder(images)
        rot = poses[:, :3, :3].transpose(1, 2)  # (B, 3, 3)
        trans = -torch.bmm(rot, poses[:, :3, 3:])  # (B, 3, 1)
        self.poses = torch.cat((rot, trans), dim=-1)  # (B, 3, 4)

        self.image_shape = torch.FloatTensor([images.shape[-1], images.shape[-2]])

        # Handle various focal length/principal point formats
        if len(focal.shape) == 0:
            # Scalar: fx = fy = value for all views
            focal = focal[None, None].repeat((1, 2))
        elif len(focal.shape) == 1:
            # Vector f: fx = fy = f_i *for view i*
            # Length should match NS (or 1 for broadcast)
            focal = focal.unsqueeze(-1).repeat((1, 2))
        else:
            focal = focal.clone()

        self.focal = focal.float()
        self.focal[..., 1] *= -1.0

        if c is None:
            # Default principal point is center of image
            c = (self.image_shape * 0.5).unsqueeze(0)
        elif len(c.shape) == 0:
            # Scalar: cx = cy = value for all views
            c = c[None, None].repeat((1, 2))
        elif len(c.shape) == 1:
            # Vector c: cx = cy = c_i *for view i*
            c = c.unsqueeze(-1).repeat((1, 2))
        self.c = c

        if self.use_global_encoder:
            self.global_encoder(images)

        return

