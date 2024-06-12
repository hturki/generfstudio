import os
from copy import deepcopy
from typing import Dict

import numpy as np
import torch
from dust3r.utils.geometry import geotrf
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.datasets.base_dataset import InputDataset

from generfstudio.fields.batched_pc_optimizer import fast_depthmap_to_pts3d
from generfstudio.generfstudio_constants import NEIGHBOR_INDICES, NEIGHBOR_IMAGES, DEPTH, PTS3D, NEIGHBOR_DEPTH

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2


class NeighboringViewsDataset(InputDataset):
    def __init__(self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1.0,
                 neighboring_views_size: int = 3, return_neighbor_points: bool = True):
        # super().__init__(dataparser_outputs, scale_factor)
        # Skip the deepcopy to save time
        self._dataparser_outputs = dataparser_outputs
        self.scale_factor = scale_factor
        self.scene_box = dataparser_outputs.scene_box
        self.metadata = dataparser_outputs.metadata
        # self.cameras = dataparser_outputs.cameras
        # self.cameras.rescale_output_resolution(scaling_factor=scale_factor)
        self.mask_color = dataparser_outputs.metadata.get("mask_color", None)

        self.neighboring_views_size = neighboring_views_size
        self.return_neighbor_points = return_neighbor_points

        if DEPTH in self.metadata:
            # self.cameras_depth = deepcopy(dataparser_outputs.cameras)
            # self.cameras_dust3r.rescale_output_resolution(224 / self.cameras_dust3r.width)
            self.c2w_opencv = torch.eye(4).unsqueeze(0).repeat(len(dataparser_outputs.cameras), 1, 1)
            self.c2w_opencv[:, :3] = deepcopy(dataparser_outputs.cameras).camera_to_worlds
            self.c2w_opencv[:, :, 1:3] *= -1  # opengl to opencv
            pixels_im = torch.stack(
                torch.meshgrid(torch.arange(dataparser_outputs.cameras.height[0].item(), dtype=torch.float32),
                               torch.arange(dataparser_outputs.cameras.width[0].item(), dtype=torch.float32),
                               indexing="ij")).permute(2, 1, 0)
            self.pixels = pixels_im.reshape(-1, 2)

    def get_metadata(self, data: Dict) -> Dict:
        metadata = super().get_metadata(data)

        image_idx = data["image_idx"]
        neighbor_indices = self.metadata[NEIGHBOR_INDICES][image_idx]
        assert len(neighbor_indices) >= self.neighboring_views_size
        neighbor_indices = np.random.choice(neighbor_indices, self.neighboring_views_size,
                                            replace=False)
        metadata[NEIGHBOR_IMAGES] = torch.stack([self.get_image_float32(x) for x in neighbor_indices])
        metadata[NEIGHBOR_INDICES] = torch.LongTensor(neighbor_indices)

        if DEPTH in self.metadata:
            metadata[DEPTH] = self.read_depth(image_idx)
            metadata[NEIGHBOR_DEPTH] = torch.stack([self.read_depth(x) for x in neighbor_indices])
            
            cameras = self._dataparser_outputs.cameras
            if self.return_neighbor_points:
                pts3d_cam = fast_depthmap_to_pts3d(
                    metadata[NEIGHBOR_DEPTH].view(metadata[NEIGHBOR_DEPTH].shape[0], -1),
                    self.pixels,
                    torch.cat([cameras.fx[neighbor_indices], cameras.fy[neighbor_indices]], -1),
                    torch.cat([cameras.cx[neighbor_indices], cameras.cy[neighbor_indices]], -1))
                metadata[PTS3D] = geotrf(self.c2w_opencv[neighbor_indices], pts3d_cam)
            else:
                pts3d_cam = fast_depthmap_to_pts3d(
                    metadata[DEPTH],
                    self.pixels,
                    torch.cat([cameras.fx[image_idx], cameras.fy[image_idx]], -1),
                    torch.cat([cameras.cx[image_idx], cameras.cy[image_idx]], -1))
                metadata[PTS3D] = geotrf(self.c2w_opencv[image_idx], pts3d_cam)

        return metadata

    def read_depth(self, image_idx: int) -> torch.Tensor:
        depth = cv2.imread(str(self.metadata[DEPTH][image_idx]), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[..., :1]

        depth[depth >= 65504.] = 0
        return torch.FloatTensor(depth)
