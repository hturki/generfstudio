from copy import deepcopy
from typing import Dict

import numpy as np
import torch
from dust3r.cloud_opt.optimizer import _fast_depthmap_to_pts3d
from dust3r.utils.geometry import geotrf
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.datasets.base_dataset import InputDataset
from torch.nn import functional as F

from generfstudio.generfstudio_constants import NEIGHBOR_INDICES, NEIGHBOR_IMAGES, DEPTH, NEIGHBOR_PTS3D, NEIGHBOR_DEPTH


class NeighboringViewsDataset(InputDataset):
    def __init__(self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1.0,
                 neighboring_views_size: int = 3, return_target_depth: bool = False):
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
        self.return_target_depth = return_target_depth

        if DEPTH in self.metadata:
            self.cameras_dust3r = deepcopy(dataparser_outputs.cameras)
            self.cameras_dust3r.rescale_output_resolution(224 / self.cameras_dust3r.width)
            self.c2w_dust3r = torch.eye(4).unsqueeze(0).repeat(len(self.cameras_dust3r), 1, 1)
            self.c2w_dust3r[:, :3] = self.cameras_dust3r.camera_to_worlds
            self.c2w_dust3r[:, :, 1:3] *= -1  # opengl to opencv
            pixels_im = torch.stack(
                torch.meshgrid(torch.arange(224, dtype=torch.float32),
                               torch.arange(224, dtype=torch.float32),
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
            neighbor_depth = torch.stack(
                [torch.load(self.metadata[DEPTH][x], map_location="cpu") for x in neighbor_indices])
            pts3d_cam = _fast_depthmap_to_pts3d(neighbor_depth,
                                                self.pixels.unsqueeze(0).expand(neighbor_depth.shape[0], -1, -1),
                                                self.cameras_dust3r.fx[neighbor_indices], torch.cat(
                    [self.cameras_dust3r.cx[neighbor_indices], self.cameras_dust3r.cy[neighbor_indices]], -1))
            metadata[NEIGHBOR_PTS3D] = geotrf(self.c2w_dust3r[neighbor_indices], pts3d_cam)

            if self.return_target_depth:
                depth = torch.load(self.metadata[DEPTH][image_idx], map_location="cpu")
                metadata[DEPTH] = F.interpolate(depth.view(1, 1, 224, 224), data["image"].shape[:2],
                                                mode="bilinear").squeeze().unsqueeze(-1)
                metadata[NEIGHBOR_DEPTH] = neighbor_depth

        return metadata
