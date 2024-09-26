import os
from copy import deepcopy
from typing import Dict

import numpy as np
import numpy.typing as npt
import torch
from PIL import Image
from dust3r.utils.geometry import geotrf
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.datasets.base_dataset import InputDataset

from generfstudio.fields.batched_pc_optimizer import fast_depthmap_to_pts3d
from generfstudio.generfstudio_constants import NEIGHBOR_INDICES, DEPTH, PTS3D, IN_HDF5

try:
    import h5py
except ImportError:
    pass

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2



class NeighboringViewsDataset(InputDataset):
    def __init__(self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1.0,
                 views_size: int = 3, in_hdf5: bool = False, base_image_first: bool = True):
        # super().__init__(dataparser_outputs, scale_factor)
        # Skip the deepcopy to save time
        self._dataparser_outputs = dataparser_outputs
        self.scale_factor = scale_factor
        self.scene_box = dataparser_outputs.scene_box
        self.metadata = dataparser_outputs.metadata
        # self.cameras = dataparser_outputs.cameras
        # self.cameras.rescale_output_resolution(scaling_factor=scale_factor)
        self.mask_color = dataparser_outputs.metadata.get("mask_color", None)

        self.views_size = views_size
        self.base_image_first = base_image_first

        if DEPTH in self.metadata:
            self.c2w_opencv = torch.eye(4).unsqueeze(0).repeat(len(dataparser_outputs.cameras), 1, 1)
            self.c2w_opencv[:, :3] = deepcopy(dataparser_outputs.cameras).camera_to_worlds
            self.c2w_opencv[:, :, 1:3] *= -1  # opengl to opencv
            pixels_im = torch.stack(
                torch.meshgrid(torch.arange(dataparser_outputs.cameras.height[0].item(), dtype=torch.float32),
                               torch.arange(dataparser_outputs.cameras.width[0].item(), dtype=torch.float32),
                               indexing="ij")).permute(2, 1, 0)
            self.pixels = pixels_im.reshape(-1, 2)

    def get_numpy_image(self, image_idx: int) -> npt.NDArray[np.uint8]:
        """Returns the image of shape (H, W, 3 or 4).

        Args:
            image_idx: The image index in the dataset.
        """
        image_filename = self._dataparser_outputs.image_filenames[image_idx]
        if self.metadata.get(IN_HDF5, False):
            f = h5py.File(image_filename.parent / "data.hdf5", 'r')
            pil_image = Image.fromarray(f[image_filename.name][:])
        else:
            pil_image = Image.open(image_filename)

        if self.scale_factor != 1.0:
            width, height = pil_image.size
            newsize = (int(width * self.scale_factor), int(height * self.scale_factor))
            pil_image = pil_image.resize(newsize, resample=Image.Resampling.BILINEAR)
        image = np.array(pil_image, dtype="uint8")  # shape is (h, w) or (h, w, 3 or 4)
        if len(image.shape) == 2:
            image = image[:, :, None].repeat(3, axis=2)
        assert len(image.shape) == 3
        assert image.dtype == np.uint8
        assert image.shape[2] in [3, 4], f"Image shape of {image.shape} is in correct."
        return image

    def get_metadata(self, data: Dict) -> Dict:
        metadata = super().get_metadata(data)

        image_idx = data["image_idx"]
        neighbor_indices = self.metadata[NEIGHBOR_INDICES][image_idx]
        assert len(neighbor_indices) >= self.views_size
        neighbor_indices = np.random.choice(neighbor_indices, self.views_size, replace=False)
        neighbor_images = [self.get_image_float32(x) for x in neighbor_indices]

        if self.base_image_first:
            data["image"] = torch.stack([data["image"]] + neighbor_images)
            data["image_idx"] = torch.LongTensor([image_idx] + neighbor_indices.tolist())
        else:
            data["image"] = torch.stack(neighbor_images + [data["image"]])
            data["image_idx"] = torch.LongTensor(neighbor_indices.tolist() + [image_idx])

        if DEPTH in self.metadata:
            metadata[DEPTH] = torch.stack([self.read_depth(x) for x in data["image_idx"]])

            image_indices = data["image_idx"]
            cameras = self._dataparser_outputs.cameras
            pts3d_cam = fast_depthmap_to_pts3d(
                metadata[DEPTH].view(metadata[DEPTH].shape[0], -1),
                self.pixels,
                torch.cat([cameras.fx[image_indices], cameras.fy[image_indices]], -1),
                torch.cat([cameras.cx[image_indices], cameras.cy[image_indices]], -1))
            metadata[PTS3D] = geotrf(self.c2w_opencv[image_indices], pts3d_cam)

        return metadata

    def read_depth(self, image_idx: int) -> torch.Tensor:
        if self.metadata.get(IN_HDF5, False):
            f = h5py.File(self.metadata[DEPTH][image_idx].parent / "data.hdf5", 'r')
            depth = f[self.metadata[DEPTH][image_idx].name][:]
        else:
            depth = cv2.imread(str(self.metadata[DEPTH][image_idx]), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[..., 0]

        valid_depth = depth[depth < 65504.]
        if valid_depth.size > 0:
            depth[depth >= 65504.] = valid_depth.max() * 3
        else:
            depth = np.zeros_like(depth)

        return torch.FloatTensor(depth)
