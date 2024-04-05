from multiprocessing import Manager

from torch.multiprocessing import Array
from typing import Dict

import numpy as np
import torch
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.utils.comms import get_world_size

from generfstudio.generfstudio_constants import NEIGHBOR_INDICES, NEIGHBOR_IMAGES
from ctypes import c_char, c_int
import numpy.typing as npt
from PIL import Image


class NeighboringViewsDataset(InputDataset):
    def __init__(self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1.0,
                 neighboring_views_size: int = 3):
        # super().__init__(dataparser_outputs, scale_factor)
        # Skip the deepcopy to save time
        self._dataparser_outputs = dataparser_outputs
        self.scale_factor = scale_factor
        self.scene_box = dataparser_outputs.scene_box
        self.metadata = dataparser_outputs.metadata
        # self.cameras = dataparser_outputs.cameras
        # self.cameras.rescale_output_resolution(scaling_factor=scale_factor)
        self.mask_color = dataparser_outputs.metadata.get("mask_color", None)

        # if get_world_size() > 1:
        #     manager = Manager()
        #     dataparser_outputs.image_filenames = manager.list(dataparser_outputs.image_filenames)
        #     self.metadata[NEIGHBOR_INDICES] = manager.list([manager.list(x) for x in self.metadata[NEIGHBOR_INDICES]])

        #
        # self.use_shared = get_world_size() > 1
        #
        if get_world_size() > 1:
            dataparser_outputs.image_filenames = [Array(c_char, str(x).encode(), lock=False)
                                                  for x in dataparser_outputs.image_filenames]
            self.metadata[NEIGHBOR_INDICES] = [Array(c_int, x, lock=False) for x in self.metadata[NEIGHBOR_INDICES]]

        self.neighboring_views_size = neighboring_views_size

    def get_metadata(self, data: Dict) -> Dict:
        metadata = super().get_metadata(data)

        neighbor_indices = self.metadata[NEIGHBOR_INDICES][data["image_idx"]]
        assert len(neighbor_indices) >= self.neighboring_views_size
        neighbor_indices = np.random.choice(neighbor_indices, self.neighboring_views_size,
                                            replace=False)
        metadata[NEIGHBOR_IMAGES] = torch.cat(
            [self.get_image_float32(x).unsqueeze(0) for x in neighbor_indices])
        metadata[NEIGHBOR_INDICES] = torch.LongTensor(neighbor_indices)

        return metadata

    def get_numpy_image(self, image_idx: int) -> npt.NDArray[np.uint8]:
        """Returns the image of shape (H, W, 3 or 4).

        Args:
            image_idx: The image index in the dataset.
        """
        image_filename = self._dataparser_outputs.image_filenames[image_idx]
        if (not isinstance(image_filename, str)) and (not isinstance(image_filename, Path)):
            image_filename = image_filename.value

        pil_image = Image.open(image_filename)
        if self.scale_factor != 1.0:
            width, height = pil_image.size
            newsize = (int(width * self.scale_factor), int(height * self.scale_factor))
            pil_image = pil_image.resize(newsize, resample=Image.BILINEAR)
        image = np.array(pil_image, dtype="uint8")  # shape is (h, w) or (h, w, 3 or 4)
        if len(image.shape) == 2:
            image = image[:, :, None].repeat(3, axis=2)
        assert len(image.shape) == 3
        assert image.dtype == np.uint8
        assert image.shape[2] in [3, 4], f"Image shape of {image.shape} is in correct."
        return image
