from typing import Dict

import numpy as np
import torch
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.datasets.base_dataset import InputDataset

from generfstudio.generfstudio_constants import NEIGHBOR_INDICES, NEIGHBOR_IMAGES


class NeighboringViewsDataset(InputDataset):
    def __init__(self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1.0,
                 neighboring_views_size: int = 3):
        # super().__init__(dataparser_outputs, scale_factor)
        # Skip the deepcopy to save time
        self._dataparser_outputs = dataparser_outputs
        self.scale_factor = scale_factor
        self.scene_box = dataparser_outputs.scene_box
        self.metadata = dataparser_outputs.metadata
        self.cameras = dataparser_outputs.cameras
        self.cameras.rescale_output_resolution(scaling_factor=scale_factor)
        self.mask_color = dataparser_outputs.metadata.get("mask_color", None)
        self.neighboring_views = self.metadata[NEIGHBOR_INDICES]
        self.neighboring_views_size = neighboring_views_size

    def get_metadata(self, data: Dict) -> Dict:
        metadata = super().get_metadata(data)

        neighbor_indices = self.neighboring_views[data["image_idx"]]
        assert len(neighbor_indices) >= self.neighboring_views_size
        neighbor_indices = np.random.choice(neighbor_indices, self.neighboring_views_size,
                                                    replace=False)
        metadata[NEIGHBOR_IMAGES] = torch.cat(
            [self.get_image_float32(x).unsqueeze(0) for x in neighbor_indices])
        metadata[NEIGHBOR_INDICES] = torch.LongTensor(neighbor_indices)

        return metadata
