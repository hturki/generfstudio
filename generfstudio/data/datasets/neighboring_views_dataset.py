from typing import Dict

import numpy as np
import torch
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.datasets.base_dataset import InputDataset

from generfstudio.generfstudio_constants import NEIGHBORING_VIEW_INDICES, NEIGHBORING_VIEW_IMAGES


class NeighboringViewsDataset(InputDataset):
    def __init__(self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1.0,
                 neighboring_views_size: int = 3):
        super().__init__(dataparser_outputs, scale_factor)
        self.neighboring_views = self.metadata[NEIGHBORING_VIEW_INDICES]
        self.neighboring_views_size = neighboring_views_size

    def get_metadata(self, data: Dict) -> Dict:
        metadata = super().get_metadata(data)

        neighboring_view_indices = self.neighboring_views[data["image_idx"]]
        assert len(neighboring_view_indices) >= self.neighboring_views_size
        neighboring_view_indices = np.random.choice(neighboring_view_indices, self.neighboring_views_size,
                                                    replace=False)
        metadata[NEIGHBORING_VIEW_IMAGES] = torch.cat(
            [self.get_image_float32(x).unsqueeze(0) for x in neighboring_view_indices])
        metadata[NEIGHBORING_VIEW_INDICES] = torch.LongTensor(neighboring_view_indices)

        return metadata
