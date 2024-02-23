"""
Dataset that provides mask values as metadata.
"""

from typing import Dict

import numpy as np
import torch
from PIL import Image
from nerfstudio.data.datasets.base_dataset import InputDataset

from generfstudio.generfstudio_constants import FG_MASK


class MaskedDataset(InputDataset):
    def get_metadata(self, data: Dict) -> Dict:
        metadata = super().get_metadata(data)

        image = np.asarray(Image.open(self.image_filenames[data["image_idx"]]))
        if image.shape[-1] == 4:
            metadata[FG_MASK] = torch.BoolTensor(image[..., 3] > 0).unsqueeze(-1).float()

        return metadata
