from copy import deepcopy
from functools import cached_property
from pathlib import Path
from typing import Dict, List, Tuple, Literal

import math
import numpy as np
import numpy.typing as npt
import torch
from jaxtyping import Float, UInt8
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.rich_utils import CONSOLE
from torch import Tensor
from torch.utils.data import Dataset

from generfstudio.generfstudio_constants import NEAR, FAR, POSENC_SCALE


class UnionDataset(Dataset):
    def __init__(self, delegates: List[InputDataset]):
        super().__init__()
        self.delegates = delegates
        
        min_bounds, max_bounds = delegates[0].scene_box.aabb
        for delegate in delegates[1:]:
            min_bounds = torch.minimum(min_bounds, delegate.scene_box.aabb[0])
            max_bounds = torch.maximum(max_bounds, delegate.scene_box.aabb[1])
        self.scene_box = SceneBox(aabb=torch.stack([min_bounds, max_bounds]))

        self.metadata = {k: delegates[0].metadata[k] for k in [NEAR, FAR, POSENC_SCALE]}
        for index, delegate in enumerate(delegates[1:]):
            for key, val in delegate.metadata.items():
                if key == NEAR:
                    self.metadata[NEAR] = min(self.metadata[NEAR], val)
                elif key == FAR:
                    self.metadata[FAR] = max(self.metadata[FAR], val)
                elif key == POSENC_SCALE:
                    self.metadata[POSENC_SCALE] = min(self.metadata[POSENC_SCALE], val)
                else:
                    CONSOLE.log(f"Ignoring metadata key {key}")

        self.len = 0
        for delegate in self.delegates:
            self.len += len(delegate)

    def __len__(self):
        return self.len

    def get_delegate(self, image_idx: int) -> Tuple[InputDataset, int]:
        local_idx = image_idx
        for delegate in self.delegates:
            delegate_len = len(delegate)
            if local_idx < delegate_len:
                return delegate, local_idx
            local_idx -= delegate_len

        raise Exception()

    def get_numpy_image(self, image_idx: int) -> npt.NDArray[np.uint8]:
        """Returns the image of shape (H, W, 3 or 4).

        Args:
            image_idx: The image index in the dataset.
        """
        delegate, local_idx = self.get_delegate(image_idx)
        return delegate.get_numpy_image(local_idx)

    def get_image_float32(self, image_idx: int) -> Float[Tensor, "image_height image_width num_channels"]:
        """Returns a 3 channel image in float32 torch.Tensor.

        Args:
            image_idx: The image index in the dataset.
        """
        delegate, local_idx = self.get_delegate(image_idx)
        return delegate.get_image_float32(local_idx)

    def get_image_uint8(self, image_idx: int) -> UInt8[Tensor, "image_height image_width num_channels"]:
        """Returns a 3 channel image in uint8 torch.Tensor.

        Args:
            image_idx: The image index in the dataset.
        """
        delegate, local_idx = self.get_delegate(image_idx)
        return delegate.get_image_uint8(local_idx)

    def get_data(self, image_idx: int, image_type: Literal["uint8", "float32"] = "float32") -> Dict:
        """Returns the ImageDataset data as a dictionary.

        Args:
            image_idx: The image index in the dataset.
            image_type: the type of images returned
        """
        delegate, local_idx = self.get_delegate(image_idx)
        return delegate.get_data(local_idx, image_type)

    def __getitem__(self, image_idx: int) -> Dict:
        data = self.get_data(image_idx)
        return data

    @cached_property
    def image_filenames(self) -> List[Path]:
        """
        Returns image filenames for this dataset.
        The order of filenames is the same as in the Cameras object for easy mapping.
        """

        image_filenames = []
        for delegate in self.delegates:
            image_filenames += delegate.image_filenames

        return image_filenames
