from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from functools import cached_property
from typing import Generic, Type, Tuple, Dict, get_origin, get_args, ForwardRef, cast, Union, Literal

import numpy as np
import torch
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers.base_datamanager import TDataset
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanager, FullImageDatamanagerConfig
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.model_components.ray_generators import RayGenerator
from nerfstudio.utils.misc import get_orig_class

from generfstudio.generfstudio_constants import NEIGHBORING_VIEW_INDICES, NEIGHBORING_VIEW_IMAGES, \
    NEIGHBORING_VIEW_CAMERAS, NEIGHBORING_VIEW_COUNT


@dataclass
class NeighboringViewsDatamanagerConfig(FullImageDatamanagerConfig):
    _target: Type = field(default_factory=lambda: NeighboringViewsDatamanager)

    image_batch_size: int = 4

    rays_per_image_batch_size: int = 128

    neighboring_views_size: int = 3


class NeighboringViewsDatamanager(FullImageDatamanager, Generic[TDataset]):
    config: NeighboringViewsDatamanagerConfig

    def __init__(
            self,
            config: FullImageDatamanagerConfig,
            device: Union[torch.device, str] = "cpu",
            test_mode: Literal["test", "val", "inference"] = "val",
            world_size: int = 1,
            local_rank: int = 0,
            **kwargs,
    ):
        super().__init__(config, device, test_mode, world_size, local_rank, **kwargs)
        # self.cur_image_batch = None
        # self.cur_image_data = None
        # self.cur_neighbor_cameras = None
        # self.cur_image_batch_rays = None
        # self.cur_ray_batch_index = None

    @cached_property
    def dataset_type(self) -> Type[TDataset]:
        """Returns the dataset type passed as the generic argument"""
        default: Type[TDataset] = cast(TDataset, TDataset.__default__)  # type: ignore
        orig_class: Type[NeighboringViewsDatamanager] = get_orig_class(self, default=None)  # type: ignore
        if type(self) is NeighboringViewsDatamanager and orig_class is None:
            return default
        if orig_class is not None and get_origin(orig_class) is NeighboringViewsDatamanager:
            return get_args(orig_class)[0]

        # For inherited classes, we need to find the correct type to instantiate
        for base in getattr(self, "__orig_bases__", []):
            if get_origin(base) is NeighboringViewsDatamanager:
                for value in get_args(base):
                    if isinstance(value, ForwardRef):
                        if value.__forward_evaluated__:
                            value = value.__forward_value__
                        elif value.__forward_module__ is None:
                            value.__forward_module__ = type(self).__module__
                            value = getattr(value, "_evaluate")(None, None, set())
                    assert isinstance(value, type)
                    if issubclass(value, InputDataset):
                        return cast(Type[TDataset], value)
        return default

    def setup_train(self):
        super().setup_train()
        self.train_ray_generator = RayGenerator(self.train_dataset.cameras.to(self.device))

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        if len(self.train_unseen_cameras) > self.config.image_batch_size:
            cur_image_batch = torch.IntTensor(np.random.choice(len(self.train_unseen_cameras),
                                                               self.config.image_batch_size, replace=False))
        else:
            cur_image_batch = torch.IntTensor(np.arange(len(self.train_unseen_cameras)))

        neighboring_view_indices = []
        data_list = defaultdict(list)
        ray_indices = []
        for image_index in cur_image_batch:
            image_data = self.cached_train[image_index]
            height = self.train_dataset.cameras.height[image_index].item()
            width = self.train_dataset.cameras.width[image_index].item()
            pixels_x = torch.randint(0, width, (self.config.rays_per_image_batch_size,))
            pixels_y = torch.randint(0, height, (self.config.rays_per_image_batch_size,))
            ray_indices.append(torch.cat(
                [torch.full_like(pixels_x, image_index).unsqueeze(-1), pixels_y.unsqueeze(-1), pixels_x.unsqueeze(-1)],
                -1))

            for key, val in image_data.items():
                if key == NEIGHBORING_VIEW_INDICES:
                    assert len(val) >= self.config.neighboring_views_size
                    neighboring_view_indices.append(np.random.choice(val, self.config.neighboring_views_size,
                                                                     replace=False))

                elif isinstance(val, torch.Tensor):
                    # Ensure that the data is the same as the image dimensions
                    assert val.shape[0] == height
                    assert val.shape[1] == width
                    data_list[key].append(val[pixels_y, pixels_x])

        data = {}
        for key, val in data_list.items():
            data[key] = torch.cat(val)

        ray_bundle = self.train_ray_generator(torch.cat(ray_indices))
        neighboring_view_indices = torch.LongTensor(neighboring_view_indices).view(-1)

        ray_bundle.metadata[NEIGHBORING_VIEW_IMAGES] = \
            [deepcopy(self.cached_train[x]["image"]).to(self.device) for x in neighboring_view_indices]

        ray_bundle.metadata[NEIGHBORING_VIEW_CAMERAS] = self.train_dataset.cameras[neighboring_view_indices].to(
            self.device)
        ray_bundle.metadata[NEIGHBORING_VIEW_COUNT] = self.config.neighboring_views_size

        return ray_bundle, data

    def next_eval(self, step: int) -> Tuple[Cameras, Dict]:
        raise Exception()
        # camera, data = super().next_eval(step)
        # return self.with_neighboring_views(camera, data)

    def next_eval_image(self, step: int) -> Tuple[Cameras, Dict]:
        camera, data = super().next_eval_image(step)
        neighboring_view_indices = data[NEIGHBORING_VIEW_INDICES]
        del data[NEIGHBORING_VIEW_INDICES]
        random_indices = neighboring_view_indices if len(neighboring_view_indices) <= self.config.neighboring_views_size \
            else np.random.choice(neighboring_view_indices, self.config.neighboring_views_size, replace=False)
        random_images = [deepcopy(self.cached_eval[x]["image"]).to(self.device) for x in random_indices]
        random_cameras = self.eval_dataset.cameras[torch.LongTensor(random_indices)].to(self.device)
        if camera.metadata is None:
            camera.metadata = {}
        camera.metadata[NEIGHBORING_VIEW_IMAGES] = random_images
        camera.metadata[NEIGHBORING_VIEW_CAMERAS] = random_cameras
        camera.metadata[NEIGHBORING_VIEW_COUNT] = self.config.neighboring_views_size

        return camera, data
