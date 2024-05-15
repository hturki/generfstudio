from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
from functools import cached_property
from itertools import cycle
from pathlib import Path
from typing import Dict, ForwardRef, Generic, List, Literal, Tuple, Type, Union, cast, get_args, get_origin, \
    Callable, Any, Optional

import torch
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.dataparser_configs import AnnotatedDataParserUnion
from nerfstudio.data.datamanagers.base_datamanager import DataManager, DataManagerConfig, TDataset
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.utils.nerfstudio_collate import nerfstudio_collate
from nerfstudio.model_components.ray_generators import RayGenerator
from nerfstudio.utils.misc import get_orig_class
from torch.nn import Parameter
from torch.utils.data import DistributedSampler, DataLoader

from generfstudio.generfstudio_constants import NEIGHBOR_IMAGES, NEIGHBOR_CAMERAS, \
    NEIGHBOR_INDICES, NEIGHBOR_PTS3D
from generfstudio.generfstudio_utils import repeat_interleave


@dataclass
class NeighboringViewsDatamanagerConfig(DataManagerConfig):
    _target: Type = field(default_factory=lambda: NeighboringViewsDatamanager)

    dataparser: AnnotatedDataParserUnion = field(default_factory=NerfstudioDataParserConfig)

    camera_res_scale_factor: float = 1.0

    image_batch_size: int = 4

    neighboring_views_size: int = 3

    rays_per_image: Optional[int] = None

    collate_fn: Callable[[Any], Any] = cast(Any, staticmethod(nerfstudio_collate))

    train_chunks: int = 10


class NeighboringViewsDatamanager(DataManager, Generic[TDataset]):
    config: NeighboringViewsDatamanagerConfig
    train_dataset: TDataset
    eval_dataset: TDataset

    def __init__(
            self,
            config: NeighboringViewsDatamanagerConfig,
            device: Union[torch.device, str] = "cpu",
            test_mode: Literal["test", "val", "inference"] = "val",
            world_size: int = 1,
            local_rank: int = 0,
            **kwargs,
    ):
        self.config = config
        self.device = device
        self.world_size = world_size
        self.local_rank = local_rank
        self.test_mode = test_mode
        self.test_split = "test" if test_mode in ["test", "inference"] else "val"
        self.dataparser_config = self.config.dataparser
        if self.config.data is not None:
            self.config.dataparser.data = Path(self.config.data)
        else:
            self.config.data = self.config.dataparser.data
        self.dataparser = self.dataparser_config.setup()
        if test_mode == "inference":
            self.dataparser.downscale_factor = 1  # Avoid opening images
        self.includes_time = self.dataparser.includes_time

        if world_size > 1:
            self.load_train_chunk_executor = ThreadPoolExecutor(max_workers=1)
            self.chunk_index = cycle(range(self.config.train_chunks))
            self.load_train_chunk_future = self.load_train_chunk_executor.submit(
                self.dataparser._generate_dataparser_outputs, "train", next(self.chunk_index), self.config.train_chunks)

        self.set_train_data()
        assert self.train_cameras.distortion_params is None

        self.eval_dataparser_outputs = self.dataparser.get_dataparser_outputs(split=self.test_split)
        self.eval_dataset = self.create_eval_dataset()
        self.eval_cameras = self.eval_dataparser_outputs.cameras
        assert self.eval_cameras.distortion_params is None

        self.eval_unseen_cameras = [i for i in range(len(self.eval_dataset))]

        super().__init__()

    def set_train_data(self):
        if self.world_size > 1:
            self.train_dataparser_outputs = self.load_train_chunk_future.result()
            self.load_train_chunk_future = self.load_train_chunk_executor.submit(
                self.dataparser._generate_dataparser_outputs, "train", next(self.chunk_index), self.config.train_chunks)
        else:
            self.train_dataparser_outputs = self.dataparser.get_dataparser_outputs(split="train")

        self.train_dataset = self.create_train_dataset()
        self.train_cameras = self.train_dataparser_outputs.cameras

    def create_train_dataset(self) -> TDataset:
        """Sets up the data loaders for training"""
        return self.dataset_type(
            dataparser_outputs=self.train_dataparser_outputs,
            scale_factor=self.config.camera_res_scale_factor,
            neighboring_views_size=self.config.neighboring_views_size
        )

    def create_eval_dataset(self) -> TDataset:
        """Sets up the data loaders for evaluation"""
        return self.dataset_type(
            dataparser_outputs=self.eval_dataparser_outputs,
            scale_factor=self.config.camera_res_scale_factor,
            neighboring_views_size=self.config.neighboring_views_size
        )

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

    def get_datapath(self) -> Path:
        return self.config.dataparser.data

    def setup_train(self):
        super().setup_train()
        self.set_train_loader()
        if self.config.rays_per_image is not None:
            self.train_ray_generator = RayGenerator(self.train_cameras.to(self.device))

    # def setup_eval(self):
    #     """Sets up the data loader for evaluation"""

    def set_train_loader(self):
        assert self.config.image_batch_size % self.world_size == 0
        batch_size = self.config.image_batch_size // self.world_size

        # do the sharding at the dataparser level for CPU memory efficiency
        # if self.world_size > 1:
        #     # if self.train_sampler is not None:
        #     #     epoch = self.train_sampler.epoch
        #     self.train_sampler = DistributedSampler(self.train_dataset, self.world_size, self.local_rank)
        #     # if self.train_sampler is not None:
        #     #     self.train_sampler.set_epoch(epoch)
        #
        #     self.train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size,
        #                                        sampler=self.train_sampler, num_workers=0,
        #                                        pin_memory=True, collate_fn=self.config.collate_fn)
        # else:
        if self.world_size > 1:
            # define train sampler to keep nerfstudio happy
            self.train_sampler = DistributedSampler(self.train_dataset, self.world_size, self.local_rank)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True,
                                           num_workers=4, pin_memory=True, collate_fn=self.config.collate_fn)

        self.iter_train_dataloader = iter(self.train_dataloader)

    @property
    def fixed_indices_eval_dataloader(self) -> List[Tuple[Cameras, Dict]]:
        """
        Pretends to be the dataloader for evaluation, it returns a list of (camera, data) tuples
        """
        image_indices = [i for i in range(len(self.eval_dataset))]
        data = []
        cameras = []
        for i in image_indices:
            data.append(self.eval_dataset[i])
            data[i]["image"] = data[i]["image"].to(self.device)
            cameras.append(self.eval_cameras[i: i + 1])

        return list(zip(cameras, data))

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Get the param groups for the data manager.
        Returns:
            A list of dictionaries containing the data manager's param groups.
        """
        return {}

    def get_train_rays_per_batch(self):
        # TODO: fix this to be the resolution of the last image rendered
        return 800 * 800

    def next_train(self, step: int) -> Tuple[Union[Cameras, RayBundle], Dict]:
        data = next(self.iter_train_dataloader, None)
        if data is None:
            if self.world_size > 1:
                self.set_train_data()
                self.setup_train()
            else:
                self.iter_train_dataloader = iter(self.train_dataloader)

            data = next(self.iter_train_dataloader)

        if self.config.rays_per_image is not None:
            batch_size, height, width = data["image"].shape[:3]
            batch_indices = repeat_interleave(torch.arange(batch_size), self.config.rays_per_image)
            pixels_x = torch.randint(0, width, (batch_size * self.config.rays_per_image,))
            pixels_y = torch.randint(0, height, (batch_size * self.config.rays_per_image,))

            full_data = data
            data = {}
            for key, val in full_data.items():
                if key in {NEIGHBOR_IMAGES, NEIGHBOR_INDICES, "image_idx"}:
                    data[key] = val
                else:
                    data[key] = val[batch_indices, pixels_y, pixels_x].to(self.device)

            to_return = self.train_ray_generator(
                torch.cat([repeat_interleave(data["image_idx"], self.config.rays_per_image).unsqueeze(-1),
                           pixels_y.unsqueeze(-1), pixels_x.unsqueeze(-1)], -1))
        else:
            data["image"] = data["image"].to(self.device)
            to_return = self.train_cameras[data["image_idx"]].to(self.device)
            if to_return.metadata is None:
                to_return.metadata = {}

        data[NEIGHBOR_IMAGES] = data[NEIGHBOR_IMAGES].to(self.device)
        self.randomize_background(data)

        to_return.metadata[NEIGHBOR_IMAGES] = data[NEIGHBOR_IMAGES]
        del data[NEIGHBOR_IMAGES]

        to_return.metadata[NEIGHBOR_CAMERAS] = self.train_cameras[data[NEIGHBOR_INDICES]].to(
            self.device)
        del data[NEIGHBOR_INDICES]

        if NEIGHBOR_PTS3D in data:
            to_return.metadata[NEIGHBOR_PTS3D] = data[NEIGHBOR_PTS3D].to(self.device)
            del data[NEIGHBOR_PTS3D]

        # Used in mv_diffusion - DDP expects us to do all differentiable model computation in the forward function
        to_return.metadata["image"] = data["image"]

        return to_return, data

    def next_eval(self, step: int) -> Tuple[Cameras, Dict]:
        raise Exception()

    def next_eval_image(self, step: int) -> Tuple[Cameras, Dict]:
        """Returns the next evaluation batch

        Returns a Camera instead of raybundle

        TODO: Make sure this logic is consistent with the vanilladatamanager"""
        image_idx = self.eval_unseen_cameras.pop(random.randint(0, len(self.eval_unseen_cameras) - 1))
        # Make sure to re-populate the unseen cameras list if we have exhausted it
        if len(self.eval_unseen_cameras) == 0:
            self.eval_unseen_cameras = [i for i in range(len(self.eval_dataset))]

        data = deepcopy(self.eval_dataset[image_idx])
        data["image"] = data["image"].to(self.device).unsqueeze(0)  # unsqueeze for randomize_background
        assert len(self.eval_cameras.shape) == 1, "Assumes single batch dimension"
        camera = self.eval_cameras[image_idx:image_idx + 1].to(self.device)
        if camera.metadata is None:
            camera.metadata = {}

        data[NEIGHBOR_IMAGES] = data[NEIGHBOR_IMAGES].to(self.device).unsqueeze(0)
        self.randomize_background(data)
        data["image"] = data["image"].to(self.device).squeeze(0)
        camera.metadata[NEIGHBOR_IMAGES] = data[NEIGHBOR_IMAGES]
        # del data[NEIGHBORING_VIEW_IMAGES] keep it to log in wandb

        camera.metadata[NEIGHBOR_CAMERAS] = self.eval_cameras[data[NEIGHBOR_INDICES].unsqueeze(0)].to(self.device)
        del data[NEIGHBOR_INDICES]

        if NEIGHBOR_PTS3D in data:
            camera.metadata[NEIGHBOR_PTS3D] = data[NEIGHBOR_PTS3D].unsqueeze(0).to(self.device)
            del data[NEIGHBOR_PTS3D]

        return camera, data

    # Randomize background if there's an alpha component and make sure it's consistent across neighboring views
    @staticmethod
    def randomize_background(data: Dict) -> None:
        if data["image"].shape[-1] == 4:
            random_bg = torch.rand(data["image"].shape[0], 3, device=data["image"].device)
            data["image"] = data["image"][..., :3] * data["image"][..., 3:] \
                            + random_bg.view(data["image"].shape[0], 1, 1, 3) * (1 - data["image"][..., 3:])
            data[NEIGHBOR_IMAGES] = data[NEIGHBOR_IMAGES][..., :3] \
                                    * data[NEIGHBOR_IMAGES][..., 3:] \
                                    + random_bg.view(data["image"].shape[0], 1, 1, 1, 3) \
                                    * (1 - data[NEIGHBOR_IMAGES][..., 3:])
