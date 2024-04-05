from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from itertools import cycle
from pathlib import Path
from typing import Dict, List, Literal, Tuple, Union, Type

import torch
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.dataparser_configs import AnnotatedDataParserUnion
from nerfstudio.data.datamanagers.base_datamanager import DataManager, DataManagerConfig, TDataset
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from torch.nn import Parameter
from torch.utils.data import DistributedSampler
from tqdm import tqdm

from generfstudio.data.datamanagers.neighboring_views_datamanager import NeighboringViewsDatamanagerConfig, \
    NeighboringViewsDatamanager
from generfstudio.data.dataparsers.co3d_dataparser import CO3DDataParserConfig
from generfstudio.data.dataparsers.dl3dv_dataparser import DL3DVDataParserConfig
from generfstudio.data.dataparsers.mvimgnet_dataparser import MVImgNetDataParserConfig
from generfstudio.data.dataparsers.objaverse_xl_dataparser import ObjaverseXLDataParserConfig
from generfstudio.data.dataparsers.r10k_dataparser import R10KDataParserConfig
from generfstudio.data.datasets.neighboring_views_dataset import NeighboringViewsDataset
from generfstudio.data.datasets.union_dataset import UnionDataset


class DistributedSamplerWrapper:
    def __init__(self, delegates: List[DistributedSampler]):
        self.delegates = delegates

    def set_epoch(self, step: int):
        for delegate in self.delegates:
            delegate.set_epoch(step)


class DataparserOutputsWrapper:

    def __init__(self, delegates: List[DataparserOutputs]):
        self.delegates = delegates
        self.metadata = {}

    def save_dataparser_transform(self, path: Path):
        self.delegates[0].save_dataparser_transform(path)


@dataclass
class UnionDatamanagerConfig(DataManagerConfig):
    _target: Type = field(default_factory=lambda: UnionDatamanager)

    inner: NeighboringViewsDatamanagerConfig = field(default_factory=lambda: NeighboringViewsDatamanagerConfig(
        _target=NeighboringViewsDatamanager[NeighboringViewsDataset],
    ))

    # Making this a config makes startup very slow
    # dataparsers: List[AnnotatedDataParserUnion] = field(default_factory=lambda: [
    # DL3DVDataParserConfig(),
    # CO3DDataParserConfig(),
    # R10KDataParserConfig(),
    # ObjaverseXLDataParserConfig(),
    # R10KDataParserConfig(data=Path("data/r10k")),
    # MVImgNetDataParserConfig()
    # ])


DATAPARSERS = [
    DL3DVDataParserConfig(),
    CO3DDataParserConfig(),
    R10KDataParserConfig(),
    MVImgNetDataParserConfig(),
    R10KDataParserConfig(data=Path("data/r10k")),
]

# DATAPARSERS = [
#     DL3DVDataParserConfig(),
#     CO3DDataParserConfig(),
#     R10KDataParserConfig(),
#     ObjaverseXLDataParserConfig(),
#     R10KDataParserConfig(data=Path("data/r10k")),
#     MVImgNetDataParserConfig()
# ]


class UnionDatamanager(DataManager):
    config: UnionDatamanagerConfig
    train_dataset: TDataset
    eval_dataset: TDataset

    def __init__(
            self,
            config: UnionDatamanagerConfig,
            device: Union[torch.device, str] = "cpu",
            test_mode: Literal["test", "val", "inference"] = "val",
            world_size: int = 1,
            local_rank: int = 0,
            **kwargs,
    ):
        self.delegates = []
        for dataparser in tqdm(DATAPARSERS):
            inner = deepcopy(config.inner)
            inner.dataparser = dataparser
            self.delegates.append(
                inner.setup(device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank,
                            **kwargs))
        self.train_dataparser_outputs = DataparserOutputsWrapper([x.train_dataparser_outputs for x in self.delegates])

        self.delegates_cycle_train = cycle(self.delegates)
        self.delegates_cycle_eval = cycle(self.delegates)
        self.delegates_cycle_eval_image = cycle(self.delegates)

        self.train_dataset = self.create_train_dataset()
        self.eval_dataset = self.create_eval_dataset()

        self.world_size = world_size
        self.test_mode = test_mode

        super().__init__()

    def create_train_dataset(self) -> TDataset:
        """Sets up the data loaders for training"""
        datasets = [x.create_train_dataset() for x in self.delegates]
        return UnionDataset(datasets)

    def create_eval_dataset(self) -> TDataset:
        """Sets up the data loaders for evaluation"""
        datasets = [x.create_eval_dataset() for x in self.delegates]
        return UnionDataset(datasets)

    def get_datapath(self) -> Path:
        return self.delegates[0].get_datapath()

    def setup_train(self):
        for delegate in self.delegates:
            delegate.setup_train()

        if self.world_size > 1:
            self.train_sampler = DistributedSamplerWrapper([x.train_sampler for x in self.delegates])

    def setup_eval(self):
        for delegate in self.delegates:
            delegate.setup_eval()

        # NeighboringViewsDatamanager doesn't support eval batches for now
        # if self.world_size > 1:
        #     self.eval_sampler = DistributedSamplerWrapper([x.eval_sampler for x in self.delegates])

    @property
    def fixed_indices_eval_dataloader(self) -> List[Tuple[Cameras, Dict]]:
        """
        Pretends to be the dataloader for evaluation, it returns a list of (camera, data) tuples
        """
        indices = []
        for delegate in self.delegates:
            indices.append(delegate.fixed_indices_eval_dataloader)

        return indices

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
        # We're accumulating gradients across multiple batches right now - we
        # would want to split the train batch across multiple delegates if not
        return next(self.delegates_cycle_train).next_train(step)

    def next_eval(self, step: int) -> Tuple[Cameras, Dict]:
        return next(self.delegates_cycle_eval).next_eval(step)

    def next_eval_image(self, step: int) -> Tuple[Cameras, Dict]:
        return next(self.delegates_cycle_eval_image).next_eval_image(step)
