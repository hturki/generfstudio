from dataclasses import dataclass
from typing import Type

import torch
from nerfstudio.engine.optimizers import OptimizerConfig
from torch.distributed.optim import ZeroRedundancyOptimizer


@dataclass
class ZeroRedundancyOptimizerConfig(OptimizerConfig):
    _target: Type = ZeroRedundancyOptimizer
    optimizer_class: Type = torch.optim.AdamW

    parameters_as_bucket_view: bool = True

    weight_decay: float = 0
    """The weight decay to use."""