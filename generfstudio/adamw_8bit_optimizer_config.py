from dataclasses import dataclass
from typing import Type

import bitsandbytes as bnb
from nerfstudio.engine.optimizers import OptimizerConfig


@dataclass
class AdamW8bitOptimizerConfig(OptimizerConfig):
    _target: Type = bnb.optim.AdamW8bit
    weight_decay: float = 0
    """The weight decay to use."""