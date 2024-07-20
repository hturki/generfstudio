from dataclasses import dataclass
from typing import Type

from bitsandbytes.optim import AdamW8bit
from nerfstudio.engine.optimizers import OptimizerConfig


@dataclass
class AdamW8bitOptimizerConfig(OptimizerConfig):
    _target: Type = AdamW8bit
    weight_decay: float = 0
    """The weight decay to use."""