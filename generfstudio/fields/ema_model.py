from typing import Iterable, Union, Optional, Any, Dict

import torch
from diffusers import EMAModel
from diffusers.utils import deprecate

from torch import nn


class EmaModel(EMAModel, nn.Module):

    def __init__(
            self,
            parameters: Iterable[torch.nn.Parameter],
            decay: float = 0.9999,
            min_decay: float = 0.0,
            update_after_step: int = 0,
            use_ema_warmup: bool = False,
            inv_gamma: Union[float, int] = 1.0,
            power: Union[float, int] = 2 / 3,
            model_cls: Optional[Any] = None,
            model_config: Dict[str, Any] = None,
            **kwargs,
    ):
        # super(EMAModel, self).__init__(parameters, decay=decay, min_decay=min_decay,
        #                                update_after_step=update_after_step, use_ema_warmup=use_ema_warmup,
        #                                inv_gamma=inv_gamma, power=power, model_cls=model_cls, model_config=model_config,
        #                                kwargs=kwargs)
        if isinstance(parameters, torch.nn.Module):
            deprecation_message = (
                "Passing a `torch.nn.Module` to `ExponentialMovingAverage` is deprecated. "
                "Please pass the parameters of the module instead."
            )
            deprecate(
                "passing a `torch.nn.Module` to `ExponentialMovingAverage`",
                "1.0.0",
                deprecation_message,
                standard_warn=False,
            )
            parameters = parameters.parameters()

            # set use_ema_warmup to True if a torch.nn.Module is passed for backwards compatibility
            use_ema_warmup = True

        if kwargs.get("max_value", None) is not None:
            deprecation_message = "The `max_value` argument is deprecated. Please use `decay` instead."
            deprecate("max_value", "1.0.0", deprecation_message, standard_warn=False)
            decay = kwargs["max_value"]

        if kwargs.get("min_value", None) is not None:
            deprecation_message = "The `min_value` argument is deprecated. Please use `min_decay` instead."
            deprecate("min_value", "1.0.0", deprecation_message, standard_warn=False)
            min_decay = kwargs["min_value"]

        parameters = list(parameters)
        self.shadow_params = [p.clone().detach() for p in parameters]

        if kwargs.get("device", None) is not None:
            deprecation_message = "The `device` argument is deprecated. Please use `to` instead."
            deprecate("device", "1.0.0", deprecation_message, standard_warn=False)
            self.to(device=kwargs["device"])

        self.temp_stored_params = None

        self.decay = decay
        self.min_decay = min_decay
        self.update_after_step = update_after_step
        self.use_ema_warmup = use_ema_warmup
        self.inv_gamma = inv_gamma
        self.power = power
        self.optimization_step = 0
        self.cur_decay_value = None  # set in `step()`

        self.model_cls = model_cls
        self.model_config = model_config
        super(nn.Module, self).__init__()
