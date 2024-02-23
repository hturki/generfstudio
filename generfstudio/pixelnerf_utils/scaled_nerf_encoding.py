from typing import Optional

import torch
from jaxtyping import Float
from nerfstudio.field_components import Encoding
from nerfstudio.utils.math import expected_sin
from torch import Tensor


class ScaledNeRFEncoding(Encoding):
    """NeRFEncoding with a configurable scaling factor instead of hardcoded to PI
    """

    def __init__(
            self,
            in_dim: int,
            num_frequencies: int,
            min_freq_exp: float,
            max_freq_exp: float,
            scale: float,
            include_input: bool = False,
    ) -> None:
        super().__init__(in_dim)

        self.num_frequencies = num_frequencies
        self.min_freq = min_freq_exp
        self.max_freq = max_freq_exp
        self.scale = scale
        self.include_input = include_input

    def get_out_dim(self) -> int:
        if self.in_dim is None:
            raise ValueError("Input dimension has not been set")
        out_dim = self.in_dim * self.num_frequencies * 2
        if self.include_input:
            out_dim += self.in_dim
        return out_dim

    def forward(
            self, in_tensor: Float[Tensor, "*bs input_dim"],
            covs: Optional[Float[Tensor, "*bs input_dim input_dim"]] = None
    ) -> Float[Tensor, "*bs output_dim"]:
        """Calculates NeRF encoding. If covariances are provided the encodings will be integrated as proposed
            in mip-NeRF.

        Args:
            in_tensor: For best performance, the input tensor should be between 0 and 1.
            covs: Covariances of input points.
        Returns:
            Output values will be between -1 and 1
        """

        # scaled_in_tensor = 2 * torch.pi * in_tensor  # scale to [0, 2pi]
        freqs = self.scale * 2 ** torch.linspace(self.min_freq, self.max_freq, self.num_frequencies,
                                                 device=in_tensor.device)
        scaled_inputs = in_tensor[..., None] * freqs  # [..., "input_dim", "num_scales"]
        scaled_inputs = scaled_inputs.view(*scaled_inputs.shape[:-2], -1)  # [..., "input_dim" * "num_scales"]

        if covs is None:
            encoded_inputs = torch.sin(torch.cat([scaled_inputs, scaled_inputs + torch.pi / 2.0], dim=-1))
        else:
            input_var = torch.diagonal(covs, dim1=-2, dim2=-1)[..., :, None] * freqs[None, :] ** 2
            input_var = input_var.reshape((*input_var.shape[:-2], -1))
            encoded_inputs = expected_sin(
                torch.cat([scaled_inputs, scaled_inputs + torch.pi / 2.0], dim=-1), torch.cat(2 * [input_var], dim=-1)
            )

        if self.include_input:
            encoded_inputs = torch.cat([in_tensor, encoded_inputs], dim=-1)
        return encoded_inputs
