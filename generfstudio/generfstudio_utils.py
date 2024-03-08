import torch
import torch.nn.functional as F
from PIL import Image


# From common.central_crop_v2 in ZeroNVS
def central_crop_v2(image: Image):
    h, w = image.size
    s = min(h, w)
    # print(s)
    oh = (h - s) // 2
    oh_resid = (h - s) % 2
    ow = (w - s) // 2
    ow_resid = (w - s) % 2
    crop_bounds = [oh, ow, h - oh - oh_resid, w - ow - ow_resid]
    # print(crop_bounds)
    new_image = image.crop(crop_bounds)
    assert new_image.size == (s, s), (image.size, (s, s), new_image.size)
    return new_image

@torch.compile
def get_pixel_aligned_features(latent: torch.Tensor, uv: torch.Tensor, uv_scaling: torch.Tensor) -> torch.Tensor:
    """
    Get pixel-aligned image features at 2D image coordinates
    """
    uv = uv * uv_scaling - 1.0
    uv = uv.unsqueeze(2)  # (NS, N, 1, 2)
    samples = F.grid_sample(
        latent,
        uv,
        align_corners=True,
        mode="bilinear",
        padding_mode="border",
    )

    return samples[:, :, :, 0].transpose(1, 2)  # (B, N, C)
@torch.compile
def repeat_interleave(input: torch.Tensor, repeats: int) -> torch.Tensor:
    """
    Repeat interleave along axis 0
    torch.repeat_interleave is currently very slow
    https://github.com/pytorch/pytorch/issues/31980
    """
    output = input.unsqueeze(1).expand(-1, repeats, *input.shape[1:])
    return output.reshape(-1, *input.shape[1:])