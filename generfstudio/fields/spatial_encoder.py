import functools
from typing import Tuple

import torch
import torch.nn.functional as F
import torchvision
from rich.console import Console
from torch import nn

CONSOLE = Console(width=120)


def get_norm_layer(norm_type="instance", group_norm_groups=32):
    """Return a normalization layer
    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none
    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == "batch":
        norm_layer = functools.partial(
            nn.BatchNorm2d, affine=True, track_running_stats=True
        )
    elif norm_type == "instance":
        norm_layer = functools.partial(
            nn.InstanceNorm2d, affine=False, track_running_stats=False
        )
    elif norm_type == "group":
        norm_layer = functools.partial(nn.GroupNorm, group_norm_groups)
    elif norm_type == "none":
        norm_layer = None
    else:
        raise NotImplementedError("normalization layer [%s] is not found" % norm_type)
    return norm_layer


class SpatialEncoder(nn.Module):
    """
    2D (Spatial/Pixel-aligned/local) image encoder
    """

    def __init__(
            self,
            backbone="resnet34",
            pretrained=True,
            num_layers=4,
            index_interp="bilinear",
            index_padding="border",
            upsample_interp="bilinear",
            feature_scale=1.0,
            use_first_pool=True,
            norm_type="batch",
    ):
        """
        :param backbone Backbone network. Should be resnet18 or resnet34
        :param num_layers number of resnet layers to use, 1-5
        :param pretrained Whether to use model weights pretrained on ImageNet
        :param index_interp Interpolation to use for indexing
        :param index_padding Padding mode to use for indexing, border | zeros | reflection
        :param upsample_interp Interpolation to use for upscaling latent code
        :param feature_scale factor to scale all latent by. Useful (<1) if image
        is extremely large, to fit in memory.
        :param use_first_pool if false, skips first maxpool layer to avoid downscaling image
        features too much (ResNet only)
        :param norm_type norm type to applied; pretrained model must use batch
        """
        super().__init__()

        if norm_type != "batch":
            assert not pretrained

        self.feature_scale = feature_scale
        self.use_first_pool = use_first_pool
        norm_layer = get_norm_layer(norm_type)

        CONSOLE.log(f"Using torchvision {backbone} encoder")
        self.model = getattr(torchvision.models, backbone)(
            pretrained=pretrained, norm_layer=norm_layer
        )

        self.model.fc = nn.Sequential()
        self.model.avgpool = nn.Sequential()
        self.latent_size = [0, 64, 128, 256, 512, 1024][num_layers]

        self.num_layers = num_layers
        self.index_interp = index_interp
        self.index_padding = index_padding
        self.upsample_interp = upsample_interp

    def index(self, latent: torch.Tensor, latent_scaling: torch.Tensor, uv: torch.Tensor,
              image_size: torch.Tensor) -> torch.Tensor:
        """
        Get pixel-aligned image features at 2D image coordinates
        :param uv (NS, N, 2) image points (x,y)
        :return (B, L, N) L is latent size
        """
        scale = latent_scaling / image_size
        uv = uv * scale - 1.0

        uv = uv.unsqueeze(2)  # (NS, N, 1, 2)
        samples = F.grid_sample(
            latent,
            uv,
            align_corners=True,
            mode=self.index_interp,
            padding_mode=self.index_padding,
        )

        return samples[:, :, :, 0]  # (B, C, N)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        For extracting ResNet's features
        :param x image (NS, 3, H, W)
        :return latent (NS, latent_size, featureH, featureW) and latent_scaling
        """
        if self.feature_scale != 1.0:
            x = F.interpolate(
                x,
                scale_factor=self.feature_scale,
                mode="bilinear" if self.feature_scale > 1.0 else "area",
                align_corners=True if self.feature_scale > 1.0 else None,
                recompute_scale_factor=True,
            )

        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)

        latents = [x]
        if self.num_layers > 1:
            if self.use_first_pool:
                x = self.model.maxpool(x)
            x = self.model.layer1(x)
            latents.append(x)
        if self.num_layers > 2:
            x = self.model.layer2(x)
            latents.append(x)
        if self.num_layers > 3:
            x = self.model.layer3(x)
            latents.append(x)
        if self.num_layers > 4:
            x = self.model.layer4(x)
            latents.append(x)

        align_corners = None if self.index_interp == "nearest" else True
        latent_sz = latents[0].shape[-2:]

        for i in range(len(latents)):
            latents[i] = F.interpolate(
                latents[i],
                latent_sz,
                mode=self.upsample_interp,
                align_corners=align_corners,
            )

        latent = torch.cat(latents, dim=1)

        latent_scaling = torch.FloatTensor([latent.shape[-1], latent.shape[-2]]).to(latent.device)
        latent_scaling = latent_scaling / (latent_scaling - 1) * 2.0

        return latent, latent_scaling
