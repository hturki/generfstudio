import torch
import torch.nn.functional as F
from torch import nn
from transformers import Dinov2Backbone


class DinoV2Encoder(nn.Module):

    def __init__(self, backbone="facebook/dinov2-base"):
        super().__init__()
        self.encoder = Dinov2Backbone.from_pretrained(backbone)
        self.out_feature_dim = self.encoder.encoder.layer[-1].mlp.fc2.out_features # should be a better way

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # Dimensions need to be multiples of 14 -  we should be careful to make sure features line up with projected pixels
        # This may stretch things a bit, is there a better way? Not an issue with square images in any case
        height, width = images.shape[-2:]
        downsampled_images = F.interpolate(images, (height + height % 14, width + width % 14), mode="bilinear")
        return self.encoder(downsampled_images)[0][0]
