import torch
from torch import nn


class DinoV2Encoder(nn.Module):

    def __init__(
            self,
            out_feature_dim: int = 128,
            use_norm: bool = True,
            chunk_size: int = 32,
    ):
        super().__init__()

        self.image_encoder = torch.hub.load("mhamilton723/FeatUp", 'dinov2', use_norm=use_norm)
        self.image_encoder.requires_grad_(False)
        self.proj_layer = nn.Linear(384, out_feature_dim)
        self.chunk_size = chunk_size

    def forward(self, rgbs: torch.Tensor) -> torch.Tensor:
        cond_features = []

        for chunk_index in range(0, rgbs.shape[0], self.chunk_size):
            chunk_results = self.image_encoder(rgbs[chunk_index:chunk_index + self.chunk_size])
            b, d, h, w = chunk_results.shape
            projected = self.proj_layer(chunk_results.permute(0, 2, 3, 1).reshape(-1, d))
            cond_features.append(projected.view(b, h, w, -1).permute(0, 3, 1, 2))

        return torch.cat(cond_features)
