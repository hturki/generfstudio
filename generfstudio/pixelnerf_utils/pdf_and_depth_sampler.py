from typing import Optional

import torch
from jaxtyping import Float
from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.model_components.ray_samplers import Sampler
from torch import Tensor


class PDFAndDepthSampler(Sampler):
    """Sample based on probability distribution and coarse depth

    Args:
        num_samples: Number of samples per ray
        train_stratified: Randomize location within each bin during training.
        single_jitter: Use a same random jitter for all samples along a ray. Defaults to False
        include_original: Add original samples to ray.
        histogram_padding: Amount to weights prior to computing PDF.
    """

    def __init__(
            self,
            num_samples: Optional[int] = None,
            num_depth_samples: int = 16,
            train_stratified: bool = True,
            single_jitter: bool = False,
            include_original: bool = True,
            histogram_padding: float = 0.01,
            depth_std: float = 0.01
    ) -> None:
        super().__init__(num_samples=num_samples)
        assert num_depth_samples < num_samples
        self.num_depth_samples = num_depth_samples
        self.train_stratified = train_stratified
        self.include_original = include_original
        self.histogram_padding = histogram_padding
        self.depth_std = depth_std
        self.single_jitter = single_jitter

    def generate_ray_samples(
            self,
            ray_bundle: Optional[RayBundle] = None,
            ray_samples: Optional[RaySamples] = None,
            weights: Optional[Float[Tensor, "*batch num_samples 1"]] = None,
            depth: Optional[Float[Tensor, "*batch 1"]] = None,
            num_samples: Optional[int] = None,
            eps: float = 1e-5,
    ) -> RaySamples:
        """Generates position samples given a distribution.

        Args:
            ray_bundle: Rays to generate samples for
            ray_samples: Existing ray samples
            weights: Weights for each bin
            depth: Coarse depth estimate
            num_samples: Number of samples per ray
            eps: Small value to prevent numerical issues.

        Returns:
            Positions and deltas for samples along a ray
        """

        if ray_samples is None or ray_bundle is None:
            raise ValueError("ray_samples and ray_bundle must be provided")
        assert weights is not None, "weights must be provided"

        num_samples = num_samples or self.num_samples
        assert num_samples is not None
        num_samples = num_samples - self.num_depth_samples
        num_bins = num_samples + 1

        weights = weights[..., 0] + self.histogram_padding

        # Add small offset to rays with zero weight to prevent NaNs
        weights_sum = torch.sum(weights, dim=-1, keepdim=True)
        padding = torch.relu(eps - weights_sum)
        weights = weights + padding / weights.shape[-1]
        weights_sum += padding

        pdf = weights / weights_sum
        cdf = torch.min(torch.ones_like(pdf), torch.cumsum(pdf, dim=-1))
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)

        if self.train_stratified and self.training:
            # Stratified samples between 0 and 1
            u = torch.linspace(0.0, 1.0 - (1.0 / num_bins), steps=num_bins, device=cdf.device)
            u = u.expand(size=(*cdf.shape[:-1], num_bins))
            if self.single_jitter:
                rand = torch.rand((*cdf.shape[:-1], 1), device=cdf.device) / num_bins
            else:
                rand = torch.rand((*cdf.shape[:-1], num_samples + 1), device=cdf.device) / num_bins
            u = u + rand
        else:
            # Uniform samples between 0 and 1
            u = torch.linspace(0.0, 1.0 - (1.0 / num_bins), steps=num_bins, device=cdf.device)
            u = u + 1.0 / (2 * num_bins)
            u = u.expand(size=(*cdf.shape[:-1], num_bins))
        u = u.contiguous()

        assert (
                ray_samples.spacing_starts is not None and ray_samples.spacing_ends is not None
        ), "ray_sample spacing_starts and spacing_ends must be provided"
        assert ray_samples.spacing_to_euclidean_fn is not None, "ray_samples.spacing_to_euclidean_fn must be provided"
        existing_bins = torch.cat(
            [
                ray_samples.spacing_starts[..., 0],
                ray_samples.spacing_ends[..., -1:, 0],
            ],
            dim=-1,
        )

        inds = torch.searchsorted(cdf, u, side="right")
        below = torch.clamp(inds - 1, 0, existing_bins.shape[-1] - 1)
        above = torch.clamp(inds, 0, existing_bins.shape[-1] - 1)
        cdf_g0 = torch.gather(cdf, -1, below)
        bins_g0 = torch.gather(existing_bins, -1, below)
        cdf_g1 = torch.gather(cdf, -1, above)
        bins_g1 = torch.gather(existing_bins, -1, above)

        t = torch.clip(torch.nan_to_num((u - cdf_g0) / (cdf_g1 - cdf_g0), 0), 0, 1)
        bins = bins_g0 + t * (bins_g1 - bins_g0)

        # Detach gradients for all bins except for depth
        bins_total = [bins.detach()]

        if self.num_depth_samples > 0:
            depth_samples = depth.expand(-1, self.num_depth_samples)
            depth_samples = depth_samples + torch.randn_like(depth_samples) * self.depth_std
            depth_bins = ((depth_samples - ray_bundle.nears) / (ray_bundle.fars - ray_bundle.nears)).clamp(0, 1)
            bins_total.append(depth_bins)

        if self.include_original:
            bins_total.append(existing_bins.detach())

        bins, _ = torch.sort(torch.cat(bins_total, -1), -1)

        # # Stop gradients
        # bins = bins.detach()

        euclidean_bins = ray_samples.spacing_to_euclidean_fn(bins)

        ray_samples = ray_bundle.get_ray_samples(
            bin_starts=euclidean_bins[..., :-1, None],
            bin_ends=euclidean_bins[..., 1:, None],
            spacing_starts=bins[..., :-1, None],
            spacing_ends=bins[..., 1:, None],
            spacing_to_euclidean_fn=ray_samples.spacing_to_euclidean_fn,
        )

        return ray_samples
