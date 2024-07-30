from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, Generic, List, Literal, Tuple, Type, Union

import torch
import torch.nn.functional as F
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.datamanagers.base_datamanager import TDataset
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanagerConfig, FullImageDatamanager, \
    _undistort_image
from nerfstudio.utils.rich_utils import CONSOLE
from rich.progress import track
from typing_extensions import assert_never


@dataclass
class UpdatingFullImageDatamanagerConfig(FullImageDatamanagerConfig):
    _target: Type = field(default_factory=lambda: UpdatingFullImageDatamanager)


class UpdatingFullImageDatamanager(FullImageDatamanager, Generic[TDataset]):

    def __init__(
            self,
            config: FullImageDatamanagerConfig,
            device: Union[torch.device, str] = "cpu",
            test_mode: Literal["test", "val", "inference"] = "val",
            world_size: int = 1,
            local_rank: int = 0,
            **kwargs,
    ):
        self.cached_with_updates = []
        super().__init__(config, device, test_mode, world_size, local_rank, **kwargs)

    def sample_train_cameras(self):
        self.train_dataparser_outputs = self.dataparser.get_updated_outputs()
        self.train_dataset = self.create_train_dataset()
        return super().sample_train_cameras()

    def next_train(self, step: int) -> Tuple[Cameras, Dict]:
        """Returns the next training batch

        Returns a Camera instead of raybundle"""
        image_idx = self.train_unseen_cameras.pop(0)
        # Make sure to re-populate the unseen cameras list if we have exhausted it
        if len(self.train_unseen_cameras) == 0:
            self.train_unseen_cameras = self.sample_train_cameras()

        data = deepcopy(self.cached_train_with_updates()[image_idx])
        data["image"] = data["image"].to(self.device)

        assert len(self.train_dataset.cameras.shape) == 1, "Assumes single batch dimension"
        camera = self.train_dataset.cameras[image_idx: image_idx + 1].to(self.device)
        if camera.metadata is None:
            camera.metadata = {}
        camera.metadata["cam_idx"] = image_idx
        return camera, data

    def cached_train_with_updates(self):
        if len(self.cached_with_updates) == 0:
            self.cached_with_updates += self.cached_train

        if len(self.cached_with_updates) != len(self.train_dataset):
            new_updates = self._load_images("train", cache_images_device=self.config.cache_images,
                                            offset=len(self.cached_with_updates))
            self.cached_with_updates += new_updates

        return self.cached_with_updates

    def _load_images(
            self, split: Literal["train", "eval"], cache_images_device: Literal["cpu", "gpu"], offset: int = 0
    ) -> List[Dict[str, torch.Tensor]]:
        # Which dataset?
        if split == "train":
            dataset = self.train_dataset
        elif split == "eval":
            dataset = self.eval_dataset
        else:
            assert_never(split)

        def undistort_idx(idx: int) -> Dict[str, torch.Tensor]:
            data = dataset.get_data(idx, image_type=self.config.cache_images_type)
            camera = dataset.cameras[idx].reshape(())
            if data["image"].shape[1] != camera.width.item() or data["image"].shape[0] != camera.height.item():
                data["image"] = F.interpolate(data["image"].permute(2, 0, 1).unsqueeze(0),
                                              [camera.height.item(), camera.width.item()], mode="bicubic",
                                              antialias=True).squeeze(0).permute(1, 2, 0)

            # assert data["image"].shape[1] == camera.width.item() and data["image"].shape[0] == camera.height.item(), (
            #     f'The size of image ({data["image"].shape[1]}, {data["image"].shape[0]}) loaded '
            #     f'does not match the camera parameters ({camera.width.item(), camera.height.item()})'
            # )
            if camera.distortion_params is None or torch.all(camera.distortion_params == 0):
                return data
            K = camera.get_intrinsics_matrices().numpy()
            distortion_params = camera.distortion_params.numpy()
            image = data["image"].numpy()

            K, image, mask = _undistort_image(camera, distortion_params, data, image, K)
            data["image"] = torch.from_numpy(image)
            if mask is not None:
                data["mask"] = mask

            dataset.cameras.fx[idx] = float(K[0, 0])
            dataset.cameras.fy[idx] = float(K[1, 1])
            dataset.cameras.cx[idx] = float(K[0, 2])
            dataset.cameras.cy[idx] = float(K[1, 2])
            dataset.cameras.width[idx] = image.shape[1]
            dataset.cameras.height[idx] = image.shape[0]
            return data

        CONSOLE.log(f"Caching / undistorting {split} images")
        with ThreadPoolExecutor(max_workers=2) as executor:
            undistorted_images = list(
                track(
                    executor.map(
                        undistort_idx,
                        range(offset, len(dataset)),
                    ),
                    description=f"Caching / undistorting {split} images from offset {offset}",
                    transient=True,
                    total=len(dataset),
                )
            )

        # Move to device.
        if cache_images_device == "gpu":
            for cache in undistorted_images:
                cache["image"] = cache["image"].to(self.device)
                if "mask" in cache:
                    cache["mask"] = cache["mask"].to(self.device)
        elif cache_images_device == "cpu":
            for cache in undistorted_images:
                cache["image"] = cache["image"].pin_memory()
                if "mask" in cache:
                    cache["mask"] = cache["mask"].pin_memory()
        else:
            assert_never(cache_images_device)

        return undistorted_images
