from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Type, Optional

import torch
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import DataParser, DataParserConfig, DataparserOutputs
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.data.utils.dataparsers_utils import get_train_eval_split_fraction
from nerfstudio.utils.comms import get_rank, get_world_size
from nerfstudio.utils.rich_utils import CONSOLE
from tqdm import tqdm

from generfstudio.generfstudio_constants import NEIGHBOR_INDICES, NEAR, FAR, POSENC_SCALE, DEPTH, IN_HDF5


@dataclass
class ObjaverseXLDataParserConfig(DataParserConfig):
    _target: Type = field(default_factory=lambda: ObjaverseXL)
    """target class to instantiate"""
    data: Path = Path("data/oxl")

    """Directory specifying location of data."""

    scene_id: Optional[str] = None

    train_split_fraction: float = 0.9
    """The percentage of the dataset to use for training. Only used when using a single scene."""

    default_scene_id: str = "000/00003e32-f857-5921-a019-f56966f903f3"

    scale_near: Optional[float] = None

    eval_scene_count: int = 50

    in_hdf5: bool = True


@dataclass
class ObjaverseXL(DataParser):
    """ObjaverseXL Dataset"""

    config: ObjaverseXLDataParserConfig

    def _generate_dataparser_outputs(self, split="train", chunk_index=None, num_chunks=None, get_default_scene=False):
        rank = get_rank()
        world_size = get_world_size()
        cached_path = self.config.data / f"cached-metadata-{split}-{self.config.scale_near}-{rank}-{world_size}-{chunk_index}-{num_chunks}.pt"
        if (not get_default_scene) and self.config.scene_id is None and cached_path.exists():
            return torch.load(cached_path)

        if get_default_scene:
            scenes = [self.config.default_scene_id]
        elif self.config.scene_id is not None:
            scenes = [self.config.scene_id]
        else:
            scenes = []
            # for partition_dir in sorted(self.config.data.iterdir()):
            # for subdir in sorted(partition_dir.iterdir()):
            for subdir in sorted(self.config.data.iterdir()):
                for scene_dir in sorted(subdir.iterdir()):
                    if not self.config.in_hdf5 or (scene_dir / "data.hdf5").exists():
                        scenes.append(f"{subdir.name}/{scene_dir.name}")
                    # scenes.append(f"{partition_dir.name}/{subdir.name}/{scene_dir.name}")

            if split == "train":
                scenes = scenes[:-self.config.eval_scene_count]
                scenes = scenes[rank::world_size]
                if chunk_index is not None:
                    scenes = scenes[chunk_index::num_chunks]
            else:
                scenes = scenes[-self.config.eval_scene_count:]

            if split != "train" and rank > 0:
                scenes = scenes[:1]

        image_filenames = []
        c2ws = []
        width = []
        height = []
        fx = []
        fy = []
        cx = []
        cy = []
        neighboring_views = []
        depth_paths = []

        min_bounds = None
        max_bounds = None
        near = 1e10
        far = -1
        for scene in tqdm(scenes):
            dest = self.config.data / scene
            transforms = dest / "transforms.json"

            if not transforms.exists():
                continue

            with transforms.open() as f:
                metadata = json.load(f)

            frames = metadata["frames"]
            assert len(frames) >= 4, len("frames")

            pose_scale_factor = 1
            scene_near = metadata["near"]
            if self.config.scale_near is not None:
                pose_scale_factor = scene_near / self.config.scale_near

            near = min(near, scene_near / pose_scale_factor)
            far = max(far, metadata["far"] / pose_scale_factor)
            current_scene_box = torch.FloatTensor([[-1, -1, -1], [1, 1, 1]]) / pose_scale_factor

            if min_bounds is None:
                min_bounds, max_bounds = current_scene_box
            else:
                min_bounds = torch.minimum(min_bounds, current_scene_box[0])
                max_bounds = torch.maximum(max_bounds, current_scene_box[1])

            scene_index_start = len(image_filenames)
            scene_index_end = scene_index_start + len(frames)

            for scene_image_index, frame in enumerate(frames):
                image_filenames.append(self.config.data / scene / frame["file_path"])
                c2w = torch.FloatTensor(frame["transform_matrix"])
                c2w[:3, 3] /= pose_scale_factor

                c2ws.append(c2w)
                width.append(frame["w"])
                height.append(frame["h"])
                fx.append(frame["fl_x"])
                fy.append(frame["fl_y"])
                cx.append(frame["cx"])
                cy.append(frame["cy"])
                depth_paths.append(self.config.data / scene / frame["depth_path"])

                neighboring_views.append(
                    list(range(scene_index_start, scene_index_start + scene_image_index))
                    + list(range(scene_index_start + scene_image_index + 1, scene_index_end)))

        c2ws = torch.stack(c2ws)
        width = torch.LongTensor(width)
        height = torch.LongTensor(height)
        fx = torch.FloatTensor(fx)
        fy = torch.FloatTensor(fy)
        cx = torch.FloatTensor(cx)
        cy = torch.FloatTensor(cy)

        CONSOLE.log(f"Derived bounds: {min_bounds} {max_bounds}")
        scene_box = SceneBox(aabb=1.1 * torch.stack([min_bounds, max_bounds]))

        if self.config.scene_id is not None:
            i_train, i_eval = get_train_eval_split_fraction(image_filenames, self.config.train_split_fraction)
            if split == "train":
                indices = i_train
            elif split in ["val", "test"]:
                indices = i_eval
            else:
                raise ValueError(f"Unknown dataparser split {split}")

            image_filenames = [image_filenames[i] for i in indices]

            #  Need to deal with indices changing post filtering if we actually want this
            neighboring_views = None  # [neighboring_views[i] for i in indices]

            idx_tensor = torch.LongTensor(indices)
            c2ws = c2ws[idx_tensor]
            width = width[idx_tensor]
            height = height[idx_tensor]
            fx = fx[idx_tensor]
            fy = fy[idx_tensor]
            cx = cx[idx_tensor]
            cy = cy[idx_tensor]

        cameras = Cameras(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            height=height,
            width=width,
            camera_to_worlds=c2ws[:, :3, :4],
            camera_type=CameraType.PERSPECTIVE,
        )

        metadata = {
            NEAR: near,
            FAR: far,
            POSENC_SCALE: 1 / (max_bounds - min_bounds).max(),
            DEPTH: depth_paths,
            IN_HDF5: self.config.in_hdf5,
        }

        if neighboring_views is not None:
            metadata[NEIGHBOR_INDICES] = neighboring_views

        # if (not get_default_scene) and self.config.scene_id is None:
        #     metadata[DEFAULT_SCENE_METADATA] = self._generate_dataparser_outputs(split, get_default_scene=True)

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            metadata=metadata
        )

        # if (not get_default_scene) and self.config.scene_id is None:
        #     torch.save(dataparser_outputs, cached_path)

        return dataparser_outputs
