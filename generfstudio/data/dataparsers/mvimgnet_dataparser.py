from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Type, Optional, Tuple

import torch
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import DataParser, DataParserConfig, DataparserOutputs
from nerfstudio.data.dataparsers.colmap_dataparser import ColmapDataParserConfig
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.data.utils.dataparsers_utils import get_train_eval_split_fraction
from nerfstudio.utils.comms import get_rank, get_world_size
from nerfstudio.utils.rich_utils import CONSOLE
from tqdm import tqdm

from generfstudio.data.dataparsers.dataparser_utils import convert_from_inner, get_xyz_in_camera
from generfstudio.generfstudio_constants import NEIGHBOR_INDICES, NEAR, FAR, POSENC_SCALE


@dataclass
class MVImgNetDataParserConfig(DataParserConfig):
    _target: Type = field(default_factory=lambda: MVImgNet)
    """target class to instantiate"""
    data: Path = Path("data/MVImgNet")
    """Directory specifying location of data."""

    scene_id: Optional[str] = None

    train_split_fraction: float = 0.9
    """The percentage of the dataset to use for training. Only used when using a single scene."""

    default_scene_id: str = "0/0000be4b"

    crop: Optional[int] = 256

    scale_near: Optional[float] = None

    eval_sequences_per_category: int = 1

    conversion_threads: int = 64


def convert_scene_metadata(source: Path, dest: Path, crop: Optional[Tuple[int, int]]) -> None:
    inner = ColmapDataParserConfig(data=source, downscale_factor=1, colmap_path=Path("sparse/0"),
                                   load_3D_points=True, eval_mode="all").setup()
    inner_outputs = inner.get_dataparser_outputs("train")
    frames = convert_from_inner(inner_outputs, dest, crop)

    xyz_in_camera, uv, is_valid_points, width, height = get_xyz_in_camera(inner_outputs, frames)

    depth = -xyz_in_camera[..., 2][is_valid_points]

    # The inner data parser already orients the poses
    # c2ws, transform = camera_utils.auto_orient_and_center_poses(
    #     c2ws,
    #     method="up",
    #     center_method="none",
    # )

    points = inner_outputs.metadata["points3D_xyz"]
    min_bounds = points.min(dim=0)[0]
    max_bounds = points.max(dim=0)[0]

    with (dest / "metadata.json").open("w") as f:
        json.dump({"frames": frames, "near": depth.min().item(), "far": depth.max().item(),
                   "scene_box": torch.stack([min_bounds, max_bounds]).tolist()}, f, indent=4)


@dataclass
class MVImgNet(DataParser):
    """MVImgNet Dataset"""

    config: MVImgNetDataParserConfig

    def _generate_dataparser_outputs(self, split="train", chunk_index=None, num_chunks=None, get_default_scene=False):
        rank = get_rank()
        world_size = get_world_size()
        cached_path = self.config.data / f"cached-metadata-{split}-{self.config.crop}-{self.config.scale_near}-{rank}-{world_size}-{chunk_index}-{num_chunks}.pt"
        if (not get_default_scene) and self.config.scene_id is None and cached_path.exists():
            return torch.load(cached_path)

        if get_default_scene:
            scenes = [self.config.default_scene_id]
        elif self.config.scene_id is not None:
            scenes = [self.config.scene_id]
        else:
            scenes = []
            to_scan = self.config.data / "original"
            check_for_colmap = True
            if not to_scan.exists():
                to_scan = self.config.data / f"crop-{self.config.crop}"
                check_for_colmap = False
                CONSOLE.log(f"Original dir not found, using {to_scan} instead")
            for category_dir in sorted(to_scan.iterdir()):
                sequence_dirs = []
                for candidate in sorted(category_dir.iterdir()):
                    if candidate.parent.name == "206" and candidate.name == "43013265":
                        continue
                    if candidate.parent.name == "227" and candidate.name == "0a005e03":
                        continue
                    colmap_results = (candidate / "sparse" / "0")
                    if check_for_colmap and (not colmap_results.exists()):
                        CONSOLE.log(f"Skipping {candidate} ({colmap_results} not found)")
                        continue
                    sequence_dirs.append(candidate)

                if split == "train":
                    sequence_dirs = sequence_dirs[:-self.config.eval_sequences_per_category]
                    sequence_dirs = sequence_dirs[rank::world_size]
                    if chunk_index is not None:
                        sequence_dirs = sequence_dirs[chunk_index::num_chunks]
                else:
                    sequence_dirs = sequence_dirs[-self.config.eval_sequences_per_category:]

                for sequence_dir in sequence_dirs:
                    scenes.append(f"{sequence_dir.parent.name}/{sequence_dir.name}")

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

        to_convert = []
        with ProcessPoolExecutor(max_workers=self.config.conversion_threads) as executor:
            for scene in tqdm(scenes):
                dest = self.config.data / f"crop-{self.config.crop}" / scene
                converted_metadata_path = dest / "metadata.json"

                if not converted_metadata_path.exists():
                    to_convert.append(
                        executor.submit(convert_scene_metadata, self.config.data / "original" / scene, dest,
                                        self.config.crop))

            for task in tqdm(to_convert):
                task.result()

        min_bounds = None
        max_bounds = None
        near = 1e10
        far = -1
        for scene in tqdm(scenes):
            dest = self.config.data / f"crop-{self.config.crop}" / scene
            converted_metadata_path = dest / "metadata.json"

            with converted_metadata_path.open() as f:
                metadata = json.load(f)

            frames = metadata["frames"]

            if len(frames) < 4:
                CONSOLE.log(f"Skipping {scene} as it has less than 3 frames")
                continue

            scene_near = max(metadata["near"], 0.01)
            if self.config.scale_near is not None:
                pose_scale_factor = scene_near / self.config.scale_near
                near = self.config.scale_near
            else:
                pose_scale_factor = 1

            near = min(near, scene_near)
            far = max(far, metadata["far"] / pose_scale_factor)
            current_scene_box = torch.FloatTensor(metadata["scene_box"]) / pose_scale_factor
            if min_bounds is None:
                min_bounds, max_bounds = current_scene_box
            else:
                min_bounds = torch.minimum(min_bounds, current_scene_box[0])
                max_bounds = torch.maximum(max_bounds, current_scene_box[1])

            scene_index_start = len(image_filenames)
            scene_index_end = scene_index_start + len(frames)

            for scene_image_index, frame in enumerate(frames):
                image_filenames.append(frame["file_path"])
                c2w = torch.FloatTensor(frame["transform_matrix"])
                c2w[:3, 3] /= pose_scale_factor
                c2ws.append(c2w)
                width.append(frame["w"])
                height.append(frame["h"])
                fx.append(frame["fl_x"])
                fy.append(frame["fl_y"])
                cx.append(frame["cx"])
                cy.append(frame["cy"])

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
            POSENC_SCALE: 1 / far,
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

        if (not get_default_scene) and self.config.scene_id is None:
            torch.save(dataparser_outputs, cached_path)

        return dataparser_outputs
