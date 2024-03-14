from __future__ import annotations

import json
import shutil
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Type, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import DataParser, DataParserConfig, DataparserOutputs
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.data.utils.dataparsers_utils import get_train_eval_split_fraction
from nerfstudio.utils.rich_utils import CONSOLE
from tqdm import tqdm

from generfstudio.generfstudio_constants import NEIGHBORING_VIEW_INDICES, NEAR, FAR, DEFAULT_SCENE_METADATA
from generfstudio.generfstudio_utils import central_crop_v2


@dataclass
class R10KDataParserConfig(DataParserConfig):
    """Scene dataset parser config. Assumes Pixel-NeRF format"""

    _target: Type = field(default_factory=lambda: R10K)
    """target class to instantiate"""
    data: Path = Path("data/acid")
    """Directory specifying location of data."""

    scene_id: Optional[str] = None  # "train/3f327a802a01c44b"

    train_split_fraction: float = 0.9
    """The percentage of the dataset to use for training. Only used when using a single scene."""

    eval_scene_count: int = 50

    default_scene_id: str = "train/3f327a802a01c44b"

    crop: Optional[Tuple[int, int]] = (256, 256)

    conversion_threads: int = 16


def convert_scene_metadata(scene: str, source: Path, dest: Path, data, crop):
    c2ws = []

    if dest.exists():
        with dest.open() as f:
            frames = json.load(f)["frames"]

        with source.open() as f:
            url = f.readline()

            for line in f:
                entry = line.strip().split()
                image_c2w = torch.eye(4)
                image_c2w[:3] = torch.FloatTensor([float(x) for x in entry[7:]]).view(3, 4)
                # opencv -> opengl
                image_c2w[:, 1:3] *= -1
                c2ws.append(image_c2w)
    else:
        frames = []

        with source.open() as f:
            url = f.readline()

            for line in f:
                entry = line.strip().split()
                timestamp = int(entry[0])
                image_path = data / scene / f"{timestamp}.png"
                if not image_path.exists():
                    return False
                image = Image.open(image_path)

                image_fx, image_fy, image_cx, image_cy = [float(x) for x in entry[1:5]]
                image_cx = image_cx * image.size[0]
                image_cy = image_cy * image.size[1]

                if crop is not None:
                    image_path = data / f"crop-{crop[0]}-{crop[1]}" / scene / image_path.name
                    image_path.parent.mkdir(exist_ok=True, parents=True)
                    cropped_image = central_crop_v2(image)
                    image_cx -= ((image.size[0] - cropped_image.size[0]) / 2)
                    image_cy -= ((image.size[1] - cropped_image.size[1]) / 2)
                    image_cx *= (crop[0] / cropped_image.size[0])
                    image_cy *= (crop[1] / cropped_image.size[1])
                    cropped_image.resize(crop, resample=Image.LANCZOS).save(image_path)

                frames.append({
                    "file_path": str(image_path),
                    "w": image.size[0],
                    "h": image.size[1],
                    "fl_x": image_fx * image.size[0],
                    "fl_y": image_fy * image.size[1],
                    "cx": image_cx,
                    "cy": image_cy,
                })

                image_c2w = torch.eye(4)
                image_c2w[:3] = torch.FloatTensor([float(x) for x in entry[7:]]).view(3, 4)
                # opencv -> opengl
                image_c2w[:, 1:3] *= -1
                c2ws.append(image_c2w)

    c2ws = torch.stack(c2ws)
    try:
        c2ws, transform = camera_utils.auto_orient_and_center_poses(
            c2ws,
            method="up",
            center_method="none",
        )
    except:
        import pdb; pdb.set_trace()
    # min_bounds = c2ws[:, :, 3].min(dim=0)[0]
    # max_bounds = c2ws[:, :, 3].max(dim=0)[0]
    #
    # origin = (max_bounds + min_bounds) * 0.5
    # CONSOLE.print(f"Calculated origin for {scene}: {origin} {min_bounds} {max_bounds}")
    #
    # pose_scale_factor = ((max_bounds - min_bounds) * 0.5).norm().item()
    # CONSOLE.print(f"Calculated pose scale factor for {scene}: {pose_scale_factor}")

    for frame, c2w in zip(frames, c2ws):
        # c2w[:, 3] = (c2w[:, 3] - origin) / pose_scale_factor
        # assert torch.logical_and(c2w >= -1, c2w <= 1).all(), c2w
        frame["transform_matrix"] = c2w.tolist()

    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w") as f:
        json.dump({"frames": frames}, f, indent=4)

    return True


@dataclass
class R10K(DataParser):
    """R10K Dataset"""

    config: R10KDataParserConfig

    def _generate_dataparser_outputs(self, split="train", get_default_scene=False):
        if get_default_scene:
            scenes = [self.config.default_scene_id]
        elif self.config.scene_id is not None:
            scenes = [self.config.scene_id]
        else:
            scenes = []
            for dir in ["train", "test", "validation"]:
                if (self.config.data / "metadata" / dir).exists():
                    for file in sorted((self.config.data / "metadata" / dir).iterdir()):
                        scene = f"{dir}/{file.stem}"
                        if (self.config.data / scene).exists():
                            scenes.append(scene)

            if split == "train":
                scenes = scenes[:-self.config.eval_scene_count]
            else:
                scenes = scenes[-self.config.eval_scene_count:]

        image_filenames = []
        c2ws = []
        width = []
        height = []
        fx = []
        fy = []
        cx = []
        cy = []
        neighboring_views = []

        converting_scenes = {}
        with ThreadPoolExecutor(max_workers=self.config.conversion_threads) as executor:
            for scene in scenes:
                metadata_name = f"{scene}-{self.config.crop[0]}-{self.config.crop[1]}.json" \
                    if self.config.crop is not None else f"{scene}.json"
                nerfstudio_metadata_path = self.config.data / "nerfstudio" / metadata_name
                if not nerfstudio_metadata_path.exists():
                    converting_scenes[scene] = executor.submit(convert_scene_metadata, scene,
                                                               self.config.data / "metadata" / f"{scene}.txt",
                                                               nerfstudio_metadata_path, self.config.data,
                                                               self.config.crop)

            for scene in tqdm(scenes):
                metadata_name = f"{scene}-{self.config.crop[0]}-{self.config.crop[1]}.json" \
                    if self.config.crop is not None else f"{scene}.json"
                nerfstudio_metadata_path = self.config.data / "nerfstudio" / metadata_name
                if scene in converting_scenes:
                    try:
                        if not converting_scenes[scene].result():
                            continue
                    except:
                        traceback.print_exc()
                        import pdb; pdb.set_trace()

                with nerfstudio_metadata_path.open() as f:
                    frames = json.load(f)["frames"]

                scene_index_start = len(image_filenames)
                scene_index_end = scene_index_start + len(frames)
                for scene_image_index, frame in enumerate(frames):
                    image_filenames.append(frame["file_path"])
                    c2ws.append(torch.FloatTensor(frame["transform_matrix"]))
                    width.append(frame["w"])
                    height.append(frame["h"])
                    fx.append(frame["fl_x"])
                    fy.append(frame["fl_y"])
                    cx.append(frame["cx"])
                    cy.append(frame["cy"])

                    neighboring_views.append(
                        list(range(scene_index_start, scene_index_start + scene_image_index))
                        + list(range(scene_index_start + scene_image_index + 1, scene_index_end)))

        # if len(missing_scenes) > 0:
        #     for scene in missing_scenes:
        #         dest = self.config.data / "missing-metadata" / f"{scene}.txt"
        #         dest.parent.mkdir(exist_ok=True, parents=True)
        #         shutil.copy(self.config.data / "metadata" / f"{scene}.txt", dest)
        #     raise Exception()

        c2ws = torch.stack(c2ws)
        width = torch.LongTensor(width)
        height = torch.LongTensor(height)
        fx = torch.FloatTensor(fx)
        fy = torch.FloatTensor(fy)
        cx = torch.FloatTensor(cx)
        cy = torch.FloatTensor(cy)

        min_bounds = c2ws[:, :3, 3].min(dim=0)[0]
        max_bounds = c2ws[:, :3, 3].max(dim=0)[0]
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
            NEAR: 1,
            FAR: 200,
        }

        if neighboring_views is not None:
            metadata[NEIGHBORING_VIEW_INDICES] = neighboring_views

        if (not get_default_scene) and self.config.scene_id is None:
            metadata[DEFAULT_SCENE_METADATA] = self._generate_dataparser_outputs(split, get_default_scene=True)

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            metadata=metadata
        )

        return dataparser_outputs
