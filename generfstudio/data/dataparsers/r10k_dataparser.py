from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Type, Optional

import torch
from PIL import Image
from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import DataParser, DataParserConfig, DataparserOutputs
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.data.utils.dataparsers_utils import get_train_eval_split_fraction
from nerfstudio.utils.comms import get_rank, get_world_size
from nerfstudio.utils.rich_utils import CONSOLE
from tqdm import tqdm

from generfstudio.generfstudio_constants import NEIGHBOR_INDICES, NEAR, FAR, POSENC_SCALE
from generfstudio.generfstudio_utils import central_crop_v2


@dataclass
class R10KDataParserConfig(DataParserConfig):
    """Scene dataset parser config. Assumes Pixel-NeRF format"""

    _target: Type = field(default_factory=lambda: R10K)
    """target class to instantiate"""
    data: Path = Path("data/acid")
    """Directory specifying location of data."""

    scene_id: Optional[str] = None  # "train/2ecba9802673be71"

    train_split_fraction: float = 0.9
    """The percentage of the dataset to use for training. Only used when using a single scene."""

    eval_scene_count: int = 50

    default_scene_id: str = "train/3f327a802a01c44b"

    crop: Optional[int] = 256

    scale_near: Optional[float] = None

    conversion_threads: int = 16

    neighbor_window_size: Optional[int] = 20


def convert_scene_metadata(scene: str, source: Path, dest: Path, data, crop: Optional[int]) -> bool:
    w2cs = []
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
            image_fx = image_fx * image.size[0]
            image_fy = image_fy * image.size[1]
            image_cx = image_cx * image.size[0]
            image_cy = image_cy * image.size[1]

            if crop is not None:
                image_path = data / f"crop-{crop}" / scene / image_path.name
                image_path.parent.mkdir(exist_ok=True, parents=True)
                cropped_image = central_crop_v2(image)
                image_cx -= ((image.size[0] - cropped_image.size[0]) / 2)
                image_cy -= ((image.size[1] - cropped_image.size[1]) / 2)
                image_cx *= (crop / cropped_image.size[0])
                image_cy *= (crop / cropped_image.size[1])
                image_fx *= (crop / cropped_image.size[0])
                image_fy *= (crop / cropped_image.size[1])
                image = cropped_image.resize((crop, crop), resample=Image.LANCZOS)
                image.save(image_path)

            frames.append({
                "file_path": str(image_path),
                "w": image.size[0],
                "h": image.size[1],
                "fl_x": image_fx,
                "fl_y": image_fy,
                "cx": image_cx,
                "cy": image_cy,
            })

            image_w2c = torch.eye(4)
            image_w2c[:3] = torch.FloatTensor([float(x) for x in entry[7:]]).view(3, 4)
            w2cs.append(image_w2c)

    w2cs = torch.stack(w2cs)

    rot = w2cs[:, :3, :3].transpose(1, 2)  # (B, 3, 3)
    trans = -torch.bmm(rot, w2cs[:, :3, 3:])  # (B, 3, 1)
    c2ws = torch.cat((rot, trans), dim=-1)
    # opencv -> opengl
    c2ws[:, :, 1:3] *= -1

    c2ws_4x4 = torch.eye(4).unsqueeze(0).repeat(len(c2ws), 1, 1)
    c2ws_4x4[:, :3] = c2ws

    c2ws, transform = camera_utils.auto_orient_and_center_poses(
        c2ws_4x4,
        method="up",
        center_method="none",
    )

    for frame, c2w in zip(frames, c2ws):
        frame["transform_matrix"] = c2w.tolist()

    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w") as f:
        json.dump({"frames": frames}, f, indent=4)

    return True


@dataclass
class R10K(DataParser):
    """R10K Dataset"""

    config: R10KDataParserConfig

    def _generate_dataparser_outputs(self, split="train", chunk_index=None, num_chunks=None, get_default_scene=False):
        rank = get_rank()
        world_size = get_world_size()
        window_size = self.config.neighbor_window_size
        cached_path = self.config.data / f"cached-metadata-{split}-{self.config.crop}-{self.config.scale_near}-{window_size}-{rank}-{world_size}-{chunk_index}-{num_chunks}.pt"
        if (not get_default_scene) and self.config.scene_id is None and cached_path.exists():
            return torch.load(cached_path)

        if get_default_scene:
            scenes = [self.config.default_scene_id]
        elif self.config.scene_id is not None:
            scenes = [self.config.scene_id]
        else:
            scenes = []
            check_for_source = (self.config.data / "train").exists()
            if not check_for_source:
                CONSOLE.log(f"Original dir not found")

            for dir in ["train", "test", "validation"]:
                if (self.config.data / "metadata" / dir).exists():
                    for file in sorted((self.config.data / "metadata" / dir).iterdir()):
                        scene = f"{dir}/{file.stem}"
                        if (not check_for_source) or (self.config.data / scene).exists():
                            scenes.append(scene)

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

        to_convert = []
        with ThreadPoolExecutor(max_workers=self.config.conversion_threads) as executor:
            for scene in scenes:
                metadata_name = f"{scene}-{self.config.crop}.json" if self.config.crop is not None else f"{scene}.json"
                nerfstudio_metadata_path = self.config.data / "nerfstudio" / metadata_name

                if not nerfstudio_metadata_path.exists():
                    to_convert.append(executor.submit(convert_scene_metadata, scene,
                                                      self.config.data / "metadata" / f"{scene}.txt",
                                                      nerfstudio_metadata_path, self.config.data,
                                                      self.config.crop))
            for task in tqdm(to_convert):
                task.result()

        for scene in tqdm(scenes):
            metadata_name = f"{scene}-{self.config.crop}.json" if self.config.crop is not None else f"{scene}.json"
            nerfstudio_metadata_path = self.config.data / "nerfstudio" / metadata_name

            if not nerfstudio_metadata_path.exists():
                continue

            with nerfstudio_metadata_path.open() as f:
                frames = json.load(f)["frames"]

            if len(frames) < 4:
                continue

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

                if window_size is not None:
                    neighboring_views.append(
                        list(range(scene_index_start + max(0, scene_image_index - window_size),
                                   scene_index_start + scene_image_index))
                        + list(range(scene_index_start + scene_image_index + 1,
                                     min(scene_index_end, scene_index_start + scene_image_index + window_size + 1))))
                else:
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

        if self.config.scale_near is not None:
            # Apparently R10K is scaled such that 1 is a reasonable near bound
            pose_scale_factor = 1 / self.config.scale_near
            c2ws[:, :3, 3] /= pose_scale_factor
        else:
            pose_scale_factor = 1

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
            NEAR: 1 / pose_scale_factor,
            FAR: 200 / pose_scale_factor,
            POSENC_SCALE: pose_scale_factor / 200,
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
