from __future__ import annotations

import json
import traceback
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
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.data.utils.dataparsers_utils import get_train_eval_split_fraction
from nerfstudio.utils.rich_utils import CONSOLE

from generfstudio.generfstudio_constants import NEIGHBORING_VIEW_INDICES, NEAR, FAR, DEFAULT_SCENE_METADATA
from generfstudio.generfstudio_utils import central_crop_v2


@dataclass
class DL3DVDataParserConfig(DataParserConfig):
    """Scene dataset parser config. Assumes Pixel-NeRF format"""

    _target: Type = field(default_factory=lambda: DL3DV)
    """target class to instantiate"""
    data: Path = Path("data/dl3dv")
    """Directory specifying location of data."""

    scene_id: Optional[str] = None

    train_split_fraction: float = 0.9
    """The percentage of the dataset to use for training. Only used when using a single scene."""

    auto_orient: bool = False

    default_scene_id: str = "4K/ac41bb001ee989c3c0237341aa37f9f985e3e55b03cc70089ebffd938063bcdb"

    crop: Optional[Tuple[int, int]] = (256, 256)

    eval_scene_count: int = 50


@dataclass
class DL3DV(DataParser):
    """DL3DV Dataset"""

    config: DL3DVDataParserConfig

    def convert_scene_metadata(self, source: Path, dest: Path) -> None:
        # Assuming this is from 1080p version of DL3DV and not 4K

        inner = NerfstudioDataParserConfig(data=source, downscale_factor=4, load_3D_points=False).setup()
        inner_outputs = inner.get_dataparser_outputs("train")

        frames = []
        c2ws = []

        # near = 1e10
        # far = -1
        # points = inner_outputs.metadata["points3D_xyz"]
        # points = torch.cat([points, torch.ones_like(points[..., :1])], -1).T
        # 
        # c2ws = inner_outputs.cameras.camera_to_worlds
        # rot = c2ws[:, :3, :3].transpose(1, 2)  # (B, 3, 3)
        # trans = -torch.bmm(rot, c2ws[:, :3, 3:])  # (B, 3, 1)
        # w2cs = torch.cat((rot, trans), dim=-1)
        # 
        # points_in_camera = (w2cs @ points).transpose(1, 2)
        # depths = -points_in_camera[..., 2]

        for image_index, image_path in enumerate(inner_outputs.image_filenames):
            image = Image.open(image_path)
            camera = inner_outputs.cameras[image_index]
            K = camera.get_intrinsics_matrices().numpy()
            distortion_params = camera.distortion_params
            distortion_params = np.array(
                [distortion_params[0], distortion_params[1], distortion_params[4], distortion_params[5],
                 distortion_params[2], distortion_params[3], 0, 0, ])
            newK, roi = cv2.getOptimalNewCameraMatrix(K, distortion_params, image.size, 0)
            image = cv2.undistort(np.asarray(image), K, distortion_params, None, newK)
            x, y, w, h = roi
            image = Image.fromarray(image[y: y + h, x: x + w])

            image_fx = newK[0, 0]
            image_fy = newK[1, 1]
            image_cx = newK[0, 2]
            image_cy = newK[1, 2]

            if self.config.crop is not None:
                cropped_image = central_crop_v2(image)
                image_cx -= ((image.size[0] - cropped_image.size[0]) / 2)
                image_cy -= ((image.size[1] - cropped_image.size[1]) / 2)
                image_cx *= (self.config.crop[0] / cropped_image.size[0])
                image_cy *= (self.config.crop[1] / cropped_image.size[1])
                image_fx *= (self.config.crop[0] / cropped_image.size[0])
                image_fy *= (self.config.crop[1] / cropped_image.size[1])
                image = cropped_image.resize(self.config.crop, resample=Image.LANCZOS)

            image_path = dest / "images" / image_path.name
            image_path.parent.mkdir(exist_ok=True, parents=True)
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

            image_c2w = torch.eye(4)
            image_c2w[:3] = camera.camera_to_worlds
            c2ws.append(image_c2w)

        c2ws = torch.stack(c2ws)
        c2ws, _ = camera_utils.auto_orient_and_center_poses(
            c2ws,
            method="up",
            center_method="none",
        )

        for frame, c2w in zip(frames, c2ws):
            frame["transform_matrix"] = c2w.tolist()

        with (dest / "metadata.json").open("w") as f:
            json.dump({"frames": frames}, f, indent=4)

    def _generate_dataparser_outputs(self, split="train", get_default_scene=False):

        if get_default_scene:
            scenes = [self.config.default_scene_id]
        elif self.config.scene_id is not None:
            scenes = [self.config.scene_id]
        else:
            scenes = []
            for subdir in sorted((self.config.data / "original").iterdir()):
                for scene_dir in sorted(subdir.iterdir()):
                    scenes.append(f"{scene_dir.parent.name}/{scene_dir.name}")

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

        for scene in scenes:
            crop_dir = f"crop-{self.config.crop[0]}-{self.config.crop[1]}" if self.config.crop is not None else "crop-none"
            dest = self.config.data / crop_dir / scene
            converted_metadata_path = dest / "metadata.json"
            if not converted_metadata_path.exists():
                try:
                    self.convert_scene_metadata(self.config.data / "original" / scene, dest)
                except:
                    traceback.print_exc()
                    # import pdb; pdb.set_trace()
                    continue

            with converted_metadata_path.open() as f:
                frames = json.load(f)["frames"]

            scene_index_start = len(image_filenames)
            scene_index_end = scene_index_start + len(frames)
            for scene_image_index, frame in enumerate(frames):
                image_filenames.append(frame["file_path"])
                c2ws.append(torch.FloatTensor(frame["transform_matrix"]).unsqueeze(0))
                width.append(frame["w"])
                height.append(frame["h"])
                fx.append(frame["fl_x"])
                fy.append(frame["fl_y"])
                cx.append(frame["cx"])
                cy.append(frame["cy"])

                neighboring_views.append(
                    list(range(scene_index_start, scene_index_start + scene_image_index))
                    + list(range(scene_index_start + scene_image_index + 1, scene_index_end)))

        c2ws = torch.cat(c2ws)
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
            NEAR: 0.01,
            FAR: 5,
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
