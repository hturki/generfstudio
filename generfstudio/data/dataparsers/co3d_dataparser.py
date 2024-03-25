from __future__ import annotations

import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import List
from typing import Type, Optional, Tuple

import math
import torch
from PIL import Image
from co3d.dataset.data_types import (
    load_dataclass_jgzip, FrameAnnotation
)
from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import DataParser, DataParserConfig, DataparserOutputs
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.data.utils.dataparsers_utils import get_train_eval_split_fraction
from nerfstudio.utils.rich_utils import CONSOLE
from tqdm import tqdm

from generfstudio.generfstudio_constants import NEIGHBORING_VIEW_INDICES, NEAR, FAR, DEFAULT_SCENE_METADATA, \
    POSENC_SCALE
from generfstudio.generfstudio_utils import central_crop_v2

PYTORCH_3D_TO_OPEN_CV = torch.diag(torch.FloatTensor([-1, -1, 1, 1]))

GROUND_PLANE_Z = torch.FloatTensor([[1, 0, 0, 0],
                                    [0, 0, 1, 0],
                                    [0, -1, 0, 0],
                                    [0, 0, 0, 1]])


@dataclass
class CO3DDataParserConfig(DataParserConfig):
    _target: Type = field(default_factory=lambda: CO3D)
    """target class to instantiate"""
    data: Path = Path("data/co3d")
    """Directory specifying location of data."""

    scene_id: Optional[str] = None

    train_split_fraction: float = 0.9
    """The percentage of the dataset to use for training. Only used when using a single scene."""

    default_scene_id: str = "hydrant/437_62405_122559"

    crop: Optional[int] = 256

    scale_near: Optional[float] = None

    eval_sequences_per_category: int = 1

    conversion_threads: int = 16


# Note - source does not include category or sequence, since annotations are of the form:
# FrameAnnotation(sequence_name='540_79043_153212', frame_number=6, frame_timestamp=0.3308457711442786,
#                 image=ImageAnnotation(path='apple/540_79043_153212/images/frame000006.jpg', size=(900, 2000)),
#                 depth=DepthAnnotation(path='apple/540_79043_153212/depths/frame000006.jpg.geometric.png',
#                                       scale_adjustment=1.0,
#                                       mask_path='apple/540_79043_153212/depth_masks/frame000006.png'),
#                 mask=MaskAnnotation(path='apple/540_79043_153212/masks/frame000006.png', mass=1000000),
#                 viewpoint=ViewpointAnnotation(R=((-0.9941162467002869, -0.10805200785398483, 0.007596105802804232),
#                                                  (0.10785271227359772, -0.9939004778862, -0.02301282249391079),
#                                                  (0.010036354884505272, -0.022058159112930298, 0.9997063279151917)),
#                                               T=(-0.35470789670944214, 1.2838153839111328, 14.110940933227539),
#                                               focal_length=(3.959122657775879, 3.9611024856567383),
#                                               principal_point=(-0.0, -0.0), intrinsics_format='ndc_isotropic'),
#                 meta={'frame_type': 'test_unseen', 'frame_splits': ['singlesequence_apple_test_0_unseen'],
#                       'eval_batch_maps': []})
def convert_scene_metadata(source: Path, dest: Path, frame_annotations: List[FrameAnnotation],
                           crop: Optional[int]) -> None:
    frames = []

    for frame_annotation in sorted(frame_annotations, key=lambda x: x.frame_number):
        # Convert from pytorch 3d to OpenCV. Inspired by:
        # https://github.com/facebookresearch/pytorch3d/blob/7566530669203769783c94024c25a39e1744e4ed/pytorch3d/renderer/camera_conversions.py#L64
        R_pytorch3d = torch.FloatTensor(frame_annotation.viewpoint.R)
        T_pytorch3d = torch.FloatTensor(frame_annotation.viewpoint.T)
        focal_pytorch3d = torch.FloatTensor(frame_annotation.viewpoint.focal_length)
        p0_pytorch3d = torch.FloatTensor(frame_annotation.viewpoint.principal_point)

        image_c2w = torch.eye(4)
        image_c2w[:3, :3] = R_pytorch3d
        image_c2w[:3, 3:] = -R_pytorch3d @ T_pytorch3d[..., None]
        image_c2w = image_c2w @ PYTORCH_3D_TO_OPEN_CV
        # opencv -> opengl
        image_c2w[:, 1:3] *= -1

        # Retype the image_size correctly and flip to width, height.
        image_size_wh = torch.IntTensor(frame_annotation.image.size).flip(dims=(0,))

        # NDC to screen conversion.
        scale = image_size_wh.min() / 2.0
        c0 = image_size_wh / 2.0

        image_cx, image_cy = -p0_pytorch3d * scale + c0
        image_fx, image_fy = focal_pytorch3d * scale

        image_path = source / frame_annotation.image.path
        if crop is not None:
            image = Image.open(image_path)
            cropped_image = central_crop_v2(image)
            image_cx -= ((image.size[0] - cropped_image.size[0]) / 2)
            image_cy -= ((image.size[1] - cropped_image.size[1]) / 2)
            image_cx *= (crop / cropped_image.size[0])
            image_cy *= (crop / cropped_image.size[1])
            image_fx *= (crop / cropped_image.size[0])
            image_fy *= (crop / cropped_image.size[1])

            image = cropped_image.resize((crop, crop), resample=Image.LANCZOS)
            image_path = dest / "images" / image_path.name
            image_path.parent.mkdir(exist_ok=True, parents=True)
            image.save(image_path)
            w, h = crop
        else:
            w, h = image_size_wh

        frames.append({
            "file_path": str(image_path),
            "transform_matrix": image_c2w.tolist(),
            "w": int(w),
            "h": int(h),
            "fl_x": image_fx.item(),
            "fl_y": image_fy.item(),
            "cx": image_cx.item(),
            "cy": image_cy.item(),
        })

    dest.mkdir(exist_ok=True, parents=True)
    with (dest / "metadata.json").open("w") as f:
        json.dump({"frames": frames}, f, indent=4)


@dataclass
class CO3D(DataParser):
    """CO3D Dataset"""

    config: CO3DDataParserConfig

    def _generate_dataparser_outputs(self, split="train", get_default_scene=False):
        cached_path = self.config.data / f"cached-metadata-{split}-{self.config.crop}-{self.config.scale_near}.pt"
        if (not get_default_scene) and self.config.scene_id is None and cached_path.exists():
            return torch.load(cached_path)

        if get_default_scene:
            scenes = [self.config.default_scene_id]
        elif self.config.scene_id is not None:
            scenes = [self.config.scene_id]
        else:
            scenes = []
            to_scan = self.config.data / "original"
            if not to_scan.exists():
                to_scan = self.config.data / f"crop-{self.config.crop}"
                CONSOLE.log(f"Original dir not found, using {to_scan} instead")
            for category_dir in sorted(to_scan.iterdir()):
                if category_dir.is_dir():
                    sequence_dirs = list(filter(lambda x: x.is_dir() and x.name not in {"eval_batches", "set_lists"},
                                                sorted(category_dir.iterdir())))
                    if split == "train":
                        sequence_dirs = sequence_dirs[:-self.config.eval_sequences_per_category]
                    else:
                        sequence_dirs = sequence_dirs[-self.config.eval_sequences_per_category:]

                    for sequence_dir in sequence_dirs:
                        scenes.append(f"{sequence_dir.parent.name}/{sequence_dir.name}")

        image_filenames = []
        c2ws = []
        width = []
        height = []
        fx = []
        fy = []
        cx = []
        cy = []
        neighboring_views = []

        loaded_frame_annotations_category = None
        loaded_frame_annotations = None

        min_near = 1e10
        max_far = -1
        to_convert = []
        with ThreadPoolExecutor(max_workers=self.config.conversion_threads) as executor:
            for scene in tqdm(scenes):
                dest = self.config.data / f"crop-{self.config.crop}" / scene
                converted_metadata_path = dest / "metadata.json"
                if not converted_metadata_path.exists():
                    category = dest.parent.name
                    if loaded_frame_annotations_category != category:
                        loaded_frame_annotations_flat = load_dataclass_jgzip(
                            f"{self.config.data}/original/{category}/frame_annotations.jgz", List[FrameAnnotation]
                        )
                        loaded_frame_annotations = defaultdict(list)
                        for annotation in loaded_frame_annotations_flat:
                            loaded_frame_annotations[annotation.sequence_name].append(annotation)

                        loaded_frame_annotations_category = category
                    if not dest.name in loaded_frame_annotations:
                        continue

                    to_convert.append(executor.submit(convert_scene_metadata, self.config.data / "original", dest,
                                                      loaded_frame_annotations[dest.name], self.config.crop))

            for task in tqdm(to_convert):
                task.result()

        bounds_skipped = 0
        for scene in tqdm(scenes):
            dest = self.config.data / f"crop-{self.config.crop}" / scene
            converted_metadata_path = dest / "metadata.json"

            if not converted_metadata_path.exists():
                CONSOLE.log(f"Skipping {converted_metadata_path}")
                continue

            with converted_metadata_path.open() as f:
                frames = json.load(f)["frames"]

            if len(frames) < 4:
                CONSOLE.log(f"Skipping {scene} as it has less than 3 frames")
                continue

            scene_c2ws = []
            for scene_image_index, frame in enumerate(frames):
                c2w = torch.FloatTensor(frame["transform_matrix"])
                scene_c2ws.append(c2w)

            scene_c2ws = torch.stack(scene_c2ws)
            scene_c2ws, transform = camera_utils.auto_orient_and_center_poses(
                scene_c2ws,
                method="up",
                center_method="none",
            )

            # https://github.com/facebookresearch/co3d/issues/18
            cam_distances = scene_c2ws[:, :3, 3].norm(dim=-1)
            min_cam_distance = cam_distances.min()
            max_cam_distance = cam_distances.max()

            if min_cam_distance < 1 or max_cam_distance > 100:
                CONSOLE.log(
                    f"Skipping {scene} as camera distances seem incorrect {min_cam_distance} {max_cam_distance}")
                bounds_skipped += 1
                continue

            c2ws.append(scene_c2ws)

            scene_index_start = len(image_filenames)
            scene_index_end = scene_index_start + len(frames)
            for scene_image_index, frame in enumerate(frames):
                image_filenames.append(frame["file_path"])
                width.append(frame["w"])
                height.append(frame["h"])
                fx.append(frame["fl_x"])
                fy.append(frame["fl_y"])
                cx.append(frame["cx"])
                cy.append(frame["cy"])

                neighboring_views.append(
                    list(range(scene_index_start, scene_index_start + scene_image_index))
                    + list(range(scene_index_start + scene_image_index + 1, scene_index_end)))

            near = (min_cam_distance - 8)
            near = max(near, 0.01)  # Some scenes seem to be closer than 8

            if self.config.scale_near is not None:
                pose_scale_factor = near / self.config.scale_near
                scene_c2ws[:, :3, 3] /= pose_scale_factor
                near = self.config.scale_near
            else:
                pose_scale_factor = 1

            min_near = min(near, min_near)
            max_far = max(max_cam_distance + 8 / pose_scale_factor, max_far)

        c2ws = torch.cat(c2ws)
        width = torch.LongTensor(width)
        height = torch.LongTensor(height)
        fx = torch.FloatTensor(fx)
        fy = torch.FloatTensor(fy)
        cx = torch.FloatTensor(cx)
        cy = torch.FloatTensor(cy)

        scene_box = SceneBox(aabb=torch.FloatTensor([[-max_far, -max_far, -max_far], [max_far, max_far, max_far]]))

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

        CONSOLE.log(f"Skipped {bounds_skipped} out of {len(scenes)} scenes due to camera bounds")

        metadata = {
            NEAR: min_near,
            FAR: max_far,
            POSENC_SCALE: 1 / max_far,
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

        if (not get_default_scene) and self.config.scene_id is None:
            torch.save(dataparser_outputs, cached_path)

        return dataparser_outputs
