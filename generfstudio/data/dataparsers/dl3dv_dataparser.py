from __future__ import annotations

import json
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Type, Optional, Tuple

import numpy as np
import torch
from dust3r.model import AsymmetricCroCo3DStereo
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import DataParser, DataParserConfig, DataparserOutputs
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.data.utils.dataparsers_utils import get_train_eval_split_fraction
from nerfstudio.utils.comms import get_rank, get_world_size
from nerfstudio.utils.rich_utils import CONSOLE
from torch_scatter import scatter_min
from tqdm import tqdm

from generfstudio.data.dataparsers.dataparser_utils import convert_from_inner, get_xyz_in_camera, create_dust3r_poses
from generfstudio.data.dataparsers.gaussian_optimizer import GaussianOptimizer
from generfstudio.generfstudio_constants import NEIGHBOR_INDICES, NEAR, FAR, POSENC_SCALE, DEPTH


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

    crop: Optional[int] = 256

    scale_near: Optional[float] = None

    eval_scene_count: int = 50

    neighbor_overlap_threshold: Optional[Tuple[float, float]] = (0.3, 0.5)

    conversion_threads: int = 96

    use_dust3r_poses: bool = True

    dust3r_model_name: str = "/data/hturki/dust3r/checkpoints/DUSt3R_ViTLarge_BaseDecoder_224_linear.pth"
    # dust3r_model_name: str = "/project_data/ramanan/hturki/dust3r/checkpoints/DUSt3R_ViTLarge_BaseDecoder_224_linear.pth"

    skip_dust3r_incomplete: bool = False

    dust3r_use_known_cameras: bool = False

    dust3r_num_neighbors: int = 10

    dust3r_batch_size: int = 10000

    dust3r_chunk_size: Optional[int] = None  # 12544

    num_workers: Optional[int] = None

    worker_offset: int = 0


def convert_scene_metadata(source: Path, dest: Path, crop: Optional[int]) -> None:
    # Assuming this is from 480p version of DL3DV and not 4K
    inner = NerfstudioDataParserConfig(data=source, downscale_factor=8, load_3D_points=True, eval_mode="all").setup()
    inner_outputs = inner.get_dataparser_outputs("train")
    frames = convert_from_inner(inner_outputs, dest, crop)

    # The inner data parser already orients the poses
    # c2ws, transform = camera_utils.auto_orient_and_center_poses(
    #     c2ws,
    #     method="up",
    #     center_method="none",
    # )

    metadata = {"frames": frames}

    if "points3D_xyz" in inner_outputs.metadata and inner_outputs.metadata["points3D_xyz"].shape[0] > 0:
        xyz_in_camera, uv, is_valid_points, width, height = get_xyz_in_camera(inner_outputs, frames)

        max_val = torch.finfo(torch.float32).max
        frame_points = []
        for frame_index, frame in enumerate(frames):
            depth_map = torch.full((height[frame_index], width[frame_index]), max_val)
            frame_valid_points = is_valid_points[frame_index]
            u = torch.round(uv[frame_index, :, 0][frame_valid_points]).long()
            v = torch.round(uv[frame_index, :, 1][frame_valid_points]).long()
            z = -xyz_in_camera[frame_index, :, 2][frame_valid_points]
            vals, indices = scatter_min(z, v * width[frame_index] + u, out=depth_map.view(-1))
            valid_point_indices = torch.arange(frame_valid_points.shape[0])[frame_valid_points]
            frame_points.append(torch.unique(valid_point_indices[indices[vals < max_val]]))

        frame_intersections = torch.zeros(len(frames), len(frames))
        for i in tqdm(range(len(frames))):
            for j in range(i + 1, len(frames)):
                intersection = np.intersect1d(frame_points[i].numpy(), frame_points[j].numpy())
                frame_intersections[i][j] = 0 if len(frame_points[i]) == 0 else len(intersection) / len(frame_points[i])
                frame_intersections[j][i] = 0 if len(frame_points[j]) == 0 else len(intersection) / len(frame_points[j])

        torch.save(frame_intersections, dest / "overlap.pt")
        depth = -xyz_in_camera[..., 2][is_valid_points]
        metadata["near"] = depth.min().item()
        metadata["far"] = depth.max().item()
        points = inner_outputs.metadata["points3D_xyz"]
        min_bounds = points.min(dim=0)[0]
        max_bounds = points.max(dim=0)[0]
        metadata["scene_box"] = torch.stack([min_bounds, max_bounds]).tolist()

    with (dest / "metadata.json.bak").open("w") as f:
        json.dump(metadata, f, indent=4)

    (dest / "metadata.json.bak").rename(dest / "metadata.json")


@dataclass
class DL3DV(DataParser):
    """DL3DV Dataset"""

    config: DL3DVDataParserConfig

    def _generate_dataparser_outputs(self, split="train", chunk_index=None, num_chunks=None, get_default_scene=False):
        rank = get_rank() + self.config.worker_offset
        world_size = self.config.num_workers if self.config.num_workers is not None else get_world_size()

        # split = "test"

        overlap_threshold = self.config.neighbor_overlap_threshold
        crop = self.config.crop
        scale_near = self.config.scale_near
        use_dust3r_poses = self.config.use_dust3r_poses
        dust3r_use_known_cameras = self.config.dust3r_use_known_cameras

        cached_path = self.config.data / f"cached-metadata-{split}-{crop}-{scale_near}-{overlap_threshold[0]}-{overlap_threshold[1]}--{rank}-{world_size}-{chunk_index}-{num_chunks}-{use_dust3r_poses}-{dust3r_use_known_cameras}.pt"
        if (not get_default_scene) and self.config.scene_id is None and cached_path.exists():
            return torch.load(cached_path)

        if use_dust3r_poses and not self.config.skip_dust3r_incomplete:
            self.model = AsymmetricCroCo3DStereo.from_pretrained(self.config.dust3r_model_name).to(f"cuda:{get_rank()}")
            for p in self.model.parameters():
                p.requires_grad_(False)

        if get_default_scene:
            scenes = [self.config.default_scene_id]
        elif self.config.scene_id is not None:
            scenes = [self.config.scene_id]
        else:
            scenes = []
            to_scan = self.config.data / "original"
            check_for_colmap = True
            if not to_scan.exists():
                to_scan = self.config.data / f"crop-{crop}"
                check_for_colmap = False
                CONSOLE.log(f"Original dir not found, using {to_scan} instead")
            for subdir in sorted(to_scan.iterdir()):
                for scene_dir in sorted(subdir.iterdir()):
                    if (not check_for_colmap) \
                            or ((scene_dir / "transforms.json").exists() and (scene_dir / "colmap/sparse/0").exists()):
                        scenes.append(f"{scene_dir.parent.name}/{scene_dir.name}")
                    # else:
                    #     CONSOLE.log(f"Skipping {scene_dir}: {(scene_dir / 'transforms.json').exists()} {(scene_dir / 'colmap/sparse/0').exists()}")

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

        with ProcessPoolExecutor(max_workers=self.config.conversion_threads) as executor:
            convert_tasks = []
            for scene in tqdm(scenes):
                source = self.config.data / "original" / scene
                dest = self.config.data / f"crop-{crop}" / scene
                converted_metadata_path = dest / "metadata.json"
                if not converted_metadata_path.exists():
                    convert_tasks.append(executor.submit(convert_scene_metadata, source, dest, crop))

            for task in tqdm(convert_tasks):
                task.result()

        min_bounds = None
        max_bounds = None
        near = 1e10
        far = -1

        if use_dust3r_poses:
            dust3r_depth_paths = []

        for scene in tqdm(scenes):
            dest = self.config.data / f"crop-{crop}" / scene
            converted_metadata_path = dest / "metadata.json"

            if not converted_metadata_path.exists():
                continue

            with converted_metadata_path.open() as f:
                metadata = json.load(f)

            frames = metadata["frames"]
            if len(frames) < 4:
                CONSOLE.log(f"Skipping {scene} as it has less than 3 frames")
                continue

            if overlap_threshold is not None:
                view_overlap_path = dest / "overlap.pt"
                if not view_overlap_path.exists():
                    CONSOLE.log(f"Skipping {scene} as it does not have point cloud")
                    continue

                frame_intersections = torch.load(view_overlap_path)

            if use_dust3r_poses:
                dust3r_dest = self.config.data / ("dust3r-posed" if dust3r_use_known_cameras else "dust3r") / scene
                duster_metadata_path = dust3r_dest / "scene.pt"

                if not duster_metadata_path.exists():
                    if self.config.skip_dust3r_incomplete:
                        continue
                    if (dust3r_dest / "FAIL").exists():
                        continue
                    create_dust3r_poses(self.model, dust3r_dest, dest, metadata, None, #frame_intersections,
                                        dust3r_use_known_cameras,
                                        num_iters=500 if dust3r_use_known_cameras else 5000)

                    if (dust3r_dest / "FAIL").exists():
                        continue

                dust3r_metadata = torch.load(duster_metadata_path, map_location="cpu")

            pose_scale_factor = 1
            if "near" in metadata:
                scene_near = max(metadata["near"], 0.01)
                if scale_near is not None:
                    pose_scale_factor = scene_near / scale_near

                near = min(near, scene_near / pose_scale_factor)
                far = max(far, metadata["far"] / pose_scale_factor)
                scene_box = dust3r_metadata["scene_box"] if use_dust3r_poses \
                    else torch.FloatTensor(metadata["scene_box"])
                current_scene_box = scene_box / pose_scale_factor

                if min_bounds is None:
                    min_bounds, max_bounds = current_scene_box
                else:
                    min_bounds = torch.minimum(min_bounds, current_scene_box[0])
                    max_bounds = torch.maximum(max_bounds, current_scene_box[1])

            scene_index_start = len(image_filenames)
            scene_index_end = scene_index_start + len(frames)
            for scene_image_index, frame in enumerate(frames):
                image_filenames.append(dest / frame["file_path"])
                if use_dust3r_poses and not dust3r_use_known_cameras:
                    c2w = dust3r_metadata["poses"][scene_image_index, :3]
                    c2w[:, 1:3] *= -1  # opencv to opengl

                    focal = float(dust3r_metadata["focals"][scene_image_index])
                    fx.append(focal * frame["w"] / 224)
                    fy.append(focal * frame["h"] / 224)
                    pp = dust3r_metadata["pp"][scene_image_index]
                    cx.append(float(pp[0]) * frame["w"] / 224)
                    cy.append(float(pp[1]) * frame["h"] / 224)
                else:
                    c2w = torch.FloatTensor(frame["transform_matrix"])
                    fx.append(frame["fl_x"])
                    fy.append(frame["fl_y"])
                    cx.append(frame["cx"])
                    cy.append(frame["cy"])

                if use_dust3r_poses:
                    frame_path_png = Path(dust3r_dest / frame["file_path"])
                    frame_path_pt = frame_path_png.parent / f"{frame_path_png.stem}.pt"
                    dust3r_depth_paths.append(frame_path_pt)

                c2w[:3, 3] /= pose_scale_factor
                c2ws.append(c2w)
                width.append(frame["w"])
                height.append(frame["h"])

                if overlap_threshold is not None:
                    min_mask = frame_intersections[scene_image_index] > overlap_threshold[0]
                    if split != "train":
                        max_mask = frame_intersections[scene_image_index] < overlap_threshold[1]
                        thresh_mask = torch.logical_and(min_mask, max_mask)
                    else:
                        thresh_mask = min_mask

                    neighbors = (scene_index_start + torch.arange(len(frames))[thresh_mask]).tolist()
                    if len(neighbors) < 3:
                        neighbors = list(range(max(scene_index_start, scene_index_start + scene_image_index - 3),
                                               scene_index_start + scene_image_index)) + list(
                            range(scene_index_start + scene_image_index,
                                  min(scene_index_end, scene_index_start + scene_image_index + 3)))
                        assert len(neighbors) >= 3, len(neighbors)

                    neighboring_views.append(neighbors)
                else:
                    neighboring_views.append(
                        list(range(scene_index_start, scene_index_start + scene_image_index))
                        + list(range(scene_index_start + scene_image_index + 1, scene_index_end)))

            if use_dust3r_poses and (not "psnr" in dust3r_metadata):
                depth = torch.stack([torch.load(x, map_location="cuda") for x in dust3r_depth_paths[-len(frames):]])
                c2ws_dust3r = torch.stack(c2ws[-len(frames):]).cuda().clone()
                c2ws_dust3r[:, :, 1:3] *= -1  # opengl to opencv

                psnr = GaussianOptimizer(image_filenames[-len(frames):], depth, c2ws_dust3r,
                    torch.FloatTensor(fx[-len(frames):]).view(-1, 1).cuda() * 224 / 256,
                    torch.FloatTensor(fy[-len(frames):]).view(-1, 1).cuda() * 224 / 256,
                    torch.FloatTensor(cx[-len(frames):]).view(-1, 1).cuda() * 224 / 256,
                    torch.FloatTensor(cy[-len(frames):]).view(-1, 1).cuda() * 224 / 256, True, False).cuda().get_psnr()

                dust3r_metadata["psnr"] = psnr
                torch.save(dust3r_metadata, duster_metadata_path)
                print(scene, torch.FloatTensor(psnr).mean())
            elif use_dust3r_poses and split != "train":
                    print(scene, torch.FloatTensor(dust3r_metadata["psnr"]).mean())

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
            # i_train, i_eval = get_train_eval_split_fraction(image_filenames, self.config.train_split_fraction)
            # if split == "train":
            #     indices = i_train
            # elif split in ["val", "test"]:
            #     indices = i_eval
            # else:
            #     raise ValueError(f"Unknown dataparser split {split}")
            # 
            # image_filenames = [image_filenames[i] for i in indices]

            #  Need to deal with indices changing post filtering if we actually want this
            neighboring_views = None  # [neighboring_views[i] for i in indices]

            # idx_tensor = torch.LongTensor(indices)
            # c2ws = c2ws[idx_tensor]
            # width = width[idx_tensor]
            # height = height[idx_tensor]
            # fx = fx[idx_tensor]
            # fy = fy[idx_tensor]
            # cx = cx[idx_tensor]
            # cy = cy[idx_tensor]

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
        }

        if neighboring_views is not None:
            metadata[NEIGHBOR_INDICES] = neighboring_views

        if use_dust3r_poses:
            metadata[DEPTH] = dust3r_depth_paths

        # if (not get_default_scene) and self.config.scene_id is None:
        #     metadata[DEFAULT_SCENE_METADATA] = self._generate_dataparser_outputs(split, get_default_scene=True)

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            metadata=metadata
        )

        if (not get_default_scene) and self.config.scene_id is None and (not self.config.skip_dust3r_incomplete):
            torch.save(dataparser_outputs, cached_path)

        return dataparser_outputs
