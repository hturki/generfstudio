from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Type, Optional

import numpy as np
import torch
from PIL import Image
from dust3r.cloud_opt.base_opt import global_alignment_loop
from dust3r.cloud_opt.optimizer import _fast_depthmap_to_pts3d
from dust3r.utils.geometry import geotrf

from generfstudio.data.dataparsers.dl3dv_dataparser import DL3DVDataParserConfig
from generfstudio.fields.batched_pc_optimizer import GlobalPointCloudOptimizer
from generfstudio.generfstudio_constants import DEPTH

try:
    from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
    from dust3r.image_pairs import make_pairs
    from dust3r.inference import inference
    from dust3r.utils.image import load_images
    from dust3r.model import AsymmetricCroCo3DStereo
except:
    pass

from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig, DataparserOutputs,
)
from nerfstudio.data.dataparsers.blender_dataparser import BlenderDataParserConfig


@dataclass
class SDSDataParserConfig(DataParserConfig):
    _target: Type = field(default_factory=lambda: SDS)
    """target class to instantiate"""
    inner: DataParserConfig = field(default_factory=lambda: DL3DVDataParserConfig(
        scene_id="1K/006771db3c057280f9277e735be6daa24339657ce999216c38da68002a443fed"))
    # scene_id="4K/ac41bb001ee989c3c0237341aa37f9f985e3e55b03cc70089ebffd938063bcdb"))
    """inner dataparser"""

    start: int = 0
    end: int = 2
    image_cond_override: Optional[Path] = None

    init_with_depth: bool = True
    use_global_optimizer: bool = True

    dust3r_checkpoint_path: str = "/data/hturki/dust3r/checkpoints/DUSt3R_ViTLarge_BaseDecoder_224_linear.pth"


@dataclass
class SDS(DataParser):
    config: SDSDataParserConfig

    def __init__(self, config: SDSDataParserConfig):
        super().__init__(config=config)
        if config.data != Path():
            config.inner.data = self.config.data
        self.inner: DataParser = config.inner.setup()
        self.start = config.start
        self.end = config.end

    def _generate_dataparser_outputs(self, split='train') -> DataparserOutputs:
        inner_outputs = self.inner.get_dataparser_outputs(split)
        if split == 'train':
            train_image_filenames = inner_outputs.image_filenames[::10][self.start:self.end + 1]
            train_image_cameras = inner_outputs.cameras[::10][self.start:self.end + 1]
            image_cond_override = self.config.image_cond_override

            if image_cond_override is not None:
                train_image_filenames.pop()
                resized = image_cond_override.parent / f"{image_cond_override.stem}-resized{image_cond_override.suffix}"
                if not resized.exists():
                    Image.open(image_cond_override).resize(
                        (train_image_cameras.width[-1], train_image_cameras.height[-1]), Image.LANCZOS).save(resized)
                train_image_filenames.append(resized)

            metadata = {
                "train_image_filenames": train_image_filenames,
                "train_image_cameras": train_image_cameras,
                "test_cameras": self.inner.get_dataparser_outputs("test").cameras
            }

            if self.config.init_with_depth:
                c2ws = torch.eye(4).unsqueeze(0).repeat(len(train_image_cameras), 1, 1)
                c2ws[:, :3] = train_image_cameras.camera_to_worlds
                c2ws[:, :, 1:3] *= -1  # opengl to opencv

                cameras = deepcopy(train_image_cameras)
                cameras.rescale_output_resolution(224 / torch.maximum(cameras.width, cameras.height))

                model = AsymmetricCroCo3DStereo.from_pretrained(self.config.dust3r_checkpoint_path).cuda()

                images = load_images([str(x) for x in train_image_filenames], size=224)
                has_alpha = False  # np.asarray(Image.open(train_image_filenames[0])).shape[-1] == 4
                if has_alpha:
                    masks = [torch.BoolTensor(np.asarray(Image.open(x).resize((224, 224), Image.NEAREST))[..., 3] > 0)
                             for x in train_image_filenames]

                if DEPTH in inner_outputs.metadata:

                    depths = []
                    for frame_path_pt in inner_outputs.metadata[DEPTH][::10][self.start:self.end + 1]:
                        frame_depth = torch.load(frame_path_pt, map_location="cpu")
                        depths.append(frame_depth)
                    # import pdb;
                    # pdb.set_trace()
                    depths = torch.stack(depths)
                    pixels_im = torch.stack(
                        torch.meshgrid(torch.arange(224, dtype=torch.float32),
                                       torch.arange(224, dtype=torch.float32),
                                       indexing="ij")).permute(2, 1, 0)
                    pixels = pixels_im.reshape(-1, 2)
                    pts3d_cam = _fast_depthmap_to_pts3d(depths, pixels.unsqueeze(0).expand(depths.shape[0], -1, -1),
                                                        cameras.fx, torch.cat([cameras.cx, cameras.cy], -1))
                    pts3d = geotrf(c2ws, pts3d_cam)
                elif self.config.use_global_optimizer:
                    with torch.no_grad():
                        view1 = {"img": [], "idx": [], "instance": []}
                        view2 = {"img": [], "idx": [], "instance": []}
                        for first in range(len(images)):
                            for second in range(len(images)):
                                if first == second:
                                    continue
                                view1["img"].append(images[first]["img"][0])
                                view2["img"].append(images[second]["img"][0])
                                view1["idx"].append(first)
                                view1["instance"].append(str(first))
                                view2["idx"].append(second)
                                view2["instance"].append(str(second))

                        view1["img"] = torch.stack(view1["img"]).cuda()
                        view2["img"] = torch.stack(view2["img"]).cuda()

                        pred1, pred2 = model(view1, view2)
                        cameras = cameras.to("cuda")
                        scene = GlobalPointCloudOptimizer(view1, view2, pred1, pred2, c2ws.cuda(), cameras.fx,
                                                          cameras.fy,
                                                          cameras.cx, cameras.cy, None, None, None, None,
                                                          cameras.shape[0] - 1, verbose=True)
                        scene.init_least_squares()
                    # scene.init_scale_from_poses()
                    # global_alignment_loop(scene, niter=300, schedule="cosine", lr=0.01)
                    pts3d = scene.pts3d_world()
                else:
                    pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
                    output = inference(pairs, model, "cuda", batch_size=1)
                    scene = global_aligner(output, device="cuda", mode=GlobalAlignerMode.PointCloudOptimizer,
                                           optimize_pp=True)
                    scene.preset_pose(c2ws)
                    scene.preset_focal(cameras.fx)  # Assume fx and fy are almost equal
                    scene.preset_principal_point(cameras.cx)  # Assume cx and cy are almost equal
                    scene.compute_global_alignment(init="known_poses", niter=3000, schedule="cosine", lr=0.1)
                    # scene.clean_pointcloud()
                    pts3d = scene.get_pts3d()

                xyz = []
                rgb = []
                scales = []
                confs = []
                for i in range(len(train_image_cameras)):
                    image_xyz = pts3d[i].detach().view(-1, 3)
                    image_rgb = images[i]["img"].squeeze(0).permute(1, 2, 0).detach().view(-1, 3).to(image_xyz.device)
                    pixel_area = cameras.generate_rays(camera_indices=i).pixel_area.view(-1, 1).to(image_xyz.device)
                    # confidence = torch.log(scene.im_conf[i].detach().view(-1, 1)).clamp_max(1)

                    if has_alpha:
                        mask = scene.get_masks()[i].view(-1)
                        image_rgb = image_rgb[mask]
                        image_xyz = image_xyz[mask]
                        pixel_area = pixel_area[mask]
                        # confidence = confidence[mask]

                    xyz.append(image_xyz)
                    rgb.append(image_rgb)

                    camera_pos = cameras[i].camera_to_worlds[:, 3]
                    distances = (image_xyz - camera_pos.to(image_xyz.device)).norm(dim=-1, keepdim=True)
                    scales.append(pixel_area.to(distances.device) * distances)

                    # confs.append(confidence)

                metadata["points3D_xyz"] = torch.cat(xyz)
                metadata["points3D_rgb"] = ((torch.cat(rgb) / 2 + 0.5) * 255).byte()
                metadata["points3D_scale"] = torch.cat(scales)
                # metadata["points3D_conf"] = torch.cat(confs)

            return DataparserOutputs(
                image_filenames=train_image_filenames,
                cameras=train_image_cameras,
                alpha_color=inner_outputs.alpha_color,
                scene_box=inner_outputs.scene_box,
                dataparser_scale=inner_outputs.dataparser_scale,
                metadata=metadata
            )
        else:
            return inner_outputs
