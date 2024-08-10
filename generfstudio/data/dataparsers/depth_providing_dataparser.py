from __future__ import annotations

import shutil
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Type, Optional, Dict

import torch
from PIL import Image
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.utils.image import load_images
from mast3r.model import AsymmetricMASt3R
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig, DataparserOutputs,
)

from generfstudio.data.dataparsers.dtu_dataparser import DTUDataParserConfig
from generfstudio.data.dataparsers.llff_dataparser import LLFFDataParserConfig
from generfstudio.pose_utils import generate_ellipse_path


@dataclass
class DepthProvidingDataParserConfig(DataParserConfig):
    _target: Type = field(default_factory=lambda: DepthProviding)
    """target class to instantiate"""

    inner: DataParserConfig = field(default_factory=lambda: LLFFDataParserConfig())
    """inner dataparser"""

    depth_model_name: str = "/data/hturki/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"

    cache_path: str = "/scratch/hturki/mast3r-cache"

    update_path: Path = Path("/scratch/hturki/new-images")

    min_conf_thresh: float = 2
    # eval_res: int = 256
    eval_factor: int = 8

    train_images: Optional[int] = None
    use_dtu: bool = False


updated_filenames = []
updated_fx = []
updated_fy = []
updated_cx = []
updated_cy = []
updated_width = []
updated_height = []
updated_c2w = []
updated_camera_type = []


@dataclass
class DepthProviding(DataParser):
    config: DepthProvidingDataParserConfig

    def __init__(self, config: DepthProvidingDataParserConfig):
        super().__init__(config=config)
        if self.config.use_dtu:
            config.inner = DTUDataParserConfig()

        if config.data != Path():
            config.inner.data = self.config.data

        if self.config.train_images is not None:
            config.inner.train_images = self.config.train_images
        self.inner: DataParser = config.inner.setup()

        if self.config.update_path.exists():
            shutil.rmtree(self.config.update_path)
        self.config.update_path.mkdir(parents=True)

    def get_dataparser_outputs(self, split: str = "train", **kwargs: Optional[Dict]) -> DataparserOutputs:
        inner_outputs = self.inner.get_dataparser_outputs(split)

        if split == "train":
            model = AsymmetricMASt3R.from_pretrained(self.config.depth_model_name).cuda()
            c2ws_opencv = torch.eye(4).unsqueeze(0).repeat(len(inner_outputs.cameras), 1, 1)
            c2ws_opencv[:, :3] = inner_outputs.cameras.camera_to_worlds
            c2ws_opencv[:, :, 1:3] *= -1  # opengl to opencv

            train_cameras = deepcopy(inner_outputs.cameras)
            train_cameras.rescale_output_resolution(512 / torch.maximum(train_cameras.width, train_cameras.height))
            train_image_filenames = inner_outputs.image_filenames
            if len(train_image_filenames) == 1:
                train_image_filenames = [train_image_filenames[0], train_image_filenames[0]]
                train_cameras = train_cameras[torch.LongTensor([0, 0])]

            imgs = load_images([str(x) for x in train_image_filenames], size=512)
            cropped_widths = torch.LongTensor([x["true_shape"][0][1] for x in imgs]).unsqueeze(-1)
            cropped_heights = torch.LongTensor([x["true_shape"][0][0] for x in imgs]).unsqueeze(-1)
            train_cameras.cx -= ((train_cameras.width - cropped_widths) / 2)
            train_cameras.cy -= ((train_cameras.height - cropped_heights) / 2)
            train_cameras.height = cropped_heights
            train_cameras.width = cropped_widths

            view1 = {"img": [], "idx": [], "instance": []}
            view2 = {"img": [], "idx": [], "instance": []}
            for first in range(len(imgs)):
                for second in range(len(imgs)):
                    if first == second:
                        continue
                    view1["img"].append(imgs[first]["img"])
                    view2["img"].append(imgs[second]["img"])
                    view1["idx"].append(first)
                    view1["instance"].append(str(first))
                    view2["idx"].append(second)
                    view2["instance"].append(str(second))

            view1["img"] = torch.cat(view1["img"]).cuda()
            view2["img"] = torch.cat(view2["img"]).cuda()

            with torch.no_grad():
                pred1, pred2 = model(view1, view2)
            output = {
                "view1": {"idx": view1["idx"], "instance": view1["instance"]},
                "view2": {"idx": view2["idx"], "instance": view2["instance"]},
                "pred1": {k: [x for x in pred1[k]] for k in ["pts3d", "conf"]},
                "pred2": {k: [x for x in pred2[k]] for k in ["pts3d_in_other_view", "conf"]},
            }

            # scene = sparse_global_alignment([str(x) for x in train_image_filenames], pairs, self.config.cache_path, model, matching_conf_thr=0)
            # pts3d, depthmaps, confs = scene.get_dense_pts3d(subsample=8)
            # pts3d = torch.cat(pts3d)
            # depth = torch.stack(depthmaps)

            scene = global_aligner(output, device="cuda", mode=GlobalAlignerMode.PointCloudOptimizer, optimize_pp=True)
            scene.preset_pose(c2ws_opencv)
            scene.preset_focal(train_cameras.fx)  # Assume fx and fy are almost equal
            scene.preset_principal_point(torch.cat([train_cameras.cx, train_cameras.cy], -1))
            scene.compute_global_alignment(init="known_poses", niter=300, schedule="cosine", lr=0.1)
            scene.clean_pointcloud()
            pts3d = scene.get_pts3d(raw=True)
            depth = scene.get_depthmaps(raw=True)

            xyz = pts3d.view(-1, 3).detach()
            rgb = torch.cat([imgs[x]["img"].squeeze(0).permute(1, 2, 0) for x in range(len(train_cameras))]).view(-1,
                                                                                                                  3) / 2 + 0.5

            ray_bundle = train_cameras.generate_rays(camera_indices=torch.arange(train_cameras.shape[0]).view(-1, 1))

            pixel_area = ray_bundle.pixel_area.squeeze(-1).permute(2, 0, 1).view(train_cameras.shape[0], -1)
            directions_norm = ray_bundle.metadata["directions_norm"].squeeze(-1).permute(2, 0, 1).view(
                train_cameras.shape[0], -1)
            scales = (depth.detach() * pixel_area.to(depth) * directions_norm.to(depth)).view(-1, 1)

            # scales = depth.detach() / train_cameras.fx.view(-1, 1).cuda()

            # mask = torch.cat(confs).view(-1) > self.config.min_conf_thresh
            mask = torch.stack([x for x in scene.im_conf]).view(-1).to(xyz.device) > self.config.min_conf_thresh
            depth = depth.detach().view(len(train_cameras), cropped_heights[0].item(), cropped_widths[0].item(), 1)
            test_cameras = self.get_dataparser_outputs("test").cameras
            # poses = generate_ellipse_path(
            #     torch.cat([train_cameras.camera_to_worlds, test_cameras.camera_to_worlds]).numpy(), n_frames=10)

            metadata = {
                "train_cameras": train_cameras,
                "train_depths": torch.where(mask.view(depth.shape), depth, 0),
                "train_rgbs": rgb.view(-1, 3),
                "ellipse_cameras": test_cameras,
                # "ellipse_cameras": Cameras(
                #     fx=test_cameras.fx.mean(dim=0, keepdim=True).expand(poses.shape[0], -1),
                #     fy=test_cameras.fx.mean(dim=0, keepdim=True).expand(poses.shape[0], -1),
                #     cx=torch.full((poses.shape[0], 1), 128),
                #     cy=torch.full((poses.shape[0], 1), 128),
                #     height=torch.full((poses.shape[0], 1), 256, dtype=torch.long),
                #     width=torch.full((poses.shape[0], 1), 256, dtype=torch.long),
                #     camera_to_worlds=torch.FloatTensor(poses),
                # ),
                "points3D_xyz": xyz[mask],
                "points3D_rgb": (rgb[mask.to(rgb.device)] * 255).byte(),
                "points3D_scale": scales[mask],
                "train_image_filenames": train_image_filenames,
                "add_image_fn": self.add_image,
                "depth_outputs": output,
                "min_conf_thresh": self.config.min_conf_thresh,
            }

            # self.base_outputs = DataparserOutputs(image_filenames=train_image_filenames,
            #                                       cameras=train_cameras,
            #                                       alpha_color=inner_outputs.alpha_color,
            #                                       scene_box=inner_outputs.scene_box,
            #                                       dataparser_scale=inner_outputs.dataparser_scale,
            #                                       metadata={})

            self.base_outputs = DataparserOutputs(image_filenames=train_image_filenames,
                                                  cameras=train_cameras,
                                                  alpha_color=inner_outputs.alpha_color,
                                                  scene_box=inner_outputs.scene_box,
                                                  dataparser_scale=inner_outputs.dataparser_scale,
                                                  metadata=metadata)

            return self.base_outputs
        else:
            eval_cameras = deepcopy(inner_outputs.cameras)
            eval_cameras.rescale_output_resolution(1 / self.config.eval_factor)

            # eval_cameras.rescale_output_resolution(
            #     self.config.eval_res / torch.minimum(eval_cameras.width, eval_cameras.height))
            # eval_cameras.cx -= ((eval_cameras.width - self.config.eval_res) / 2)
            # eval_cameras.cy -= ((eval_cameras.height - self.config.eval_res) / 2)
            # eval_cameras.height.fill_(self.config.eval_res)
            # eval_cameras.width.fill_(self.config.eval_res)
            return DataparserOutputs(
                image_filenames=inner_outputs.image_filenames,
                cameras=eval_cameras,
                alpha_color=inner_outputs.alpha_color,
                scene_box=inner_outputs.scene_box,
                dataparser_scale=inner_outputs.dataparser_scale,
                metadata=inner_outputs.metadata,
            )

    def get_updated_outputs(self):
        global updated_filenames
        global updated_fx
        global updated_fy
        global updated_cx
        global updated_cy
        global updated_width
        global updated_height
        global updated_c2w
        global updated_camera_type

        return DataparserOutputs(
            image_filenames=self.base_outputs.image_filenames + updated_filenames,
            cameras=Cameras(
                fx=torch.cat([self.base_outputs.cameras.fx] + updated_fx),
                fy=torch.cat([self.base_outputs.cameras.fy] + updated_fy),
                cx=torch.cat([self.base_outputs.cameras.cx] + updated_cx),
                cy=torch.cat([self.base_outputs.cameras.cy] + updated_cy),
                height=torch.cat([self.base_outputs.cameras.height] + updated_height),
                width=torch.cat([self.base_outputs.cameras.width] + updated_width),
                camera_to_worlds=torch.cat([self.base_outputs.cameras.camera_to_worlds] + updated_c2w),
                camera_type=torch.cat([self.base_outputs.cameras.camera_type] + updated_camera_type),
            ),
            alpha_color=self.base_outputs.alpha_color,
            scene_box=self.base_outputs.scene_box,
            dataparser_scale=self.base_outputs.dataparser_scale,
            metadata=self.base_outputs.metadata,
        )

    def add_image(self, rgb: torch.Tensor, camera: Cameras) -> Path:
        global updated_filenames
        global updated_fx
        global updated_fy
        global updated_cx
        global updated_cy
        global updated_width
        global updated_height
        global updated_c2w
        global updated_camera_type

        save_path = self.config.update_path / f"{len(updated_filenames)}.png"
        Image.fromarray((rgb * 255).byte().cpu().numpy()).save(save_path)
        updated_filenames.append(save_path)

        camera = camera.to("cpu")
        updated_fx.append(camera.fx)
        updated_fy.append(camera.fy)
        updated_cx.append(camera.cx)
        updated_cy.append(camera.cy)
        updated_height.append(camera.height)
        updated_width.append(camera.width)
        updated_c2w.append(camera.camera_to_worlds)
        updated_camera_type.append(camera.camera_type)

        return save_path
