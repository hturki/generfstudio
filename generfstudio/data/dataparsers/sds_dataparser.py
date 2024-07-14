from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Type, Optional

import numpy as np
import torch
from PIL import Image

from generfstudio.data.dataparsers.dl3dv_dataparser import DL3DVDataParserConfig
from generfstudio.fields.batched_pc_optimizer import GlobalPointCloudOptimizer

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


@dataclass
class SDSDataParserConfig(DataParserConfig):
    _target: Type = field(default_factory=lambda: SDS)
    """target class to instantiate"""
    inner: DataParserConfig = field(default_factory=lambda: DL3DVDataParserConfig(
        scene_id="7K/ebbbe07e9e87e488553e470ec266d6a2c967b891294cab928938a69055cac117"))
        # scene_id="1K/006771db3c057280f9277e735be6daa24339657ce999216c38da68002a443fed"))
    # scene_id="4K/ac41bb001ee989c3c0237341aa37f9f985e3e55b03cc70089ebffd938063bcdb"))
    """inner dataparser"""

    num_images: int = 2
    image_cond_override: Optional[Path] = None

    init_with_depth: bool = True
    use_global_optimizer: bool = False

    dust3r_checkpoint_path: str = "/data/hturki/dust3r/checkpoints/DUSt3R_ViTLarge_BaseDecoder_224_linear.pth"


@dataclass
class SDS(DataParser):
    config: SDSDataParserConfig

    def __init__(self, config: SDSDataParserConfig):
        super().__init__(config=config)
        if config.data != Path():
            config.inner.data = self.config.data
        self.inner: DataParser = config.inner.setup()

    def _generate_dataparser_outputs(self, split='train') -> DataparserOutputs:
        inner_outputs = self.inner.get_dataparser_outputs(split)
        if split == 'train':
            image_indices = np.linspace(0, len(inner_outputs.image_filenames) - 1, self.config.num_images, dtype=int)

            train_image_filenames = np.array(inner_outputs.image_filenames)[image_indices].tolist()
            train_image_cameras = inner_outputs.cameras[torch.LongTensor(image_indices)]
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
                "train_image_cameras_all": inner_outputs.cameras,
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

                if self.config.use_global_optimizer:
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
                                                          cameras.fy, cameras.cx, cameras.cy, cameras.shape[0] - 1,
                                                          None, verbose=True)
                        scene.init_least_squares()
                    # scene.init_scale_from_poses()
                    # global_alignment_loop(scene, niter=300, schedule="cosine", lr=0.01)
                    pts3d = scene.pts3d_world()
                    depth = scene.depth()
                else:
                    pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
                    output = inference(pairs, model, "cuda", batch_size=1)
                    scene = global_aligner(output, device="cuda", mode=GlobalAlignerMode.PointCloudOptimizer,
                                           optimize_pp=True)
                    scene.preset_pose(c2ws)
                    scene.preset_focal(cameras.fx)  # Assume fx and fy are almost equal
                    scene.preset_principal_point(torch.cat([cameras.cx, cameras.cy], -1))
                    scene.compute_global_alignment(init="known_poses", niter=300, schedule="cosine", lr=0.1)
                    # scene.clean_pointcloud()
                    pts3d = scene.get_pts3d(raw=True)
                    depth = scene.get_depthmaps(raw=True)

                xyz = pts3d.view(-1, 3).detach()
                scales = depth.detach() / cameras.fx.view(-1, 1).cuda()
                # scales = scales.view(-1, 1).expand(-1, 3)
                rgb = torch.cat([images[x]["img"].squeeze(0).permute(1, 2, 0) for x in range(len(cameras))]).view(-1,
                                                                                                                  3) / 2 + 0.5

                metadata["points3D_xyz"] = xyz
                metadata["points3D_rgb"] = (rgb * 255).byte()
                metadata["points3D_scale"] = scales.view(-1, 1)

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
