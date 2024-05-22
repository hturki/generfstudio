from copy import deepcopy
from typing import Optional

import torch
from dust3r.model import AsymmetricCroCo3DStereo
from gsplat import project_gaussians, rasterize_gaussians
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.utils import profiler
from torch import nn
from torch.nn import functional as F

from generfstudio.fields.batched_pc_optimizer import GlobalPointCloudOptimizer
from generfstudio.generfstudio_constants import RGB, FEATURES, ACCUMULATION, ALIGNMENT_LOSS, NEIGHBOR_RESULTS, DEPTH, \
    VALID_ALIGNMENT, ACCUMULATION_FEATURES


class Dust3rField(nn.Module):

    def __init__(
            self,
            model_name: str,
            in_feature_dim: int = 512,
            out_feature_dim: int = 128,
            image_dim: int = 224,
            project_features: bool = True,
            min_conf_thr: int = 3,
            pnp_method: str = "pytorch3d",
            use_elem_wise_pt_conf: bool = False,
            alignment_iter: int = 300,
            alignment_lr: float = 0.01,
            alignment_schedule: str = "cosine",
            use_confidence_opacity: bool = False,
            depth_available: bool = False,
    ) -> None:
        super().__init__()
        if not depth_available:
            self.model = AsymmetricCroCo3DStereo.from_pretrained(model_name)
            for p in self.model.parameters():
                p.requires_grad_(False)
        else:
            self.model = None

        self.image_dim = image_dim
        self.project_features = project_features
        self.min_conf_thr = min_conf_thr
        self.pnp_method = pnp_method
        self.use_elem_wise_pt_conf = use_elem_wise_pt_conf
        self.alignment_iter = alignment_iter
        self.alignment_lr = alignment_lr
        self.alignment_schedule = alignment_schedule
        self.use_confidence_opacity = use_confidence_opacity
        self.feature_layer = nn.Linear(in_feature_dim, out_feature_dim)

    @profiler.time_function
    @torch.cuda.amp.autocast(enabled=False)
    def splat_gaussians(self, camera: Cameras, xyz: torch.Tensor, scales: torch.Tensor, conf: torch.Tensor,
                        w2c: torch.Tensor, rgbs: Optional[torch.Tensor], features: Optional[torch.Tensor],
                        return_depth: bool):
        xys, depths, radii, conics, comp, num_tiles_hit, cov3d = project_gaussians(
            xyz,
            scales,
            1,
            torch.cat([torch.ones_like(conf), torch.zeros_like(xyz)], -1),
            w2c[:3],
            camera.fx[0].item(),
            camera.fy[0].item(),
            camera.cx[0].item(),
            camera.cy[0].item(),
            camera.height[0].item(),
            camera.width[0].item(),
            16,
        )

        inputs = []
        if rgbs is not None:
            inputs.append(rgbs)
        if features is not None:
            inputs.append(features.float())
        if return_depth:
            inputs.append(depths.unsqueeze(-1))

        inputs = torch.cat(inputs, -1)

        splat_results, alpha = rasterize_gaussians(
            xys,
            depths,
            radii,
            conics,
            num_tiles_hit,
            inputs,
            conf,
            camera.height[0].item(),
            camera.width[0].item(),
            16,
            background=torch.zeros(inputs.shape[-1], device=xys.device),
            return_alpha=True,
        )

        outputs = {ACCUMULATION: alpha.unsqueeze(-1)}

        if rgbs is not None:
            outputs[RGB] = splat_results[..., :3]

        if features is not None:
            offset = 3 if rgbs is not None else 0
            outputs[FEATURES] = splat_results[..., offset:offset + features.shape[-1]]

        if return_depth:
            outputs[DEPTH] = splat_results[..., -1:]

        return outputs

    @profiler.time_function
    def forward(self, target_cameras: Cameras, neighbor_cameras: Cameras, cond_rgbs: torch.Tensor,
                cond_features: torch.Tensor, cond_pts3d: torch.Tensor, return_depth: bool,
                target_cameras_feature: Optional[Cameras] = None):
        with torch.no_grad():
            if cond_rgbs.shape[-1] != self.image_dim:
                with profiler.time_function("interpolate_rgb_for_dust3r"):
                    cond_rgbs = F.interpolate(cond_rgbs.view(-1, *cond_rgbs.shape[2:]), self.image_dim, mode="bicubic")
                    cond_rgbs = cond_rgbs.view(len(target_cameras), -1, *cond_rgbs.shape[1:])

            with profiler.time_function("feature_layer_for_dust3r"):
                # bilinear interpolation can be expensive for high-dimensional features
                b, d, h, w = cond_features.shape
                cond_features = cond_features.permute(0, 2, 3, 1)
                cond_features = self.feature_layer(cond_features.view(-1, d)).view(b, h, w, -1)

            with profiler.time_function("interpolate_features_for_dust3r"):
                cond_features = F.interpolate(cond_features.permute(0, 3, 1, 2), self.image_dim)
                cond_features = cond_features.view(len(target_cameras), -1, *cond_features.shape[1:])

        with profiler.time_function("dust3r_camera_prep"):
            neighbor_cameras = deepcopy(neighbor_cameras)
            neighbor_cameras.rescale_output_resolution(self.image_dim / neighbor_cameras.height[0, 0].item())

            neighbor_c2ws = torch.eye(4, device=neighbor_cameras.device).view(1, 1, 4, 4).repeat(
                neighbor_cameras.shape[0],
                neighbor_cameras.shape[1],
                1, 1)
            neighbor_c2ws[:, :, :3] = neighbor_cameras.camera_to_worlds
            neighbor_c2ws[:, :, :, 1:3] *= -1  # opengl to opencv (what dust3r expects)

            # shift the camera to center of scene looking at center
            R = target_cameras.camera_to_worlds[:, :3, :3].clone()  # 3 x 3, clone to avoid changing original cameras
            R[:, :, 1:3] *= -1  # opengl to opencv (what gsplat expects)
            T = target_cameras.camera_to_worlds[:, :3, 3:]  # 3 x 1

            w2cs = torch.eye(4, device=R.device).unsqueeze(0).repeat(len(target_cameras), 1, 1)
            R_inv = R[:, :3, :3].transpose(1, 2)
            w2cs[:, :3, :3] = R_inv
            w2cs[:, :3, 3:] = -torch.bmm(R_inv, T)

        rgbs = []
        features = []
        accumulations = []
        if return_depth:
            depths = []
        if target_cameras_feature is not None:
            accumulation_features = []

        if not self.training:
            neighbor_rgbs = []
            neighbor_depths = []
            neighbor_accumulations = []

            neighbor_w2cs = torch.eye(4, device=R.device).view(1, 1, 4, 4).repeat(neighbor_cameras.shape[0],
                                                                                  neighbor_cameras.shape[1], 1, 1)

            # We can reuse neighbor_c2ws since dust3r and gsplat use the same coordinate convention
            neighbor_R_inv = neighbor_c2ws[:, :, :3, :3].transpose(2, 3)
            neighbor_w2cs[:, :, :3, :3] = neighbor_R_inv
            neighbor_T = neighbor_c2ws[:, :, :3, 3:]
            neighbor_inv_T = -torch.bmm(neighbor_R_inv.view(-1, 3, 3), neighbor_T.view(-1, 3, 1))
            neighbor_w2cs[:, :, :3, 3:] = neighbor_inv_T.view(neighbor_w2cs[:, :, :3, 3:].shape)

        with profiler.time_function("get_pixel_areas"):
            camera_indices = torch.cat(
                [torch.arange(neighbor_cameras.shape[0]).repeat_interleave(neighbor_cameras.shape[1]).unsqueeze(-1),
                 torch.arange(neighbor_cameras.shape[1]).repeat(neighbor_cameras.shape[0]).unsqueeze(-1)], -1)
            pixel_areas = neighbor_cameras.generate_rays(camera_indices=camera_indices).pixel_area.permute(2, 0, 1,
                                                                                                           3)
        if cond_pts3d is None:
            with profiler.time_function("dust3r_for_all_cameras"):
                with torch.no_grad():
                    cond_rgbs_duster_input = cond_rgbs * 2 - 1
                    with profiler.time_function("dust3r_load_images"):
                        view1 = {"img": [], "idx": [], "instance": []}
                        view2 = {"img": [], "idx": [], "instance": []}
                        for i in range(len(target_cameras)):
                            for first in range(len(cond_rgbs_duster_input[i])):
                                for second in range(len(cond_rgbs_duster_input[i])):
                                    if first == second:
                                        continue
                                    view1["img"].append(cond_rgbs_duster_input[i, first])
                                    view2["img"].append(cond_rgbs_duster_input[i, second])
                                    view_idx_first = i * len(cond_rgbs_duster_input[i]) + first
                                    view1["idx"].append(view_idx_first)
                                    view1["instance"].append(str(view_idx_first))
                                    view_idx_second = i * len(cond_rgbs_duster_input[i]) + second
                                    view2["idx"].append(view_idx_second)
                                    view2["instance"].append(str(view_idx_second))

                        view1["img"] = torch.stack(view1["img"])
                        view2["img"] = torch.stack(view2["img"])

                    with profiler.time_function("dust3r_inference"):
                        pred1, pred2 = self.model(view1, view2)

                    with profiler.time_function("dust3r_create_scene"):
                        scene = GlobalPointCloudOptimizer(view1, view2, pred1, pred2, neighbor_c2ws.view(-1, 4, 4),
                                                          neighbor_cameras.fx.view(-1, 1),
                                                          neighbor_cameras.fy.view(-1, 1),
                                                          neighbor_cameras.cx.view(-1, 1),
                                                          neighbor_cameras.cy.view(-1, 1),
                                                          cond_rgbs.view(-1, *cond_rgbs.shape[2:]).permute(0, 2, 3,
                                                                                                           1),
                                                          cond_rgbs_duster_input.shape[1],
                                                          verbose=not self.training)
                        alignment_loss, valid_alignment = scene.init_least_squares()

                        # min_conf_thr=self.min_conf_thr,
                        # pnp_method=self.pnp_method,
                        # use_elem_wise_pt_conf=self.use_elem_wise_pt_conf)

                # with torch.enable_grad(), profiler.time_function("dust3r_scale_opt"):
                #     alignment_loss = global_alignment_loop(scene, niter=self.alignment_iter,
                #                                            schedule=self.alignment_schedule,
                #                                            lr=self.alignment_lr)
                cond_pts3d = scene.pts3d_world().float()
                depth = scene.depth().float()
        else:
            alignment_loss = 0
            valid_alignment = torch.ones(len(target_cameras), dtype=torch.bool, device=cond_pts3d.device)
            camera_pos = neighbor_cameras.camera_to_worlds[..., 3]
            depth = (cond_pts3d - camera_pos.unsqueeze(2)).norm(dim=-1)
            depth = depth.view(-1, depth.shape[-1])

        scales = (depth * pixel_areas.view(depth.shape))

        for i in range(len(target_cameras)):
            if self.training and not valid_alignment[i]:
                continue
            with profiler.time_function("create_splats"):
                scene_rgbs = cond_rgbs[i].permute(0, 2, 3, 1).view(-1, 3)
                scene_features = cond_features[i].permute(0, 2, 3, 1)
                scene_features = scene_features.view(-1, scene_features.shape[-1])
                if self.model is None:
                    scene_xyz = cond_pts3d[i].view(-1, 3)
                else:
                    scene_xyz = cond_pts3d[i * neighbor_cameras.shape[1]:(i + 1) * neighbor_cameras.shape[1]] \
                        .reshape(-1, 3)

                scene_scales = scales[i * neighbor_cameras.shape[1]:(i + 1) * neighbor_cameras.shape[1]] \
                    .reshape(-1, 1).expand(-1, 3)
                scene_opacity = torch.ones_like(scene_xyz[..., :1])

            rendering = self.splat_gaussians(target_cameras[i], scene_xyz, scene_scales, scene_opacity, w2cs[i],
                                             scene_rgbs,
                                             scene_features if target_cameras_feature is None else None,
                                             return_depth)
            rgbs.append(rendering[RGB])
            accumulations.append(rendering[ACCUMULATION])
            if return_depth:
                depths.append(rendering[DEPTH])
            if target_cameras_feature is not None:
                rendering_features = self.splat_gaussians(target_cameras_feature[i], scene_xyz, scene_scales,
                                                          scene_opacity, w2cs[i], None, scene_features, False)
                features.append(rendering_features[FEATURES])
                accumulation_features.append(rendering_features[ACCUMULATION])
            else:
                features.append(rendering[FEATURES])

        outputs = {
            RGB: torch.stack(rgbs),
            FEATURES: torch.stack(features),
            ACCUMULATION: torch.stack(accumulations),
            ALIGNMENT_LOSS: alignment_loss,
            VALID_ALIGNMENT: valid_alignment,
        }

        outputs[ACCUMULATION_FEATURES] = torch.stack(accumulation_features) if target_cameras_feature is not None \
            else outputs[ACCUMULATION]

        if return_depth:
            outputs[DEPTH] = torch.stack(depths)

        # Eval done only on one batch
        if not self.training:
            scene_neighbor_rgbs = []
            scene_neighbor_depths = []
            scene_neighbor_accumulations = []
            for j in range(len(neighbor_cameras[i])):
                neighbor_renderings = self.splat_gaussians(neighbor_cameras[i, j], scene_xyz, scene_scales,
                                                           scene_opacity, neighbor_w2cs[i, j], scene_rgbs, None,
                                                           True)
                scene_neighbor_rgbs.append(neighbor_renderings[RGB])
                scene_neighbor_depths.append(neighbor_renderings[DEPTH])
                scene_neighbor_accumulations.append(neighbor_renderings[ACCUMULATION])

            neighbor_rgbs.append(torch.cat(scene_neighbor_rgbs, 1))
            neighbor_depths.append(torch.cat(scene_neighbor_depths, 1))
            neighbor_accumulations.append(torch.cat(scene_neighbor_accumulations, 1))

            outputs[NEIGHBOR_RESULTS] = {
                RGB: torch.stack(neighbor_rgbs),
                DEPTH: torch.stack(neighbor_depths),
                ACCUMULATION: torch.stack(neighbor_accumulations),
            }

        return outputs
