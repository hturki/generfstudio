from copy import deepcopy
from typing import Optional

import math
import torch
from dust3r.model import AsymmetricCroCo3DStereo
from gsplat import projection, isect_tiles, isect_offset_encode, rasterize_to_pixels
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.utils import profiler
from torch import nn
from torch.nn import functional as F

from generfstudio.fields.batched_pc_optimizer import GlobalPointCloudOptimizer
from generfstudio.generfstudio_constants import RGB, FEATURES, ACCUMULATION, ALIGNMENT_LOSS, NEIGHBOR_RESULTS, DEPTH, \
    VALID_ALIGNMENT, ACCUMULATION_FEATURES, DEPTH_GT


class Dust3rField(nn.Module):

    def __init__(
            self,
            model_name: str,
            in_feature_dim: int = 512,
            out_feature_dim: int = 128,
            image_dim: int = 224,
            depth_precomputed: bool = False,
    ) -> None:
        super().__init__()

        if not depth_precomputed:
            self.model = AsymmetricCroCo3DStereo.from_pretrained(model_name)
            for p in self.model.parameters():
                p.requires_grad_(False)
        else:
            self.model = None

        self.image_dim = image_dim
        # self.pnp_method = pnp_method
        self.out_feature_dim = out_feature_dim
        if out_feature_dim > 0 and in_feature_dim != out_feature_dim:
            self.feature_layer = nn.Linear(in_feature_dim, out_feature_dim)
        else:
            self.feature_layer = None

    @profiler.time_function
    @torch.cuda.amp.autocast(enabled=False)
    def splat_gaussians(self, cameras: Cameras, xyz: torch.Tensor, scales: torch.Tensor, conf: torch.Tensor,
                        w2cs: torch.Tensor, rgbs: Optional[torch.Tensor], features: Optional[torch.Tensor],
                        return_depth: bool, cameras_per_scene: int, bg_colors: Optional[torch.Tensor]):
        quats = torch.cat([torch.ones_like(conf), torch.zeros_like(xyz)], -1)

        fx = cameras.fx.view(-1, 1)
        zeros = torch.zeros_like(fx)
        ones = torch.ones_like(fx)

        K = torch.cat([
            torch.cat([cameras.fx.view(-1, 1), zeros, cameras.cx.view(-1, 1)], -1).unsqueeze(1),
            torch.cat([zeros, cameras.fy.view(-1, 1), cameras.cy.view(-1, 1)], -1).unsqueeze(1),
            torch.cat([zeros, zeros, ones], -1).unsqueeze(1)
        ], 1)

        W = cameras.width[0, 0].item()
        H = cameras.height[0, 0].item()
        radii, means2d, depths, conics, compensations = projection(xyz, None, quats, scales, w2cs, K, W, H,
                                                                   cameras_per_scene=cameras_per_scene)

        tile_size = 16
        tile_width = math.ceil(W / tile_size)
        tile_height = math.ceil(H / tile_size)
        tiles_per_gauss, isect_ids, gauss_ids = isect_tiles(
            means2d, radii, depths, tile_size, tile_width, tile_height
        )

        # assert (tiles_per_gauss > 0).any()

        isect_offsets = isect_offset_encode(isect_ids, len(cameras), tile_width, tile_height)

        inputs = []
        if rgbs is not None:
            inputs.append(rgbs.expand(cameras_per_scene, -1, -1) if cameras_per_scene > 1 else rgbs)
        if features is not None:
            inputs.append(features.float())
        if return_depth:
            inputs.append(depths.unsqueeze(-1))

        inputs = torch.cat(inputs, -1)

        if bg_colors is not None:
            bg_inputs = []
            if rgbs is not None:
                bg_inputs.append(bg_colors)
            if features is not None:
                bg_inputs.append(
                    torch.zeros(bg_colors.shape[0], features.shape[-1], dtype=bg_colors.dtype, device=bg_colors.device))
            if return_depth:
                bg_inputs.append(torch.zeros_like(bg_colors[..., :1]))

            bg_colors = torch.cat(bg_inputs, -1)
            if cameras_per_scene > 1:
                bg_colors = bg_colors.expand(cameras_per_scene, -1)

        splat_results, alpha = rasterize_to_pixels(
            means2d,
            conics,
            inputs,
            conf.squeeze(-1),
            W,
            H,
            tile_size,
            isect_offsets,
            gauss_ids,
            backgrounds=bg_colors,
        )

        outputs = {ACCUMULATION: alpha}

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
                cond_features: torch.Tensor, cond_pts3d: Optional[torch.Tensor], neighbor_depth: Optional[torch.Tensor],
                camera_rgbs: Optional[torch.Tensor], camera_fg_masks: Optional[torch.Tensor],
                neighbor_fg_masks: Optional[torch.Tensor], bg_colors: Optional[torch.Tensor], return_depth: bool,
                target_cameras_feature: Optional[Cameras] = None):

        image_dim = self.image_dim if cond_pts3d is None else target_cameras.height[0].item()

        if cond_rgbs.shape[-1] != image_dim:
            with profiler.time_function("interpolate_rgb_for_dust3r"), torch.no_grad():
                cond_rgbs = F.interpolate(cond_rgbs.view(-1, *cond_rgbs.shape[2:]), image_dim, mode="bicubic")
                cond_rgbs = cond_rgbs.view(len(target_cameras), -1, *cond_rgbs.shape[1:])

        if self.out_feature_dim > 0:
            with profiler.time_function("feature_layer_for_dust3r"):
                # bilinear interpolation can be expensive for high-dimensional features
                b, d, h, w = cond_features.shape
                cond_features = cond_features.permute(0, 2, 3, 1)  # b, d, h, w
                if self.feature_layer is not None:
                    cond_features = self.feature_layer(cond_features.reshape(-1, d)).view(b, h, w, -1)
                    cond_features = cond_features.permute(0, 3, 1, 2)  # b, d, h, w

            if cond_features.shape[-1] != image_dim:
                with profiler.time_function("interpolate_features_for_dust3r"):
                    cond_features = F.interpolate(cond_features, image_dim)
                    cond_features = cond_features.view(len(target_cameras), -1, *cond_features.shape[1:])

        with profiler.time_function("dust3r_camera_prep"):
            if neighbor_cameras.height[0, 0].item() != image_dim:
                neighbor_cameras = deepcopy(neighbor_cameras)
                neighbor_cameras.rescale_output_resolution(image_dim / neighbor_cameras.height[0, 0].item())

            neighbor_c2ws = torch.eye(4, device=neighbor_cameras.device).view(1, 1, 4, 4).repeat(
                neighbor_cameras.shape[0], neighbor_cameras.shape[1], 1, 1)
            neighbor_c2ws[:, :, :3] = neighbor_cameras.camera_to_worlds
            neighbor_c2ws[:, :, :, 1:3] *= -1  # opengl to opencv (what dust3r expects)

            R = target_cameras.camera_to_worlds[:, :3, :3].clone()  # 3 x 3, clone to avoid changing original cameras
            R[:, :, 1:3] *= -1  # opengl to opencv (what gsplat expects)
            T = target_cameras.camera_to_worlds[:, :3, 3:]  # 3 x 1

            w2cs = torch.eye(4, device=R.device).unsqueeze(0).repeat(len(target_cameras), 1, 1)
            R_inv = R[:, :3, :3].transpose(1, 2)
            w2cs[:, :3, :3] = R_inv
            w2cs[:, :3, 3:] = -torch.bmm(R_inv, T)

        if not self.training:
            neighbor_w2cs = torch.eye(4, device=R.device).view(1, 1, 4, 4).repeat(neighbor_cameras.shape[0],
                                                                                  neighbor_cameras.shape[1], 1, 1)

            # We can reuse neighbor_c2ws since dust3r and gsplat use the same coordinate convention
            neighbor_R_inv = neighbor_c2ws[:, :, :3, :3].transpose(2, 3)
            neighbor_w2cs[:, :, :3, :3] = neighbor_R_inv
            neighbor_T = neighbor_c2ws[:, :, :3, 3:]
            neighbor_inv_T = -torch.bmm(neighbor_R_inv.view(-1, 3, 3), neighbor_T.view(-1, 3, 1))
            neighbor_w2cs[:, :, :3, 3:] = neighbor_inv_T.view(neighbor_w2cs[:, :, :3, 3:].shape)

        dust3r_on_gt = False
        if cond_pts3d is None:
            with profiler.time_function("dust3r_for_all_cameras"):
                with torch.no_grad():
                    cond_rgbs_dust3r_input = cond_rgbs * 2 - 1

                    if neighbor_fg_masks is not None and neighbor_fg_masks.shape[-1] != image_dim:
                        with torch.no_grad():
                            neighbor_fg_masks = F.interpolate(
                                neighbor_fg_masks.view(-1, *neighbor_fg_masks.shape[2:]).unsqueeze(1).float(),
                                image_dim)
                            neighbor_fg_masks = neighbor_fg_masks.squeeze(1).view(len(target_cameras), -1,
                                                                                  image_dim * image_dim).bool()

                    if neighbor_cameras.shape[1] == 1:
                        dust3r_on_gt = True
                        if camera_rgbs.shape[-1] != image_dim:
                            with torch.no_grad():
                                camera_rgbs = F.interpolate(camera_rgbs, image_dim, mode="bicubic")

                        if camera_fg_masks is not None and camera_fg_masks.shape[-1] != image_dim:
                            with torch.no_grad():
                                camera_fg_masks = F.interpolate(camera_fg_masks.unsqueeze(1).float(), image_dim)
                                camera_fg_masks = camera_fg_masks.squeeze(1).view(len(target_cameras), -1).bool()

                        camera_rgbs_dust3r_input = camera_rgbs * 2 - 1

                        pc_c2ws = []
                        pc_fx = []
                        pc_fy = []
                        pc_cx = []
                        pc_cy = []
                        if camera_fg_masks is not None:
                            pc_fg_masks = []

                        with profiler.time_function("dust3r_load_images"):
                            view1 = {"img": [], "idx": [], "instance": []}
                            view2 = {"img": [], "idx": [], "instance": []}
                            for pi, (cond_rgb, camera_rgb) in enumerate(
                                    zip(cond_rgbs_dust3r_input, camera_rgbs_dust3r_input)):
                                pc_c2ws.append(neighbor_c2ws[pi, 0])
                                pc_fx.append(neighbor_cameras.fx[pi, 0])
                                pc_fy.append(neighbor_cameras.fy[pi, 0])
                                pc_cx.append(neighbor_cameras.cx[pi, 0])
                                pc_cy.append(neighbor_cameras.cy[pi, 0])

                                camera_c2w = torch.eye(4, device=R.device)
                                camera_c2w[:3, :3] = R[pi]
                                camera_c2w[:3, 3:] = T[pi]
                                pc_c2ws.append(camera_c2w)
                                pc_fx.append(target_cameras.fx[pi] * image_dim / target_cameras.width[pi])
                                pc_fy.append(target_cameras.fy[pi] * image_dim / target_cameras.height[pi])
                                pc_cx.append(target_cameras.cx[pi] * image_dim / target_cameras.width[pi])
                                pc_cy.append(target_cameras.cy[pi] * image_dim / target_cameras.height[pi])

                                if camera_fg_masks is not None:
                                    pc_fg_masks.append(neighbor_fg_masks[pi, 0])
                                    pc_fg_masks.append(camera_fg_masks[pi])

                                view1["img"].append(cond_rgb[0])
                                view2["img"].append(camera_rgb)
                                view1["idx"].append(pi * 2)
                                view1["instance"].append(str(pi * 2))
                                view2["idx"].append(pi * 2 + 1)
                                view2["instance"].append(str(pi * 2 + 1))

                                view2["img"].append(cond_rgb[0])
                                view1["img"].append(camera_rgb)
                                view2["idx"].append(pi * 2)
                                view2["instance"].append(str(pi * 2))
                                view1["idx"].append(pi * 2 + 1)
                                view1["instance"].append(str(pi * 2 + 1))

                            view1["img"] = torch.stack(view1["img"])
                            view2["img"] = torch.stack(view2["img"])

                        with profiler.time_function("dust3r_inference"):
                            pred1, pred2 = self.model(view1, view2)

                        scene = GlobalPointCloudOptimizer(view1, view2, pred1, pred2, torch.stack(pc_c2ws),
                                                          torch.FloatTensor(pc_fx).unsqueeze(-1).to(R.device),
                                                          torch.FloatTensor(pc_fy).unsqueeze(-1).to(R.device),
                                                          torch.FloatTensor(pc_cx).unsqueeze(-1).to(R.device),
                                                          torch.FloatTensor(pc_cy).unsqueeze(-1).to(R.device),
                                                          2, torch.stack(pc_fg_masks)
                                                          if camera_fg_masks is not None else None,
                                                          verbose=not self.training)
                        alignment_loss, valid_alignment = scene.init_least_squares()
                        cond_pts3d = scene.pts3d_world()[::2].float().detach()
                        depth = scene.depth()[::2].float().detach()

                        # zeros = torch.zeros_like(target_cameras.fx)
                        # ones = torch.ones_like(target_cameras.fx)
                        #
                        # K = torch.cat([
                        #     torch.cat([target_cameras.fx * self.image_dim / target_cameras.width, zeros,
                        #                target_cameras.cx * self.image_dim / target_cameras.width], -1).unsqueeze(1),
                        #     torch.cat([zeros, target_cameras.fy * self.image_dim / target_cameras.height,
                        #                target_cameras.cy * self.image_dim / target_cameras.height], -1).unsqueeze(1),
                        #     torch.cat([zeros, zeros, ones], -1).unsqueeze(1)
                        # ], 1)
                        #
                        # K_inv = K.inverse().permute(0, 2, 1)
                        #
                        # with torch.cuda.amp.autocast(enabled=False):
                        #     pixels_uncal = torch.matmul(self.pixels_homo, K_inv)
                        #     pnp = efficient_pnp(pred2["pts3d_in_other_view"].view(cond_rgbs.shape[0], -1, 3),
                        #                         pixels_uncal[..., :2] / pixels_uncal[..., 2:],
                        #                         pred2["conf"].log().view(cond_rgbs.shape[0], -1))
                        #
                        #
                        #
                        # import pdb; pdb.set_trace()
                        # rot = pnp.R.transpose(1, 2)  # (B, 3, 3)
                        # trans = -torch.bmm(rot, pnp.T.unsqueeze(-1))  # (B, 3, 1)
                        # pnp_c2ws = torch.cat((rot, trans), dim=-1)
                    else:
                        with profiler.time_function("dust3r_load_images"):
                            view1 = {"img": [], "idx": [], "instance": []}
                            view2 = {"img": [], "idx": [], "instance": []}
                            for i in range(len(target_cameras)):
                                for first in range(len(cond_rgbs_dust3r_input[i])):
                                    for second in range(len(cond_rgbs_dust3r_input[i])):
                                        if first == second:
                                            continue
                                        view1["img"].append(cond_rgbs_dust3r_input[i, first])
                                        view2["img"].append(cond_rgbs_dust3r_input[i, second])
                                        view_idx_first = i * len(cond_rgbs_dust3r_input[i]) + first
                                        view1["idx"].append(view_idx_first)
                                        view1["instance"].append(str(view_idx_first))
                                        view_idx_second = i * len(cond_rgbs_dust3r_input[i]) + second
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
                                                              # cond_rgbs.view(-1, *cond_rgbs.shape[2:]).permute(0, 2, 3,
                                                              #                                                  1),
                                                              cond_rgbs_dust3r_input.shape[1],
                                                              neighbor_fg_masks.view(-1, image_dim,
                                                                                     image_dim),
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
            depth = neighbor_depth.view(len(target_cameras), -1)
            valid_alignment = depth.max(dim=-1)[0] > 0

        # If none are valid, discard the entire batch when computing loss
        if self.training and torch.any(valid_alignment):
            depth = depth[valid_alignment]
            neighbor_cameras = neighbor_cameras[valid_alignment]
            cond_pts3d = cond_pts3d[valid_alignment]
            cond_rgbs = cond_rgbs[valid_alignment]
            cond_features = cond_features[valid_alignment]
            target_cameras = target_cameras[valid_alignment]
            w2cs = w2cs[valid_alignment]
            if target_cameras_feature is not None:
                target_cameras_feature = target_cameras_feature[valid_alignment]
            if neighbor_fg_masks is not None:
                neighbor_fg_masks = neighbor_fg_masks[valid_alignment]
                bg_colors = bg_colors[valid_alignment]

        scales = depth / neighbor_cameras.fx.view(-1, 1)
        # scales = (depth * pixel_areas.view(depth.shape))
        xyz = cond_pts3d.view(len(target_cameras), -1, 3)
        rgbs = cond_rgbs.permute(0, 1, 3, 4, 2).view(len(target_cameras), -1, 3)

        if self.out_feature_dim > 0:
            features = cond_features.permute(0, 1, 3, 4, 2)
            features = features.view(len(target_cameras), -1, features.shape[-1])
        else:
            features = None

        scales = scales.view(len(target_cameras), -1, 1).expand(-1, -1, 3)

        conf = neighbor_fg_masks.view(xyz[..., :1].shape).float() if neighbor_fg_masks is not None \
            else torch.ones_like(xyz[..., :1])

        rendering = self.splat_gaussians(
            target_cameras,
            xyz,
            scales,
            conf,
            w2cs,
            rgbs,
            features if target_cameras_feature is None else None,
            return_depth,
            1,
            bg_colors,
        )

        outputs = {
            RGB: rendering[RGB],
            ACCUMULATION: rendering[ACCUMULATION],
            ALIGNMENT_LOSS: alignment_loss,
            VALID_ALIGNMENT: valid_alignment,
        }

        if dust3r_on_gt:
            outputs[DEPTH_GT] = scene.depth()[1::2].detach()
            if self.training:
                outputs[DEPTH_GT] = outputs[DEPTH_GT][valid_alignment]

        if return_depth:
            outputs[DEPTH] = rendering[DEPTH]

        if target_cameras_feature is not None:
            rendering_feature = self.splat_gaussians(
                target_cameras_feature,
                xyz,
                scales,
                conf,
                w2cs,
                None,
                features,
                False,
                1,
                None
            )
            outputs[FEATURES] = rendering_feature[FEATURES]
            outputs[ACCUMULATION_FEATURES] = rendering_feature[ACCUMULATION]
        else:
            outputs[FEATURES] = rendering[FEATURES]
            outputs[ACCUMULATION_FEATURES] = rendering[ACCUMULATION]

        if not self.training:
            neighbor_renderings = self.splat_gaussians(
                neighbor_cameras.flatten(),
                xyz,
                scales,
                conf,
                neighbor_w2cs.squeeze(0),
                rgbs,
                None,
                True,
                neighbor_cameras.shape[1],
                bg_colors
            )

            outputs[NEIGHBOR_RESULTS] = {
                RGB: torch.cat([*neighbor_renderings[RGB]], 1),
                DEPTH: torch.cat([*neighbor_renderings[DEPTH]], 1),
                ACCUMULATION: torch.cat([*neighbor_renderings[ACCUMULATION]], 1),
            }

        return outputs
