from copy import deepcopy
from typing import Optional, Callable, Tuple

import math
import torch
from dust3r.model import AsymmetricCroCo3DStereo
from gsplat import fully_fused_projection, isect_tiles, isect_offset_encode, rasterize_to_pixels
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.utils import profiler
from torch import nn
from torch.nn import functional as F

from generfstudio.fields.batched_pc_optimizer import GlobalPointCloudOptimizer, fast_depthmap_to_pts3d
from generfstudio.generfstudio_constants import RGB, FEATURES, ACCUMULATION, ALIGNMENT_LOSS, NEIGHBOR_RESULTS, DEPTH, \
    VALID_ALIGNMENT, ACCUMULATION_FEATURES, DEPTH_GT


class Dust3rField(nn.Module):

    def __init__(
            self,
            model_name: str,
            in_feature_dim: int = 512,
            out_feature_dim: int = 128,
            noisy_cond_views: int = 0,
            depth_precomputed: bool = False,
    ) -> None:
        super().__init__()

        if not depth_precomputed:
            self.model = AsymmetricCroCo3DStereo.from_pretrained(model_name)
            for p in self.model.parameters():
                p.requires_grad_(False)
        else:
            self.model = None

        self.out_feature_dim = out_feature_dim
        if out_feature_dim > 0 and in_feature_dim != out_feature_dim:
            self.feature_layer = nn.Linear(in_feature_dim, out_feature_dim)
        else:
            self.feature_layer = None

        self.noisy_cond_views = noisy_cond_views

    @profiler.time_function
    @torch.cuda.amp.autocast(enabled=False)
    def splat_gaussians(self, w2cs: torch.Tensor, fx: torch.Tensor, fy: torch.Tensor, cx: torch.Tensor,
                        cy: torch.Tensor, xyz: torch.Tensor, scales: torch.Tensor, conf: torch.Tensor,
                        rgbs: Optional[torch.Tensor], features: Optional[torch.Tensor], camera_dim: int,
                        return_depth: bool, cameras_per_scene: int, bg_colors: Optional[torch.Tensor]):
        quats = torch.cat([torch.ones_like(conf), torch.zeros_like(xyz)], -1)

        zeros = torch.zeros_like(fx)
        ones = torch.ones_like(fx)

        K = torch.cat([
            torch.cat([fx, zeros, cx], -1).unsqueeze(1),
            torch.cat([zeros, fy, cy], -1).unsqueeze(1),
            torch.cat([zeros, zeros, ones], -1).unsqueeze(1)
        ], 1)

        radii, means2d, depths, conics, compensations = fully_fused_projection(
            xyz, None, quats, scales, w2cs, K, camera_dim, camera_dim, cameras_per_scene=cameras_per_scene)

        tile_size = 16
        tile_width = math.ceil(camera_dim / tile_size)
        tile_height = math.ceil(camera_dim / tile_size)
        tiles_per_gauss, isect_ids, gauss_ids = isect_tiles(
            means2d, radii, depths, tile_size, tile_width, tile_height
        )

        # assert (tiles_per_gauss > 0).any()

        isect_offsets = isect_offset_encode(isect_ids, w2cs.shape[0], tile_width, tile_height)

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
            camera_dim,
            camera_dim,
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

    # Intrinsics should be passed for image size inferred by rgb
    @profiler.time_function
    @torch.inference_mode()
    def get_pts3d_and_depth(self, rgbs: torch.Tensor, fg_masks: Optional[torch.Tensor], c2ws: torch.Tensor,
                            fx: torch.Tensor, fy: torch.Tensor, cx: torch.Tensor, cy: torch.Tensor):
        rgbs = rgbs * 2 - 1

        view1 = {"img": [], "idx": [], "instance": []}
        view2 = {"img": [], "idx": [], "instance": []}
        for i in range(rgbs.shape[0]):
            for first in range(rgbs.shape[1]):
                for second in range(rgbs.shape[1]):
                    if first == second:
                        continue
                    view1["img"].append(rgbs[i, first])
                    view2["img"].append(rgbs[i, second])
                    view_idx_first = i * rgbs.shape[1] + first
                    view1["idx"].append(view_idx_first)
                    view1["instance"].append(str(view_idx_first))
                    view_idx_second = i * rgbs.shape[1] + second
                    view2["idx"].append(view_idx_second)
                    view2["instance"].append(str(view_idx_second))

        view1["img"] = torch.stack(view1["img"])
        view2["img"] = torch.stack(view2["img"])

        with profiler.time_function("dust3r_inference"):
            pred1, pred2 = self.model(view1, view2)

        scene = GlobalPointCloudOptimizer(
            view1, view2, pred1, pred2, c2ws.view(-1, 4, 4), fx.view(-1, 1), fy.view(-1, 1), cx.view(-1, 1),
            cy.view(-1, 1), rgbs.shape[1], fg_masks.flatten(0, 1) if fg_masks is not None else None,
            verbose=not self.training)

        alignment_loss, valid_alignment = scene.init_least_squares()
        pts3d = scene.pts3d_world().detach()
        depth = scene.depth().detach()

        return pts3d, depth, alignment_loss, valid_alignment

    @profiler.time_function
    def forward(self, target_cameras: Cameras, neighbor_cameras: Cameras,
                target_rgbs: Optional[torch.Tensor], neighbor_rgbs: torch.Tensor,
                target_fg_masks: Optional[torch.Tensor], neighbor_fg_masks: Optional[torch.Tensor],
                neighbor_features: Optional[torch.Tensor],
                cond_pts3d: Optional[torch.Tensor], cond_depth: Optional[torch.Tensor],
                bg_colors: Optional[torch.Tensor], return_target_depth: bool,
                to_noise_fn: Optional[
                    Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]],
                rgb_resolution: int, feature_resolution: int):

        image_dim = self.image_dim if cond_pts3d is None else target_cameras.height[0].item()

        if neighbor_rgbs.shape[-1] != image_dim:
            with torch.no_grad():
                neighbor_rgbs = F.interpolate(neighbor_rgbs.view(-1, *neighbor_rgbs.shape[2:]), image_dim,
                                              mode="bicubic")
                neighbor_rgbs = neighbor_rgbs.view(len(target_cameras), -1, *neighbor_rgbs.shape[1:])

        if self.out_feature_dim > 0:
            # bilinear interpolation can be expensive for high-dimensional features
            b, d, h, w = neighbor_features.shape
            neighbor_features = neighbor_features.permute(0, 2, 3, 1)  # b, d, h, w
            if self.feature_layer is not None:
                neighbor_features = self.feature_layer(neighbor_features.reshape(-1, d)).view(b, h, w, -1)
                neighbor_features = neighbor_features.permute(0, 3, 1, 2)  # b, d, h, w

            if neighbor_features.shape[-1] != image_dim:
                neighbor_features = F.interpolate(neighbor_features, image_dim)
                neighbor_features = neighbor_features.view(len(target_cameras), -1, *neighbor_features.shape[1:])

        w2cs = torch.eye(4, device=target_cameras.device).unsqueeze(0).repeat(len(target_cameras), 1, 1)

        R = target_cameras.camera_to_worlds[:, :3, :3].clone()  # 3 x 3, clone to avoid changing original cameras
        R[:, :, 1:3] *= -1  # opengl to opencv
        T = target_cameras.camera_to_worlds[:, :3, 3:]  # 3 x 1

        R_inv = R[:, :3, :3].transpose(1, 2)
        w2cs[:, :3, :3] = R_inv
        w2cs[:, :3, 3:] = -torch.bmm(R_inv, T)

        neighbor_cameras_dust3r = neighbor_cameras
        if neighbor_cameras_dust3r.height[0, 0].item() != image_dim:
            neighbor_cameras_dust3r = deepcopy(neighbor_cameras_dust3r)
            neighbor_cameras_dust3r.rescale_output_resolution(image_dim / neighbor_cameras_dust3r.height[0, 0].item())

        neighbor_c2ws = torch.eye(4, device=neighbor_cameras_dust3r.device).view(1, 1, 4, 4).repeat(
            neighbor_cameras_dust3r.shape[0], neighbor_cameras_dust3r.shape[1], 1, 1)
        neighbor_c2ws[:, :, :3] = neighbor_cameras_dust3r.camera_to_worlds
        neighbor_c2ws[:, :, :, 1:3] *= -1  # opengl to opencv

        if self.noisy_cond_views > 0 or (not self.training):
            neighbor_w2cs = torch.eye(4, device=R.device).view(1, 1, 4, 4).repeat(neighbor_cameras_dust3r.shape[0],
                                                                                  neighbor_cameras_dust3r.shape[1], 1,
                                                                                  1)

            # We can reuse neighbor_c2ws since dust3r and gsplat use the same coordinate convention
            neighbor_R_inv = neighbor_c2ws[:, :, :3, :3].transpose(2, 3)
            neighbor_w2cs[:, :, :3, :3] = neighbor_R_inv
            neighbor_T = neighbor_c2ws[:, :, :3, 3:]
            neighbor_inv_T = -torch.bmm(neighbor_R_inv.view(-1, 3, 3), neighbor_T.view(-1, 3, 1))
            neighbor_w2cs[:, :, :3, 3:] = neighbor_inv_T.view(neighbor_w2cs[:, :, :3, 3:].shape)

        if cond_pts3d is None:
            with torch.no_grad():
                if target_rgbs.shape[-1] != image_dim:
                    target_rgbs = F.interpolate(target_rgbs, image_dim, mode="bicubic")

                camera_rgbs_dust3r_input = target_rgbs * 2 - 1
                cond_rgbs_dust3r_input = neighbor_rgbs * 2 - 1

                if target_fg_masks is not None and target_fg_masks.shape[-1] != image_dim:
                    target_fg_masks = F.interpolate(target_fg_masks.unsqueeze(1).float(), image_dim)
                    target_fg_masks = target_fg_masks.squeeze(1).view(len(target_cameras), -1).bool()

                if neighbor_fg_masks is not None and neighbor_fg_masks.shape[-1] != image_dim:
                    neighbor_fg_masks = F.interpolate(
                        neighbor_fg_masks.view(-1, *neighbor_fg_masks.shape[2:]).unsqueeze(1).float(), image_dim)
                    neighbor_fg_masks = neighbor_fg_masks.squeeze(1).view(len(target_cameras), -1,
                                                                          image_dim * image_dim).bool()

                c2ws = torch.eye(4, device=R.device).view(1, 4, 4).repeat(target_cameras.shape[0], 1, 1)
                c2ws[:, :3, :3] = R
                c2ws[:, :3, 3:] = T

                view1 = {"img": [], "idx": [], "instance": []}
                view2 = {"img": [], "idx": [], "instance": []}
                for i in range(len(target_cameras)):
                    images_in_scene = [camera_rgbs_dust3r_input[i], *cond_rgbs_dust3r_input[i]]
                    for first in range(len(images_in_scene)):
                        for second in range(len(images_in_scene)):
                            if first == second:
                                continue
                            view1["img"].append(images_in_scene[first])
                            view2["img"].append(images_in_scene[second])
                            view_idx_first = i * len(images_in_scene) + first
                            view1["idx"].append(view_idx_first)
                            view1["instance"].append(str(view_idx_first))
                            view_idx_second = i * len(images_in_scene) + second
                            view2["idx"].append(view_idx_second)
                            view2["instance"].append(str(view_idx_second))

                view1["img"] = torch.stack(view1["img"])
                view2["img"] = torch.stack(view2["img"])

                with profiler.time_function("dust3r_inference"):
                    pred1, pred2 = self.model(view1, view2)

                scene = GlobalPointCloudOptimizer(
                    view1, view2, pred1, pred2,
                    torch.cat([c2ws.unsqueeze(1), neighbor_c2ws], 1).view(-1, 4, 4),
                    torch.cat(
                        [(target_cameras.fx * image_dim / target_cameras.width).unsqueeze(1),
                         neighbor_cameras_dust3r.fx], 1).view(-1, 1),
                    torch.cat(
                        [(target_cameras.fy * image_dim / target_cameras.height).unsqueeze(1),
                         neighbor_cameras_dust3r.fy], 1).view(-1, 1),
                    torch.cat(
                        [(target_cameras.cx * image_dim / target_cameras.width).unsqueeze(1),
                         neighbor_cameras_dust3r.cx], 1).view(-1, 1),
                    torch.cat(
                        [(target_cameras.cy * image_dim / target_cameras.height).unsqueeze(1),
                         neighbor_cameras_dust3r.cy], 1).view(-1, 1),
                    cond_rgbs_dust3r_input.shape[1] + 1,
                    torch.cat([target_fg_masks.unsqueeze(1), neighbor_fg_masks], 1).flatten(0, 1)
                    if target_fg_masks is not None else None,
                    verbose=not self.training)

                alignment_loss, valid_alignment = scene.init_least_squares()
                all_pts3d = scene.pts3d_world().detach()
                all_depth = scene.depth().detach()

                if self.training:
                    pts3d = all_pts3d.view(target_cameras.shape[0], -1, *all_pts3d.shape[1:])[:, 1:]
                    depth = all_depth.view(target_cameras.shape[0], -1, *all_depth.shape[1:])[:, 1:]
                else:
                    pts3d = all_pts3d.view(target_cameras.shape[0], -1, *all_pts3d.shape[1:])[:, :1]
                    depth = all_depth.view(target_cameras.shape[0], -1, *all_depth.shape[1:])[:, :1]

                if self.noisy_cond_views > 0 and to_noise_fn is not None:
                    pixels = scene.pixels
                    noisy_focals = scene.focals.view(target_cameras.shape[0], -1, 2)[:,
                                   1:1 + self.noisy_cond_views].view(-1, 2)
                    noisy_pp = scene.pp.view(target_cameras.shape[0], -1, 2)[:,
                               1:1 + self.noisy_cond_views].view(-1, 2)
        else:
            import pdb;
            pdb.set_trace()
            pts3d = cond_pts3d
            alignment_loss = 0
            depth = cond_depth  # .view(len(target_cameras), -1)
            valid_alignment = depth.max(dim=-1)[0] > 0

            # neighbor cameras should be set to image dim (256 when using gt depth)
            noisy_focals = torch.cat([neighbor_cameras_dust3r.fx, neighbor_cameras_dust3r.fy], -1)[:,
                           :self.noisy_cond_views].view(-1, 2)
            noisy_pp = torch.cat([neighbor_cameras_dust3r.cx, neighbor_cameras_dust3r.cy], -1)[:,
                       :self.noisy_cond_views].view(-1, 2)

            pixels_im = torch.stack(
                torch.meshgrid(torch.arange(image_dim, device=noisy_focals.device, dtype=torch.float32),
                               torch.arange(image_dim, device=noisy_focals.device, dtype=torch.float32),
                               indexing="ij")).permute(2, 1, 0)
            pixels = pixels_im.reshape(-1, 2)

        rgbs = neighbor_rgbs
        conf = neighbor_fg_masks.view(pts3d[..., :1].shape).float() if neighbor_fg_masks is not None \
            else torch.ones_like(pts3d[..., :1])
        if neighbor_fg_masks is not None:
            conf_acc = torch.ones_like(pts3d[..., :1])
        if self.noisy_cond_views > 0 and to_noise_fn is not None:
            rgb_to_noise = rgbs[:, :self.noisy_cond_views].flatten(0, 1)
            depth_to_noise = depth[:, :self.noisy_cond_views].flatten(0, 1).view(-1, 1, image_dim, image_dim)
            noisy_rgb, noisy_depth, noisy_opacity = to_noise_fn(rgb_to_noise, depth_to_noise)

            noisy_pts3d_cam = fast_depthmap_to_pts3d(noisy_depth.view(-1, image_dim * image_dim),
                                                     pixels,
                                                     noisy_focals,
                                                     noisy_pp)
            noisy_pts3d = neighbor_c2ws[:, :self.noisy_cond_views].view(-1, 4, 4) @ torch.cat(
                [noisy_pts3d_cam, torch.ones_like(noisy_pts3d_cam[..., :1])], -1).transpose(1, 2)
            noisy_pts3d = noisy_pts3d.transpose(1, 2)[..., :3].view(target_cameras.shape[0], self.noisy_cond_views, -1,
                                                                    3)
            pts3d = torch.cat([noisy_pts3d, pts3d[:, self.noisy_cond_views:]], 1)
            depth = torch.cat([noisy_depth.view(target_cameras.shape[0], self.noisy_cond_views, -1),
                               depth[:, self.noisy_cond_views:]], 1)
            rgbs = torch.cat([noisy_rgb.view(target_cameras.shape[0], self.noisy_cond_views, 3, image_dim, image_dim),
                              rgbs[:, self.noisy_cond_views:]], 1)
            conf[:, :self.noisy_cond_views] *= noisy_opacity.view(-1, 1, 1, 1)
            if neighbor_fg_masks is not None:
                conf_acc[:, :self.noisy_cond_views] *= noisy_opacity.view(-1, 1, 1, 1)

        # If none are valid, discard the entire batch when computing loss
        if self.training and torch.any(valid_alignment):
            neighbor_cameras_dust3r = neighbor_cameras_dust3r[valid_alignment]
            pts3d = pts3d[valid_alignment]
            depth = depth[valid_alignment]
            rgbs = rgbs[valid_alignment]
            conf = conf[valid_alignment]

            if self.out_feature_dim > 0:
                neighbor_features = neighbor_features[valid_alignment]

            target_cameras = target_cameras[valid_alignment]
            w2cs = w2cs[valid_alignment]
            if target_cameras_feature is not None:
                target_cameras_feature = target_cameras_feature[valid_alignment]
            if neighbor_fg_masks is not None:
                neighbor_fg_masks = neighbor_fg_masks[valid_alignment]
                bg_colors = bg_colors[valid_alignment]

        # scales = (depth * pixel_areas.view(depth.shape))
        xyz = pts3d.reshape(len(target_cameras), -1, 3).float()
        rgbs = rgbs.permute(0, 1, 3, 4, 2).reshape(len(target_cameras), -1, 3)
        conf = conf.reshape(len(target_cameras), -1, 1).float()

        if self.out_feature_dim > 0:
            features = neighbor_features.permute(0, 1, 3, 4, 2)
            features = features.view(len(target_cameras), -1, features.shape[-1])
        else:
            features = None

        outputs = {
            ALIGNMENT_LOSS: alignment_loss,
            VALID_ALIGNMENT: valid_alignment,
        }

        scales = depth.flatten(0, 1) / neighbor_cameras_dust3r.fx.view(-1, 1)
        scales = scales.view(len(target_cameras), -1, 1).expand(-1, -1, 3)

        rendering = self.splat_gaussians(
            target_cameras,
            xyz,
            scales,
            conf,
            w2cs,
            rgbs,
            features if target_cameras_feature is None else None,
            return_target_depth,
            1,
            bg_colors,
        )

        outputs[RGB] = rendering[RGB]
        if ACCUMULATION not in outputs:
            outputs[ACCUMULATION] = rendering[ACCUMULATION]

        if cond_pts3d is None:
            outputs[DEPTH_GT] = all_depth[::(neighbor_cameras_dust3r.shape[1] + 1)].detach()
            if self.training:
                outputs[DEPTH_GT] = outputs[DEPTH_GT][valid_alignment]

        if return_target_depth:
            outputs[DEPTH] = rendering[DEPTH]

        if neighbor_fg_masks is not None:
            with torch.no_grad():
                depth_acc = torch.where(neighbor_fg_masks.view(xyz[..., 0].shape), depth, 1000)

                scales_acc = depth_acc.flatten(0, 1) / neighbor_cameras_dust3r.fx.view(-1, 1)
                scales_acc = scales_acc.view(len(target_cameras), -1, 1).expand(-1, -1, 3)
                rendering_acc = self.splat_gaussians(
                    target_cameras_feature if target_cameras_feature is not None else target_cameras,
                    xyz,
                    scales_acc,
                    conf_acc,
                    w2cs,
                    None,
                    None,
                    return_target_depth,
                    1,
                    None,
                )
                outputs[ACCUMULATION_FEATURES] = rendering_acc[ACCUMULATION]

        if target_cameras_feature is not None and (self.out_feature_dim > 0 or neighbor_fg_masks is None):
            rendering_feature = self.splat_gaussians(
                target_cameras_feature,
                xyz,
                scales,
                conf,
                w2cs,
                None,
                features,
                features is None,
                1,
                None
            )

            if self.out_feature_dim > 0:
                outputs[FEATURES] = rendering_feature[FEATURES]

            if neighbor_fg_masks is None:
                outputs[ACCUMULATION_FEATURES] = rendering_feature[ACCUMULATION]
        else:
            if self.out_feature_dim > 0:
                outputs[FEATURES] = rendering[FEATURES]

            if neighbor_fg_masks is None:
                outputs[ACCUMULATION_FEATURES] = rendering[ACCUMULATION]

        if not self.training:
            neighbor_renderings = self.splat_gaussians(
                neighbor_cameras_dust3r.flatten(),
                xyz,
                scales,
                conf,
                neighbor_w2cs.squeeze(0),
                rgbs,
                None,
                True,
                neighbor_cameras_dust3r.shape[1],
                bg_colors
            )

            outputs[NEIGHBOR_RESULTS] = {
                RGB: torch.cat([*neighbor_renderings[RGB]], 1),
                DEPTH: torch.cat([*neighbor_renderings[DEPTH]], 1),
                ACCUMULATION: torch.cat([*neighbor_renderings[ACCUMULATION]], 1),
            }

        return outputs
