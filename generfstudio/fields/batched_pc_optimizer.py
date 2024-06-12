from typing import Dict, Tuple, Optional

import cv2
import math
import torch
from nerfstudio.utils.rich_utils import CONSOLE

try:
    from pytorch3d.ops import efficient_pnp, corresponding_points_alignment
except:
    pass

from torch import nn


# This class assumes that the scene graph is complete for each scene, ie: there is an observation for each image pair
# in the scene
class GlobalPointCloudOptimizer(nn.Module):

    def __init__(self, view1: Dict, view2: Dict, pred1: Dict, pred2: Dict, poses: torch.Tensor,
                 fx: torch.Tensor,  fy: torch.Tensor, cx: torch.Tensor, cy: torch.Tensor,
                 poses_per_scene: int, fg_masks: Optional[torch.Tensor],
                 filter_dist_quantiles: Optional[Tuple[float, float]] = None,
                 filter_weight_quantile: Optional[float] = None, verbose: bool = False):
    # def __init__(self, view1: Dict, view2: Dict, pred1: Dict, pred2: Dict, poses: torch.Tensor, fx: torch.Tensor,
    #                  fy: torch.Tensor, cx: torch.Tensor, cy: torch.Tensor, poses_per_scene: int,
    #                  filter_dist_quantiles: Optional[Tuple[float, float]] = None,
    #                  filter_weight_quantile: Optional[float] = None, verbose: bool = False):
        super().__init__()

        num_preds = pred1["conf"].shape[0]
        neighbor_pose_count = poses_per_scene - 1
        assert num_preds % neighbor_pose_count == 0
        self.poses_per_scene = poses_per_scene

        pts3d_i = pred1["pts3d"].view(num_preds, -1, 3)
        pts3d_j = pred2["pts3d_in_other_view"].view(num_preds, -1, 3)

        # avg_dist = torch.cat([pts3d_i, pts3d_j], 1).norm(dim=-1, keepdim=True).mean(dim=1, keepdim=True)
        self.pts3d_i = pts3d_i  # / avg_dist
        self.pts3d_j = pts3d_j  # / avg_dist

        assert filter_dist_quantiles is None or filter_weight_quantile is None, "Can use one filter at most"
        self.filter_dist_quantiles = torch.FloatTensor(filter_dist_quantiles).to(self.pts3d_i.device) \
            if filter_dist_quantiles is not None else None
        self.filter_weight_quantile = torch.FloatTensor(filter_weight_quantile).to(self.pts3d_i.device) \
            if filter_weight_quantile is not None else None

        self.weight_i = pred1["conf"].log().view(num_preds, -1)
        self.weight_j = pred2["conf"].log().view(num_preds, -1)

        self.view1_idx = torch.LongTensor(view1["idx"]).to(poses.device)
        self.view2_idx = torch.LongTensor(view2["idx"]).to(poses.device)
        self.poses = poses
        self.focals = torch.cat([fx, fy], -1)
        self.pp = torch.cat([cx, cy], -1)
        self.fg_masks = fg_masks

        pixels_im = torch.stack(
            torch.meshgrid(torch.arange(view1["img"].shape[-1], device=poses.device, dtype=torch.float32),
                           torch.arange(view1["img"].shape[-2], device=poses.device, dtype=torch.float32),
                           indexing="ij")).permute(2, 1, 0)
        self.pixels = pixels_im.reshape(-1, 2)

        # We could use the highest confidence values, but using the first for simplicity since we want to run
        # init later
        self.register_parameter("pts3d_cam", nn.Parameter(
            self.pts3d_i.view(num_preds // neighbor_pose_count, neighbor_pose_count, -1, 3)[:, 0].clone()))

        # self.register_parameter("depth_log", nn.Parameter(
        #     self.pts3d_i.view(num_preds // neighbor_pose_count, neighbor_pose_count, -1, 3)[:, 0, :, 2].log().clone()))
        self.register_parameter("pred_scales_log", nn.Parameter(torch.zeros(num_preds, device=poses.device)))

        self.verbose = verbose

    def forward(self):
        pts3d_ours = self.pts3d_world()
        # pts3d_depth = self.pts3d_world(from_depth=True)
        pred_scales = self.pred_scales_log.exp().view(-1, 1, 1)
        pts3d_pred_i = (self.poses[self.view1_idx] @ torch.cat(
            [pred_scales * self.pts3d_i, torch.ones_like(self.pts3d_i[..., :1])], -1).transpose(1, 2)).transpose(1, 2)[
                       ..., :3]
        pts3d_pred_j = (self.poses[self.view1_idx] @ torch.cat(
            [pred_scales * self.pts3d_j, torch.ones_like(self.pts3d_j[..., :1])], -1).transpose(1, 2)).transpose(1, 2)[
                       ..., :3]

        li = ((pts3d_ours[self.view1_idx] - pts3d_pred_i).norm(dim=-1) * self.weight_i).mean()
        lj = ((pts3d_ours[self.view2_idx] - pts3d_pred_j).norm(dim=-1) * self.weight_j).mean()

        # li += ((pts3d_depth[self.view1_idx] - pts3d_pred_i).norm(dim=-1) * self.weight_i).mean()
        # lj += ((pts3d_depth[self.view2_idx] - pts3d_pred_j).norm(dim=-1) * self.weight_j).mean()

        return li + lj  # + (pts3d_depth - pts3d_ours).norm(dim=-1).mean()

    # @torch.cuda.amp.autocast(enabled=False)
    # def forward_render(self):
    #     scene_xyz = self.pts3d_world().reshape(-1, 3)
    #     scene_rgbs = self.rgbs.reshape(-1, 3)
    #     scene_scales = self.gaussian_scales_log.exp().expand(-1, 3)
    #     scene_opacity = torch.ones_like(scene_xyz[..., :1])
    #
    #     loss = 0
    #
    #     for i in range(len(self.poses)):
    #         xys, depths, radii, conics, comp, num_tiles_hit, cov3d = project_gaussians(
    #             scene_xyz,
    #             scene_scales,
    #             1,
    #             torch.cat([torch.ones_like(scene_opacity), torch.zeros_like(scene_xyz)], -1),
    #             self.w2cs[i, :3],
    #             self.focals[i, 0].item(),
    #             self.focals[i, 1].item(),
    #             self.pp[i, 0].item(),
    #             self.pp[i, 1].item(),
    #             self.rgbs[i].shape[0],
    #             self.rgbs[i].shape[1],
    #             16,
    #         )
    #
    #         splat_results = rasterize_gaussians(
    #             xys,
    #             depths,
    #             radii,
    #             conics,
    #             num_tiles_hit,
    #             scene_rgbs,
    #             scene_opacity,
    #             self.rgbs[i].shape[0],
    #             self.rgbs[i].shape[1],
    #             16,
    #             background=torch.zeros(scene_rgbs.shape[-1], device=xys.device),
    #             return_alpha=False,
    #         )
    #
    #         loss += nn.MSELoss()(splat_results, self.rgbs[i])
    #
    #     return loss  # + 0.01 * self.forward()

    def init_least_squares(self):
        if self.fg_masks is not None:
            constraint_i = self.fg_masks[self.view1_idx]
            constraint_j = self.fg_masks[self.view2_idx]
        elif self.filter_dist_quantiles is not None:
            metric_i = self.pts3d_i.norm(dim=-1)
            metric_j = self.pts3d_j.norm(dim=-1)

            quantiles_i = torch.quantile(metric_i, self.filter_quantiles, dim=-1)
            constraint_i = torch.logical_and(metric_i > quantiles_i[0].unsqueeze(-1),
                                             metric_i < quantiles_i[1].unsqueeze(-1))

            quantiles_j = torch.quantile(metric_j, self.filter_quantiles, dim=-1)
            constraint_j = torch.logical_and(metric_j > quantiles_j[0].unsqueeze(-1),
                                             metric_j < quantiles_j[1].unsqueeze(-1))
        elif self.filter_weight_quantile is not None:
            metric_i = self.weight_i
            metric_j = self.weight_j

            quantiles_i = torch.quantile(metric_i, self.filter_quantiles, dim=-1)
            constraint_i = metric_i > quantiles_i[0].unsqueeze(-1)

            quantiles_j = torch.quantile(metric_j, self.filter_quantiles, dim=-1)
            constraint_j = metric_j > quantiles_j[0].unsqueeze(-1)
        else:
            constraint_i = None
            constraint_j = None

        points_world_i = self.poses[self.view1_idx] \
                         @ torch.cat([self.pts3d_i, torch.ones_like(self.pts3d_i[..., :1])], -1).transpose(1, 2)
        displacement_i = points_world_i.transpose(1, 2)[..., :3] - self.poses[self.view1_idx, :3, 3].unsqueeze(1)

        points_world_j = self.poses[self.view1_idx] \
                         @ torch.cat([self.pts3d_j, torch.ones_like(self.pts3d_j[..., :1])], -1).transpose(1, 2)
        displacement_j = points_world_j.transpose(1, 2)[..., :3] - self.poses[self.view1_idx, :3, 3].unsqueeze(1)

        num_pairs = len(self.pts3d_i)
        A = []
        B = []
        weights = []
        # w1 = self.weight_i.clone()
        # w1[self.fi] = 0
        #
        # w2 = self.weight_j.clone()
        # w2[self.fj] = 0

        pairs_per_scene = self.poses_per_scene * (self.poses_per_scene - 1)
        num_scenes = num_pairs // pairs_per_scene
        for scene_index in range(num_scenes):
            scene_A = []
            scene_B = []
            # scene_weights = []

            scene_start = scene_index * pairs_per_scene
            scene_end = (scene_index + 1) * pairs_per_scene

            for first in range(scene_start, scene_end):
                for second in range(first + 1, scene_end):
                    for view_idx_1, displacement_1, weights_1, c1, view_idx_2, displacement_2, weights_2, c2 in [
                        (self.view1_idx[first], displacement_i[first], self.weight_i[first],
                         constraint_i[first] if constraint_i is not None else None,
                         self.view2_idx[second], displacement_j[second], self.weight_j[second],
                         constraint_j[second] if constraint_j is not None else None),
                        (self.view2_idx[first], displacement_j[first], self.weight_j[first],
                         constraint_j[first] if constraint_j is not None else None,
                         self.view1_idx[second], displacement_i[second], self.weight_i[second],
                         constraint_i[second] if constraint_i is not None else None)]:
                        if view_idx_1 == view_idx_2:
                            assert self.view1_idx[first] != self.view1_idx[second]
                            weights_chunk = torch.minimum(weights_1, weights_2)
                            if c1 is not None:
                                # constraint = torch.logical_and(c1, c2) > 0
                                weights_chunk[torch.logical_or(torch.logical_not(c1), torch.logical_not(c2))] = 0
                                # weights_1 = weights_1[constraint]

                                # weights_2 = weights_2[constraint]
                                # displacement_1 = displacement_1[constraint]
                                # displacement_2 = displacement_2[constraint]

                            weights_chunk = weights_chunk.repeat_interleave(3).unsqueeze(-1)
                            A_chunk = torch.zeros(weights_chunk.shape[0], pairs_per_scene, device=weights_chunk.device)
                            A_chunk[:, first % pairs_per_scene] = displacement_1.reshape(-1)
                            A_chunk[:, second % pairs_per_scene] = -displacement_2.reshape(-1)
                            scene_A.append(weights_chunk * A_chunk)
                            B_chunk = self.poses[self.view1_idx[second], :3, 3:] - self.poses[self.view1_idx[first], :3,
                                                                                   3:]
                            scene_B.append(weights_chunk * B_chunk.repeat(weights_1.shape[0], 1))
                            # scene_weights.append(weights_chunk)

            A.append(torch.cat(scene_A))
            B.append(torch.cat(scene_B))
            # scene_weights = torch.cat(scene_weights)

        A = torch.stack(A)
        B = torch.stack(B)
        # weights = torch.stack(weights)
        # est = sklearn.linear_model.LinearRegression() #positive=True)
        # regressor = sklearn.linear_model.RANSACRegressor(estimator=est, random_state=42)
        # regressor.fit(A.cpu().numpy(), B.squeeze().cpu().numpy(), weights.squeeze().cpu().numpy())
        # scales = torch.FloatTensor(regressor.estimator_.coef_).to(A.device)

        solution = torch.linalg.lstsq(A, B)

        scene_scales = solution.solution.squeeze(-1)
        valid_scenes = scene_scales.min(dim=-1)[0] > 0

        if self.verbose:
            # CONSOLE.log(f"Least squares solution RANSAC: {scales}")
            CONSOLE.log(f"Least squares solution OLS: {solution}")

        scales = solution.solution.reshape(-1)
        valid_pairs = valid_scenes.repeat_interleave(pairs_per_scene)
        # if scales.min() < 0:
        #     # import pdb; pdb.set_trace()
        #     CONSOLE.log("abort least squares")
        #     return
        # solution = scipy.optimize.nnls(A.cpu().numpy(), B.squeeze().cpu().numpy())
        # CONSOLE.log(f"Least squares solution (scipy): {solution}")
        # solution = torch.FloatTensor(solution[0]).to(A.device)
        valid_scales = scales[valid_pairs]
        self.pred_scales_log.data[valid_pairs] = valid_scales.log()
        valid_view1_idx = self.view1_idx[valid_pairs]

        self.pts3d_cam.data[valid_view1_idx] = 0
        weights_denom = torch.zeros_like(self.pts3d_cam.data[..., :1])

        pts3d_pred_i = valid_scales.view(-1, 1, 1) * self.pts3d_i[valid_pairs]
        valid_weight_i = self.weight_i[valid_pairs].unsqueeze(-1)
        self.pts3d_cam.data.index_add_(0, valid_view1_idx, pts3d_pred_i * valid_weight_i)
        weights_denom.index_add_(0, valid_view1_idx, valid_weight_i)

        valid_view2_idx = self.view2_idx[valid_pairs]
        valid_poses_j_c2w = self.poses[valid_view2_idx]
        valid_poses_j_w2c = torch.eye(4, device=valid_poses_j_c2w.device).view(1, 4, 4).repeat(
            valid_poses_j_c2w.shape[0], 1, 1)
        valid_R_j_inv = valid_poses_j_c2w[:, :3, :3].transpose(1, 2)
        valid_poses_j_w2c[:, :3, :3] = valid_R_j_inv
        valid_T_j = valid_poses_j_c2w[:, :3, 3:]
        valid_poses_j_w2c[:, :3, 3:] = -torch.bmm(valid_R_j_inv, valid_T_j)

        valid_pts3d_j = self.pts3d_j[valid_pairs]
        pts3d_pred_j = (valid_poses_j_w2c @ self.poses[self.view1_idx[valid_pairs]]
                        @ torch.cat([valid_scales.view(-1, 1, 1) * valid_pts3d_j,
                                     torch.ones_like(valid_pts3d_j[..., :1])], -1).transpose(1, 2)) \
                           .transpose(1, 2)[..., :3]
        valid_weight_j = self.weight_j[valid_pairs].unsqueeze(-1)
        self.pts3d_cam.data.index_add_(0, valid_view2_idx, pts3d_pred_j * valid_weight_j)
        weights_denom.index_add_(0, valid_view2_idx, valid_weight_j)

        denom_mask = (weights_denom > 0).squeeze()
        self.pts3d_cam.data[denom_mask] /= weights_denom[denom_mask]

        return solution.residuals[valid_scenes].mean(), valid_scenes

        # pts3d_pred_i = scales.view(-1, 1, 1) * self.pts3d_i
        # pts3d_pred_j = (self.poses[self.view2_idx].inverse() @ self.poses[self.view1_idx] @ torch.cat(
        #      [scales.view(-1, 1, 1) * self.pts3d_j, torch.ones_like(self.pts3d_j[..., :1])], -1).transpose(1, 2)) \
        #                     .transpose(1, 2)[..., :3]
        #
        # for view_idx, pts3d, weight in list(zip(self.view1_idx, pts3d_pred_i, self.weight_i.unsqueeze(-1))):
        #     self.pts3d_cam.data[view_idx] += pts3d * weight
        #     weights_chunk[view_idx] += weight
        #
        # for view_idx, pts3d, weight in zip(self.view2_idx, pts3d_pred_j, self.weight_j.unsqueeze(-1)):
        #     self.pts3d_cam.data[view_idx] += pts3d * weight
        #     weights_chunk[view_idx] += weight
        #
        # self.pts3d_cam.data /= weights_chunk.clamp_min(1e-8)
        # self.depth_log.data[:] = self.pts3d_cam.data[..., 2].log().clone()

        # self.gaussian_scales_log.data[:] = (
        #         self.pts3d_cam.data[..., 2].reshape(-1, 1) * self.pixel_areas.reshape(-1, 1)).log()
        # self.pts3d_cam.data[:] = (
        #         self.pts3d_i.view(self.depth_log.shape[0], -1, self.depth_log.shape[1], 3)[:, 0] * scales.view(
        #     self.depth_log.shape[0], -1)[:, 0].view(-1, 1, 1))
        # self.depth_log.data[:] = (self.pts3d_i.view(self.depth_log.shape[0], -1, self.depth_log.shape[1], 3)[:, 0, :,
        #                           2] * scales.view(self.depth_log.shape[0], -1)[:, 0].view(-1, 1)).log()
        # self.pred_scales_log.requires_grad_(False)

    def init_scale_from_poses(self, pnp_method="pytorch3d", min_conf_thr=3, niter_PnP=10):
        focals = self.focals[self.view1_idx]
        pp = self.pp[self.view1_idx]

        zeros = torch.zeros_like(focals[..., :1])
        ones = torch.ones_like(focals[..., :1])

        K = torch.cat([
            torch.cat([focals[..., :1], zeros, pp[..., :1]], -1).unsqueeze(1),
            torch.cat([zeros, focals[..., 1:], pp[..., 1:]], -1).unsqueeze(1),
            torch.cat([zeros, zeros, ones], -1).unsqueeze(1)
        ], 1)

        # pixels_im = torch.stack(
        #     torch.meshgrid(torch.arange(view1["img"].shape[-1], device=K.device, dtype=torch.float32),
        #                    torch.arange(view1["img"].shape[-2], device=K.device, dtype=torch.float32),
        #                    indexing="ij")).permute(2, 1, 0)
        # pixels = pixels_im.reshape(-1, 2)

        if pnp_method == "pytorch3d":
            pixels_homo = torch.cat([self.pixels, torch.ones_like(self.pixels[:, :1])], -1)
            K_inv = K.inverse().permute(0, 2, 1)

            with torch.cuda.amp.autocast(enabled=False):
                pixels_uncal = torch.matmul(pixels_homo, K_inv)
                pnp = efficient_pnp(self.pts3d_j, pixels_uncal[..., :2] / pixels_uncal[..., 2:], self.weight_j)

            rot = pnp.R.transpose(1, 2)  # (B, 3, 3)
            trans = -torch.bmm(rot, pnp.T.unsqueeze(-1))  # (B, 3, 1)
            pnp_c2ws = torch.cat((rot, trans), dim=-1)
        else:
            pixels_npy = self.pixels.cpu().numpy()
            masks: torch.Tensor = self.weight_j > math.log(min_conf_thr)  # mask_thr.view(-1, 1, 1)

            pnp_c2ws = []
            for pred_j, mask, K_i in zip(self.pts3d_j.cpu().numpy(), masks.cpu().numpy(), K.cpu().numpy()):
                if pnp_method == "cv2_ransac":
                    _, R, T, _ = cv2.solvePnPRansac(pred_j[mask], pixels_npy[mask], K_i, None,
                                                    iterationsCount=niter_PnP, reprojectionError=5,
                                                    flags=cv2.SOLVEPNP_SQPNP)
                elif pnp_method == "cv2":
                    _, R, T = cv2.solvePnP(pred_j[mask], pixels_npy[mask], K_i, None,
                                           flags=cv2.SOLVEPNP_SQPNP)
                else:
                    raise Exception(f"{pnp_method} not recognized")

                rot_ij = cv2.Rodrigues(R)[0].T
                trans_ij = -(rot_ij @ T)
                pnp_c2ws.append(torch.cat([torch.FloatTensor(rot_ij), torch.FloatTensor(trans_ij)], -1))

            pnp_c2ws = torch.stack(pnp_c2ws).to(self.poses.device)

        P1 = torch.eye(4, device=self.poses.device)[:3]

        edge_poses = self.poses[torch.cat([self.view1_idx.unsqueeze(-1), self.view2_idx.unsqueeze(-1)], -1)]
        poses_eps = (edge_poses[:, 1, :3, 3] - edge_poses[:, 0, :3, 3]).norm(dim=-1, keepdim=True) / 100
        poses_align_points = torch.cat(
            [edge_poses[:, :, :3, 3], edge_poses[:, :, :3, 3] + poses_eps.unsqueeze(-1) * edge_poses[:, :, :3, 2]], 1)

        pnp_c2ws_eps = pnp_c2ws[:, :3, 3].norm(dim=-1, keepdim=True) / 100
        pnp_c2ws_and_P1 = torch.cat([pnp_c2ws.unsqueeze(1), P1.view(1, 1, 3, 4).expand(len(pnp_c2ws), -1, -1, -1)], 1)
        pnp_align_points = torch.cat([pnp_c2ws_and_P1[:, :, :3, 3],
                                      pnp_c2ws_and_P1[:, :, :3, 3] + pnp_c2ws_eps.unsqueeze(-1) * pnp_c2ws_and_P1[:, :,
                                                                                                  :3, 2]], 1)

        with torch.cuda.amp.autocast(enabled=False):
            alignment = corresponding_points_alignment(pnp_align_points, poses_align_points, estimate_scale=True)

        self.pred_scales_log.data[:] = alignment.s.log()
        self.pts3d_cam.data[:] = (
                self.pts3d_i.view(self.pts3d_cam.shape[0], -1, self.pts3d_cam.shape[1], 3)[:, 0] * alignment.s.view(
            self.pts3d_cam.shape[0], -1)[:, 0].view(-1, 1, 1))
        # self.depth_log.data[:] = (self.pts3d_i.view(self.depth_log.shape[0], -1, self.depth_log.shape[1], 3)[:, 0, :,
        #                           2] * alignment.s.view(self.depth_log.shape[0], -1)[:, 0].view(-1, 1)).log()

    def pts3d_world(self, from_depth=True):
        # return self.pts3d_cam
        # pts3d = self.pts3d_cam
        # pts3d = self.pred_scales_log.exp().view(-1, 1, 1)[::2] * self.pts3d_i[::2]


        pts3d = fast_depthmap_to_pts3d(self.pts3d_cam[..., 2], self.pixels, self.focals,
                                        self.pp) if from_depth else self.pts3d_cam
        transformed = self.poses @ torch.cat([pts3d, torch.ones_like(self.pts3d_cam[..., :1])], -1).transpose(
            1, 2)
        return transformed.transpose(1, 2)[..., :3]

    def depth(self):
        return self.pts3d_cam[..., 2]
        # return self.depth_log.exp()


def fast_depthmap_to_pts3d(depth, pixel_grid, focal, pp):
    pp = pp.unsqueeze(1)
    focal = focal.unsqueeze(1)
    assert focal.shape == (len(depth), 1, 2)
    assert pp.shape == (len(depth), 1, 2)
    assert pixel_grid.shape == (depth.shape[1], 2)
    depth = depth.unsqueeze(-1)

    return torch.cat((depth * (pixel_grid - pp) / focal, depth), dim=-1)
