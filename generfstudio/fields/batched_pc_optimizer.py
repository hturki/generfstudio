from typing import Dict, Tuple

import cv2
import math
import scipy
import sklearn
import torch
from gsplat import project_gaussians, rasterize_gaussians
from nerfstudio.utils.rich_utils import CONSOLE
from pytorch3d.ops import efficient_pnp, corresponding_points_alignment
from torch import nn


class GlobalPointCloudOptimizer(nn.Module):

    def __init__(self, view1: Dict, view2: Dict, pred1: Dict, pred2: Dict, poses: torch.Tensor, fx: torch.Tensor,
                 fy: torch.Tensor, cx: torch.Tensor, cy: torch.Tensor, pixel_areas: torch.Tensor, rgbs: torch.Tensor,
                 w2cs: torch.Tensor, projmats: torch.Tensor, neighbor_pose_count: int,
                 filter_quantiles: Tuple[int, int] = (0.2, 0.8), filter_weight: bool = False, verbose: bool = False):
        super().__init__()

        num_preds = pred1["conf"].shape[0]
        assert num_preds % neighbor_pose_count == 0

        pts3d_i = pred1["pts3d"].view(num_preds, -1, 3)
        pts3d_j = pred2["pts3d_in_other_view"].view(num_preds, -1, 3)

        avg_dist = torch.cat([pts3d_i, pts3d_j], 1).norm(dim=-1, keepdim=True).mean(dim=1, keepdim=True)
        self.pts3d_i = pts3d_i / avg_dist
        self.pts3d_j = pts3d_j / avg_dist

        self.filter_quantiles = torch.FloatTensor(filter_quantiles).to(self.pts3d_i.device)
        self.filter_weight = filter_weight

        self.weight_i = pred1["conf"].log().view(num_preds, -1)
        self.weight_j = pred2["conf"].log().view(num_preds, -1)

        self.view1_idx = torch.LongTensor(view1["idx"]).to(poses.device)
        self.view2_idx = torch.LongTensor(view2["idx"]).to(poses.device)
        self.poses = poses
        self.focals = torch.cat([fx, fy], -1)
        self.pp = torch.cat([cx, cy], -1)
        self.pixel_areas = pixel_areas
        self.rgbs = rgbs
        self.w2cs = w2cs
        self.projmats = projmats

        pixels_im = torch.stack(
            torch.meshgrid(torch.arange(view1["img"].shape[-1], device=poses.device, dtype=torch.float32),
                           torch.arange(view1["img"].shape[-2], device=poses.device, dtype=torch.float32),
                           indexing="ij")).permute(2, 1, 0)
        self.pixels = pixels_im.reshape(-1, 2)

        self.register_parameter("our_points", nn.Parameter(
            self.pts3d_i.view(num_preds // neighbor_pose_count, neighbor_pose_count, -1, 3)[:, 0].clone()))

        self.register_parameter("depth_log", nn.Parameter(
            self.pts3d_i.view(num_preds // neighbor_pose_count, neighbor_pose_count, -1, 3)[:, 0, :, 2].log().clone()))
        self.register_parameter("pred_scales_log", nn.Parameter(torch.zeros(num_preds, device=poses.device)))
        self.register_parameter("gaussian_scales_log", nn.Parameter(torch.zeros_like(self.depth_log.view(-1, 1))))

        self.verbose = verbose

    def forward_reproj(self):
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

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self):
        scene_xyz = self.pts3d_world().reshape(-1, 3)
        scene_rgbs = self.rgbs.reshape(-1, 3)
        scene_scales = self.gaussian_scales_log.exp().expand(-1, 3)
        scene_opacity = torch.ones_like(scene_xyz[..., :1])

        loss = 0

        for i in range(len(self.poses)):
            xys, depths, radii, conics, comp, num_tiles_hit, cov3d = project_gaussians(
                scene_xyz,
                scene_scales,
                1,
                torch.cat([torch.ones_like(scene_opacity), torch.zeros_like(scene_xyz)], -1),
                self.w2cs[i, :3],
                self.projmats[i] @ self.w2cs[i],
                self.focals[i, 0].item(),
                self.focals[i, 1].item(),
                self.pp[i, 0].item(),
                self.pp[i, 1].item(),
                self.rgbs[i].shape[0],
                self.rgbs[i].shape[1],
                16,
            )

            splat_results = rasterize_gaussians(
                xys,
                depths,
                radii,
                conics,
                num_tiles_hit,
                scene_rgbs,
                scene_opacity,
                self.rgbs[i].shape[0],
                self.rgbs[i].shape[1],
                16,
                background=torch.zeros(scene_rgbs.shape[-1], device=xys.device),
                return_alpha=False,
            )

            loss += nn.MSELoss()(splat_results, self.rgbs[i])

        return loss  # + 0.01 * self.forward_reproj()

    def init_least_squares(self):
        # if filter_weight:
        #     weight_i_quantiles = torch.quantile(self.weight_i, self.filter_quantiles[0], dim=-1)
        #     self.fi = self.weight_i < weight_i_quantiles[0].unsqueeze(-1)
        #
        #     weight_j_quantiles = torch.quantile(self.weight_j, self.filter_quantiles[0], dim=-1)
        #     self.fj = self.weight_j < weight_j_quantiles[0].unsqueeze(-1)

        dist_i = self.pts3d_i.norm(dim=-1)
        quantiles_i = torch.quantile(dist_i, self.filter_quantiles, dim=-1)
        constraint_i = torch.logical_and(dist_i > quantiles_i[0].unsqueeze(-1), dist_i < quantiles_i[1].unsqueeze(-1))
        points_world_i = self.poses[self.view1_idx] \
                         @ torch.cat([self.pts3d_i, torch.ones_like(self.pts3d_i[..., :1])], -1).transpose(1, 2)
        displacement_i = points_world_i.transpose(1, 2)[..., :3] - self.poses[self.view1_idx, :3, 3].unsqueeze(1)

        dist_j = self.pts3d_j.norm(dim=-1)
        quantiles_j = torch.quantile(dist_j, self.filter_quantiles, dim=-1)
        constraint_j = torch.logical_and(dist_j > quantiles_j[0].unsqueeze(-1), dist_j < quantiles_j[1].unsqueeze(-1))
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

        for first in range(num_pairs):
            for second in range(first + 1, num_pairs):
                for view_idx_1, displacement_1, weights_1, c1, view_idx_2, displacement_2, weights_2, c2 in [
                    (self.view1_idx[first], displacement_i[first], self.weight_i[first], constraint_i[first],
                     self.view2_idx[second], displacement_j[second], self.weight_j[second], constraint_j[second]),
                    (self.view2_idx[first], displacement_j[first], self.weight_j[first], constraint_j[first],
                     self.view1_idx[second], displacement_i[second], self.weight_i[second], constraint_i[second])]:
                    if view_idx_1 == view_idx_2:
                        assert self.view1_idx[first] != self.view1_idx[second]
                        constraint = torch.minimum(weights_1, weights_2) > 0

                        # constraint = torch.logical_and(torch.minimum(weights_1, weights_2) > math.log(3), torch.logical_and(c1, c2))
                        weights_1c = weights_1[constraint]
                        weights_chunk = torch.minimum(weights_1c, weights_2[constraint]) \
                            .repeat_interleave(3).unsqueeze(-1)
                        A_chunk = torch.zeros(weights_chunk.shape[0], num_pairs, device=weights_chunk.device)
                        A_chunk[:, first] = displacement_1[constraint].reshape(-1)
                        A_chunk[:, second] = -displacement_2[constraint].reshape(-1)
                        A.append(A_chunk)
                        # A.append(weights_chunk * A_chunk)
                        B_chunk = self.poses[self.view1_idx[second], :3, 3:] - self.poses[self.view1_idx[first], :3, 3:]
                        B.append(B_chunk.repeat(weights_1c.shape[0], 1))
                        # B.append(weights_chunk * B_chunk.repeat(weights_1.shape[0], 1))
                        weights.append(weights_chunk)

        A = torch.cat(A)
        B = torch.cat(B)
        weights = torch.cat(weights)
        # est = sklearn.linear_model.LinearRegression() #positive=True)
        # regressor = sklearn.linear_model.RANSACRegressor(estimator=est, random_state=42)
        # regressor.fit(A.cpu().numpy(), B.squeeze().cpu().numpy(), weights.squeeze().cpu().numpy())
        # scales = torch.FloatTensor(regressor.estimator_.coef_).to(A.device)
        solution = torch.linalg.lstsq(weights * A, weights * B)

        if self.verbose:
            # CONSOLE.log(f"Least squares solution RANSAC: {scales}")
            solution = torch.linalg.lstsq(weights * A, weights * B)
            CONSOLE.log(f"Least squares solution OLS: {solution}")
            # huber = sklearn.linear_model.HuberRegressor()
            # huber.fit(A.cpu().numpy(), B.squeeze().cpu().numpy(), weights.squeeze().cpu().numpy())
            # CONSOLE.log(f"Least squares solution Huber: {huber.coef_}")

        scales = solution.solution.squeeze()
        if scales.min() < 0:
            # import pdb; pdb.set_trace()
            CONSOLE.log("abort least squares")
            return
            # solution = scipy.optimize.nnls(A.cpu().numpy(), B.squeeze().cpu().numpy())
            # CONSOLE.log(f"Least squares solution (scipy): {solution}")
            # solution = torch.FloatTensor(solution[0]).to(A.device)
        self.pred_scales_log.data[:] = scales.log()
        self.our_points.data.fill_(0)
        weights_chunk = torch.zeros_like(self.our_points[..., :1])
        pts3d_pred_i = scales.view(-1, 1, 1) * self.pts3d_i
        pts3d_pred_j = (self.poses[self.view2_idx].inverse() @ self.poses[self.view1_idx] @ torch.cat(
            [scales.view(-1, 1, 1) * self.pts3d_j, torch.ones_like(self.pts3d_j[..., :1])], -1).transpose(1, 2)) \
                           .transpose(1, 2)[..., :3]

        for view_idx, pts3d, weight in list(zip(self.view1_idx, pts3d_pred_i, self.weight_i.unsqueeze(-1))):
            self.our_points.data[view_idx] += pts3d * weight
            weights_chunk[view_idx] += weight

        for view_idx, pts3d, weight in zip(self.view2_idx, pts3d_pred_j, self.weight_j.unsqueeze(-1)):
            self.our_points.data[view_idx] += pts3d * weight
            weights_chunk[view_idx] += weight

        self.our_points.data /= weights_chunk
        self.depth_log.data[:] = self.our_points.data[..., 2].log().clone()

        # self.gaussian_scales_log.data[:] = (
        #         self.our_points.data[..., 2].reshape(-1, 1) * self.pixel_areas.reshape(-1, 1)).log()
        # self.our_points.data[:] = (
        #         self.pts3d_i.view(self.depth_log.shape[0], -1, self.depth_log.shape[1], 3)[:, 0] * scales.view(
        #     self.depth_log.shape[0], -1)[:, 0].view(-1, 1, 1))
        # self.depth_log.data[:] = (self.pts3d_i.view(self.depth_log.shape[0], -1, self.depth_log.shape[1], 3)[:, 0, :,
        #                           2] * scales.view(self.depth_log.shape[0], -1)[:, 0].view(-1, 1)).log()
        # self.pred_scales_log.requires_grad_(False)

    def init_scale_from_poses(self, pnp_method="pytorch3d", min_conf_thr=3, niter_PnP=10):

        # view1_idx = torch.LongTensor(view1["idx"]).to(poses.device)
        # view2_idx = torch.LongTensor(view2["idx"]).to(poses.device)
        # pred2_conf = pred2["conf"]
        # pts3d_in_other_view = pred2["pts3d_in_other_view"]
        # pred_confs = pred1["conf"].view(pred1["conf"].shape[0] // neighbor_pose_count, neighbor_pose_count, -1)
        # pred_confs_mean = pred_confs.mean(dim=-1)
        # conf_argmax_offset = torch.arange(0, pred_confs_mean.shape[0] * pred_confs_mean.shape[1],
        #                                   pred_confs_mean.shape[1],
        #                                   device=pred_confs_mean.device)
        # highest_conf1 = conf_argmax_offset + pred_confs_mean.argmax(-1)
        #
        # if not (use_elem_wise_pt_conf or will_optimize):
        #     view1_idx = view1_idx[highest_conf1]
        #     view2_idx = view2_idx[highest_conf1]
        #     pred2_conf = pred2_conf[highest_conf1]
        #     pts3d_in_other_view = pts3d_in_other_view[highest_conf1]

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
        self.our_points.data[:] = (
                self.pts3d_i.view(self.depth_log.shape[0], -1, self.depth_log.shape[1], 3)[:, 0] * alignment.s.view(
            self.depth_log.shape[0], -1)[:, 0].view(-1, 1, 1))
        self.depth_log.data[:] = (self.pts3d_i.view(self.depth_log.shape[0], -1, self.depth_log.shape[1], 3)[:, 0, :,
                                  2] * alignment.s.view(self.depth_log.shape[0], -1)[:, 0].view(-1, 1)).log()

    def pts3d_world(self, from_depth=True):
        # return self.our_points
        # pts3d = self.our_points
        # pts3d = self.pred_scales_log.exp().view(-1, 1, 1)[::2] * self.pts3d_i[::2]
        pts3d = _fast_depthmap_to_pts3d(self.depth_log.exp(), self.pixels, self.focals,
                                        self.pp) if from_depth else self.our_points
        transformed = self.poses @ torch.cat([pts3d, torch.ones_like(pts3d[..., :1])], -1).transpose(1, 2)
        return transformed.transpose(1, 2)[..., :3]

    def depth(self):
        # return self.our_points[..., 2]
        # return self.pts3d_i[::2][..., 2]
        # return self.pts3d_i.view(3, 2, -1, 3)[:, 0][..., 2]
        return self.depth_log.exp()

    def get_masks(self):
        return self.im_conf > 3


def _fast_depthmap_to_pts3d(depth, pixel_grid, focal, pp):
    pp = pp.unsqueeze(1)
    focal = focal.unsqueeze(1)
    assert focal.shape == (len(depth), 1, 2)
    assert pp.shape == (len(depth), 1, 2)
    assert pixel_grid.shape == (depth.shape[1], 2)
    depth = depth.unsqueeze(-1)

    return torch.cat((depth * (pixel_grid - pp) / focal, depth), dim=-1)
