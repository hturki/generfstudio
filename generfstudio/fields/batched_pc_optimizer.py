from typing import Dict

import cv2
import math
import torch
from pytorch3d.ops import efficient_pnp, corresponding_points_alignment
from torch import nn


class GlobalPointCloudOptimizer(nn.Module):

    def __init__(self, view1: Dict, view2: Dict, pred1: Dict, pred2: Dict, poses: torch.Tensor, fx: torch.Tensor,
                 fy: torch.Tensor, cx: torch.Tensor, cy: torch.Tensor, neighbor_pose_count: int, verbose: bool = False):
        super().__init__()

        num_preds = pred1["conf"].shape[0]
        assert num_preds % neighbor_pose_count == 0

        self.pts3d_i = pred1["pts3d"].view(num_preds, -1, 3)
        self.pts3d_j = pred2["pts3d_in_other_view"].view(num_preds, -1, 3)
        self.weight_i = pred1["conf"].log().view(num_preds, -1)
        self.weight_j = pred2["conf"].log().view(num_preds, -1)
        self.view1_idx = torch.LongTensor(view1["idx"]).to(poses.device)
        self.view2_idx = torch.LongTensor(view2["idx"]).to(poses.device)
        self.poses = poses
        self.focals = torch.cat([fx, fy], -1)
        self.pp = torch.cat([cx, cy], -1)

        pixels_im = torch.stack(
            torch.meshgrid(torch.arange(view1["img"].shape[-1], device=poses.device, dtype=torch.float32),
                           torch.arange(view1["img"].shape[-2], device=poses.device, dtype=torch.float32),
                           indexing="ij")).permute(2, 1, 0)
        self.pixels = pixels_im.reshape(-1, 2)

        self.register_parameter("depth_log", nn.Parameter(
            self.pts3d_i.view(num_preds // neighbor_pose_count, neighbor_pose_count, -1, 3)[:, 0, :, 2].log()))
        self.register_parameter("pred_scales_log", nn.Parameter(torch.zeros(num_preds, device=poses.device)))
        self.verbose = verbose

    def forward(self):
        pts3d_ours = self.pts3d_world()
        pred_scales = self.pred_scales_log.exp().view(-1, 1, 1)
        pts3d_pred_i = (self.poses[self.view1_idx] @ torch.cat(
            [pred_scales * self.pts3d_i, torch.ones_like(self.pts3d_i[..., :1])], -1).transpose(1, 2)).transpose(1, 2)[
                       ..., :3]
        pts3d_pred_j = (self.poses[self.view1_idx] @ torch.cat(
            [pred_scales * self.pts3d_j, torch.ones_like(self.pts3d_j[..., :1])], -1).transpose(1, 2)).transpose(1, 2)[
                       ..., :3]

        li = ((pts3d_ours[self.view1_idx] - pts3d_pred_i).norm(dim=-1) * self.weight_i).mean()
        lj = ((pts3d_ours[self.view2_idx] - pts3d_pred_j).norm(dim=-1) * self.weight_j).mean()

        return li + lj

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
        self.depth_log.data[:] = (self.pts3d_i.view(self.depth_log.shape[0], -1, self.depth_log.shape[1], 3)[:, 0, :,
                                  2] * alignment.s.view(self.depth_log.shape[0], -1)[:, 0].view(-1, 1)).log()

    def pts3d_world(self):
        pts3d = _fast_depthmap_to_pts3d(self.depth_log.exp(), self.pixels, self.focals, self.pp)
        transformed = self.poses @ torch.cat([pts3d, torch.ones_like(pts3d[..., :1])], -1).transpose(1, 2)
        return transformed.transpose(1, 2)[..., :3]

    def depth(self):
        return self.depth_log.exp()


def _fast_depthmap_to_pts3d(depth, pixel_grid, focal, pp):
    pp = pp.unsqueeze(1)
    focal = focal.unsqueeze(1)
    assert focal.shape == (len(depth), 1, 2)
    assert pp.shape == (len(depth), 1, 2)
    assert pixel_grid.shape == (depth.shape[1], 2)
    depth = depth.unsqueeze(-1)

    return torch.cat((depth * (pixel_grid - pp) / focal, depth), dim=-1)
