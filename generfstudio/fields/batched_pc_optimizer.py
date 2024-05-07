from typing import Dict

import cv2
import torch
from pytorch3d.ops import efficient_pnp, corresponding_points_alignment
from torch import nn


class GlobalPointCloudOptimizer(nn.Module):

    def __init__(self, view1: Dict, view2: Dict, pred1: Dict, pred2: Dict, poses: torch.Tensor, fx: torch.Tensor,
                 fy: torch.Tensor, cx: torch.Tensor, cy: torch.Tensor, neighbor_pose_count: int, min_conf_thr=3,
                 pnp_method="pytorch3d", use_elem_wise_pt_conf: bool = False, will_optimize: bool = False,
                 niter_PnP=10):
        super().__init__()

        assert pred1["conf"].shape[0] % neighbor_pose_count == 0

        view1_idx = torch.LongTensor(view1["idx"]).to(poses.device)
        view2_idx = torch.LongTensor(view2["idx"]).to(poses.device)
        pred2_conf = pred2["conf"]
        pts3d_in_other_view = pred2["pts3d_in_other_view"]
        pred_confs = pred1["conf"].view(pred1["conf"].shape[0] // neighbor_pose_count, neighbor_pose_count, -1)
        pred_confs_mean = pred_confs.mean(dim=-1)
        conf_argmax_offset = torch.arange(0, pred_confs_mean.shape[0] * pred_confs_mean.shape[1],
                                          pred_confs_mean.shape[1],
                                          device=pred_confs_mean.device)
        highest_conf1 = conf_argmax_offset + pred_confs_mean.argmax(-1)

        if not (use_elem_wise_pt_conf or will_optimize):
            view1_idx = view1_idx[highest_conf1]
            view2_idx = view2_idx[highest_conf1]
            pred2_conf = pred2_conf[highest_conf1]
            pts3d_in_other_view = pts3d_in_other_view[highest_conf1]

        # best_depthmaps = {}
        fx1 = fx[view1_idx]
        fy1 = fy[view1_idx]
        cx1 = cx[view1_idx]
        cy1 = cy[view1_idx]

        zeros = torch.zeros_like(fx1)
        ones = torch.ones_like(fx1)

        K = torch.cat([
            torch.cat([fx1, zeros, cx1], -1).unsqueeze(1),
            torch.cat([zeros, fy1, cy1], -1).unsqueeze(1),
            torch.cat([zeros, zeros, ones], -1).unsqueeze(1)
        ], 1)

        pixels_im = torch.stack(
            torch.meshgrid(torch.arange(view1["img"].shape[-1], device=K.device, dtype=torch.float32),
                           torch.arange(view1["img"].shape[-2], device=K.device, dtype=torch.float32),
                           indexing="ij")).permute(2, 1, 0)
        pixels = pixels_im.reshape(-1, 2)

        masks: torch.Tensor = pred2_conf > min_conf_thr  # mask_thr.view(-1, 1, 1)

        if pnp_method == "pytorch3d":
            batch_size = pts3d_in_other_view.shape[0]
            pixels_homo = torch.cat([pixels, torch.ones_like(pixels[:, :1])], -1)
            K_inv = K.inverse().permute(0, 2, 1)

            with torch.cuda.amp.autocast(enabled=False):
                pixels_uncal = torch.matmul(pixels_homo, K_inv)
                pnp = efficient_pnp(pts3d_in_other_view.view(batch_size, -1, 3),
                                    pixels_uncal[..., :2] / pixels_uncal[..., 2:], masks.view(batch_size, -1).float())

            rot = pnp.R.transpose(1, 2)  # (B, 3, 3)
            trans = -torch.bmm(rot, pnp.T.unsqueeze(-1))  # (B, 3, 1)
            pnp_c2ws = torch.cat((rot, trans), dim=-1)
        else:
            pixels_npy = pixels_im.cpu().numpy()

            pnp_c2ws = []
            for pred_j, mask, K_i in zip(pts3d_in_other_view.cpu().numpy(), masks.cpu().numpy(), K.cpu().numpy()):
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

            pnp_c2ws = torch.stack(pnp_c2ws).to(poses.device)

        P1 = torch.eye(4, device=poses.device)[:3]

        edge_poses = poses[torch.cat([view1_idx.unsqueeze(-1), view2_idx.unsqueeze(-1)], -1)]
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

        if use_elem_wise_pt_conf:
            preds = pred1["pts3d"].view(pred1["pts3d"].shape[0] // neighbor_pose_count, neighbor_pose_count, -1, 3)
            pred_confs_argmax = pred_confs.argmax(dim=1)
            pts3d = preds.transpose(1, 2).reshape(-1, neighbor_pose_count, 3)[
                torch.arange(preds.shape[0] * preds.shape[2], device=preds.device), pred_confs_argmax.view(
                    -1)].view(preds.shape[0], preds.shape[2], 3)

            self.scaled_pts3d = pts3d * alignment.s[(pred_confs_argmax + conf_argmax_offset.unsqueeze(-1))].unsqueeze(
                -1)
        else:
            scale = alignment.s
            if will_optimize:
                scale = scale[highest_conf1]
            self.scaled_pts3d = pred1["pts3d"].view(pred1["pts3d"].shape[0], -1, 3)[highest_conf1] * scale.view(-1, 1,
                                                                                                                1)
        self.scale = alignment.s

        if use_elem_wise_pt_conf or will_optimize:
            view1_idx = view1_idx[highest_conf1]

        self.poses = poses[view1_idx]

        self.focals = torch.cat([fx[view1_idx], fy[view1_idx]], -1)
        self.pp = torch.cat([cx[view1_idx], cy[view1_idx]], -1)

        self.register_parameter("depth_log", torch.nn.Parameter((self.scaled_pts3d[..., 2]).log().nan_to_num(neginf=0)))
        self.pixels = pixels
        self.will_optimize = will_optimize

    # def get_pts3d(self, raw=False):
    #     res = self.depth_to_pts3d()
    #     if not raw:
    #         res = [dm[:h * w].view(h, w, 3) for dm, (h, w) in zip(res, self.imshapes)]
    #     return res

    def pts3d_world(self):
        pts3d = _fast_depthmap_to_pts3d(self.depth_log.exp(), self.pixels, self.focals, self.pp) if self.will_optimize \
            else self.scaled_pts3d
        transformed = self.poses @ torch.cat([pts3d, torch.ones_like(pts3d[..., :1])], -1).transpose(1, 2)
        return transformed.transpose(1, 2)[..., :3]

    def depth(self):
        if self.will_optimize:
            return self.depth_log.exp()
        else:
            return self.scaled_pts3d.norm(dim=-1)


def _fast_depthmap_to_pts3d(depth, pixel_grid, focal, pp):
    pp = pp.unsqueeze(1)
    focal = focal.unsqueeze(1)
    assert focal.shape == (len(depth), 1, 2)
    assert pp.shape == (len(depth), 1, 2)
    assert pixel_grid.shape == (depth.shape[1], 2)
    depth = depth.unsqueeze(-1)

    return torch.cat((depth * (pixel_grid - pp) / focal, depth), dim=-1)
