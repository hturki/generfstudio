from typing import Dict, Optional

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
                 fx: torch.Tensor, fy: torch.Tensor, cx: torch.Tensor, cy: torch.Tensor,
                 poses_per_scene: int, fg_masks: Optional[torch.Tensor], verbose: bool = False):
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

        self.verbose = verbose

    def init_least_squares(self):
        if self.fg_masks is not None:
            constraint_i = self.fg_masks[self.view1_idx]
            constraint_j = self.fg_masks[self.view2_idx]
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

        pairs_per_scene = self.poses_per_scene * (self.poses_per_scene - 1)
        num_scenes = num_pairs // pairs_per_scene
        for scene_index in range(num_scenes):
            scene_A = []
            scene_B = []

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
                                weights_chunk[torch.logical_or(torch.logical_not(c1), torch.logical_not(c2))] = 0

                            weights_chunk = weights_chunk.repeat_interleave(3).unsqueeze(-1)
                            A_chunk = torch.zeros(weights_chunk.shape[0], pairs_per_scene, device=weights_chunk.device)
                            A_chunk[:, first % pairs_per_scene] = displacement_1.reshape(-1)
                            A_chunk[:, second % pairs_per_scene] = -displacement_2.reshape(-1)
                            scene_A.append(weights_chunk * A_chunk)
                            B_chunk = self.poses[self.view1_idx[second], :3, 3:] - self.poses[self.view1_idx[first], :3,
                                                                                   3:]
                            scene_B.append(weights_chunk * B_chunk.repeat(weights_1.shape[0], 1))

            A.append(torch.cat(scene_A))
            B.append(torch.cat(scene_B))

        A = torch.stack(A)
        B = torch.stack(B)
        solution = torch.linalg.lstsq(A, B)

        scene_scales = solution.solution.squeeze(-1)
        valid_scenes = scene_scales.min(dim=-1)[0] > 0

        if self.verbose:
            CONSOLE.log(f"Least squares solution OLS: {solution}")

        scales = solution.solution.reshape(-1)
        valid_pairs = valid_scenes.repeat_interleave(pairs_per_scene)

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


    
    def pts3d_world(self):
        pts3d = fast_depthmap_to_pts3d(self.pts3d_cam[..., 2], self.pixels, self.focals, self.pp)
        transformed = self.poses @ torch.cat([pts3d, torch.ones_like(self.pts3d_cam[..., :1])], -1).transpose(
            1, 2)
        return transformed.transpose(1, 2)[..., :3]

    def depth(self):
        return self.pts3d_cam[..., 2]


def fast_depthmap_to_pts3d(depth, pixel_grid, focal, pp):
    pp = pp.unsqueeze(1)
    focal = focal.unsqueeze(1)
    assert focal.shape == (len(depth), 1, 2)
    assert pp.shape == (len(depth), 1, 2)
    assert pixel_grid.shape == (depth.shape[1], 2)
    depth = depth.unsqueeze(-1)

    return torch.cat((depth * (pixel_grid - pp) / focal, depth), dim=-1)
