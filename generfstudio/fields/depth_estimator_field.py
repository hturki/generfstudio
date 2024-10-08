from typing import Optional

import torch
from mast3r.model import AsymmetricMASt3R
from nerfstudio.utils import profiler
from torch import nn
from torch_scatter import scatter_max

from generfstudio.fields.batched_pc_optimizer import GlobalPointCloudOptimizer


class DepthEstimatorField(nn.Module):

    def __init__(
            self,
            model_name: str,
    ) -> None:
        super().__init__()

        self.model = AsymmetricMASt3R.from_pretrained(model_name)
        for p in self.model.parameters():
            p.requires_grad_(False)

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

        with torch.cuda.amp.autocast(enabled=False):
            scene = GlobalPointCloudOptimizer(
                view1, view2, pred1, pred2, c2ws.view(-1, 4, 4), fx.view(-1, 1), fy.view(-1, 1), cx.view(-1, 1),
                cy.view(-1, 1), rgbs.shape[1], fg_masks.flatten(0, 1) if fg_masks is not None else None,
                verbose=not self.training)

            alignment_loss, valid_alignment = scene.init_least_squares()
            pts3d = scene.pts3d_world().detach()
            depth = scene.depth().detach()

        indices_base = torch.arange(depth.shape[-1], device=depth.device).unsqueeze(0).expand(pred1["conf"].shape[0],
                                                                                              -1)
        pred_1_indices = (scene.view1_idx.unsqueeze(-1) * depth.shape[-1]) + indices_base
        confs = scatter_max(pred1["conf"].view(-1), pred_1_indices.view(-1))[0]
        pred_2_indices = (scene.view1_idx.unsqueeze(-1) * depth.shape[-1]) + indices_base
        scatter_max(pred2["conf"].view(-1), pred_2_indices.view(-1), out=confs)

        return pts3d, depth, alignment_loss, valid_alignment, confs.view(depth.shape)
