from typing import List, Tuple

import math
import numpy as np
import scipy
import sklearn
import torch
from PIL import Image
from dust3r.cloud_opt.commons import cosine_schedule
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.geometry import geotrf
from gsplat import fully_fused_projection, isect_tiles, isect_offset_encode, rasterize_to_pixels
# from gsplat.experimental.cuda import quat_scale_to_covar_perci, projection, isect_tiles, isect_offset_encode, \
#     rasterize_to_pixels
from pytorch_msssim import SSIM
from torch import nn
from torchmetrics.image import PeakSignalNoiseRatio
from tqdm import tqdm

from generfstudio.fields.batched_pc_optimizer import fast_depthmap_to_pts3d


class GaussianOptimizer(nn.Module):

    def __init__(self, image_paths: List[str], c2ws: torch.Tensor, fx: torch.Tensor,
                 fy: torch.Tensor, cx: torch.Tensor, cy: torch.Tensor):
        # def __init__(self, image_paths: List[str], depth: torch.Tensor, c2ws: torch.Tensor, fx: torch.Tensor,
        #              fy: torch.Tensor, cx: torch.Tensor, cy: torch.Tensor, opt_depth_only: bool, opt_scales: bool,
        #              opt_opacity: bool):
        super().__init__()
        self.register_buffer("rgbs", torch.stack(
            [torch.FloatTensor(np.asarray(Image.open(x).resize((224, 224), Image.LANCZOS))) / 255.0 for x in
             image_paths]), persistent=False)

        self.register_buffer("fx", fx, persistent=False)
        self.register_buffer("fy", fy, persistent=False)
        self.register_buffer("cx", cx, persistent=False)
        self.register_buffer("cy", cy, persistent=False)
        self.register_buffer("c2ws", c2ws, persistent=False)

    @torch.inference_mode()
    def align_with_dust3r(self, model_name: str, num_neighbors: int = 1):
        model = AsymmetricCroCo3DStereo.from_pretrained(model_name).to(self.rgbs.device)
        view1 = {"img": [], "idx": [], "instance": []}
        view2 = {"img": [], "idx": [], "instance": []}
        for i in range(len(self.rgbs) - num_neighbors):
            for j in range(i + 1, i + num_neighbors + 1):
                view1["img"].append(self.rgbs[i].permute(2, 0, 1) * 2 - 1)
                view2["img"].append(self.rgbs[j].permute(2, 0, 1) * 2 - 1)
                view1["idx"].append(i)
                view1["instance"].append(str(i))
                view2["idx"].append(j)
                view2["instance"].append(str(j))

                view1["img"].append(self.rgbs[j].permute(2, 0, 1) * 2 - 1)
                view2["img"].append(self.rgbs[i].permute(2, 0, 1) * 2 - 1)
                view1["idx"].append(j)
                view1["instance"].append(str(j))
                view2["idx"].append(i)
                view2["instance"].append(str(i))

        view1["img"] = torch.stack(view1["img"])
        view2["img"] = torch.stack(view2["img"])

        pred1, pred2 = model(view1, view2)

        num_preds = len(pred1["pts3d"])
        pts3d_i = pred1["pts3d"].view(num_preds, -1, 3)
        pts3d_j = pred2["pts3d_in_other_view"].view(num_preds, -1, 3)

        weight_i = pred1["conf"].log().view(num_preds, -1)
        weight_j = pred2["conf"].log().view(num_preds, -1)

        view1_idx = torch.LongTensor(view1["idx"]).to(self.rgbs.device)
        view2_idx = torch.LongTensor(view2["idx"]).to(self.rgbs.device)

        points_world_i = geotrf(self.c2ws[view1_idx], pts3d_i)
        displacement_i = points_world_i - self.c2ws[view1_idx, :3, 3].unsqueeze(1)

        points_world_j = geotrf(self.c2ws[view1_idx], pts3d_j)
        displacement_j = points_world_j - self.c2ws[view1_idx, :3, 3].unsqueeze(1)
        A = []
        B = []
        weights = []

        # We have symmetric predictions to minimize via least squares
        for i in range(0, num_preds, 2):
            assert view1_idx[i] == view2_idx[i + 1]
            assert view2_idx[i] == view1_idx[i + 1]

            for first, second, weight_1, weight_2, displacement_1, displacement_2 in [
                (i, i + 1, weight_i[i], weight_j[i + 1], displacement_i[i], displacement_j[i + 1]),
                (i + 1, i, weight_i[i + 1], weight_j[i], displacement_i[i + 1], displacement_j[i])
            ]:
                _, indices = torch.sort(torch.minimum(weight_1, weight_2), descending=True)
                constraint = indices[:5000]
                # constraint = torch.logical_and(weight_1 >= math.log(3), weight_2 >= math.log(3)) > 0
                weight_1 = weight_1[constraint]
                weight_2 = weight_2[constraint]
                displacement_1 = displacement_1[constraint]
                displacement_2 = displacement_2[constraint]

                weights_chunk = torch.minimum(weight_1, weight_2).repeat_interleave(3).unsqueeze(-1)
                A_chunk = torch.zeros(weights_chunk.shape[0], num_preds, device=weights_chunk.device)
                A_chunk[:, first] = displacement_1.reshape(-1)
                A_chunk[:, second] = -displacement_2.reshape(-1)

                # A.append(A_chunk.to_sparse())
                A.append((weights_chunk * A_chunk).to_sparse())

                B_chunk = self.c2ws[view1_idx[second], :3, 3:] - self.c2ws[view1_idx[first], :3, 3:]
                B.append(weights_chunk * B_chunk.repeat(weight_1.shape[0], 1))
                # B.append(B_chunk.repeat(weight_1.shape[0], 1))
                # weights.append(weights_chunk)

        A = torch.cat(A)
        B = torch.cat(B)

        est = sklearn.linear_model.LinearRegression(positive=True)
        regressor = sklearn.linear_model.RANSACRegressor(estimator=est, random_state=42)

        import pdb;
        pdb.set_trace()
        lol = scipy.sparse.linalg.lsmr(A.cpu().to_dense().numpy(), B.squeeze().cpu().numpy())

        regressor.fit(A.cpu().to_dense().numpy(), B.squeeze().cpu().numpy(), torch.cat(weights).squeeze().cpu().numpy())

        solution = torch.linalg.lstsq(A.to_dense(), B)

    def old(self):
        self.register_buffer("rgbs", torch.stack(
            [torch.FloatTensor(np.asarray(Image.open(x).resize((224, 224), Image.LANCZOS))).to(depth.device) / 255.0
             for x in image_paths]), persistent=False)

        self.opt_opacity = opt_opacity
        if self.opt_opacity:
            self.register_parameter("opacity_logit", nn.Parameter(torch.full_like(self.rgbs.view(-1, 3)[..., :1], 0.1)))
        else:
            self.register_buffer("_opacity", torch.ones_like(self.rgbs.view(-1, 3)[..., :1]), persistent=False)

        self.register_buffer("quats",
                             torch.cat([torch.ones_like(self.opacity), torch.zeros_like(self.rgbs.view(-1, 3))], -1),
                             persistent=False)

        self.opt_scales = opt_scales
        if opt_scales:
            scales = (depth / fx).view(-1, 1)
            self.register_parameter("scales_log", nn.Parameter(scales.log()))

        pixels_im = torch.stack(
            torch.meshgrid(torch.arange(224, dtype=torch.float32, device=depth.device),
                           torch.arange(224, dtype=torch.float32, device=depth.device),
                           indexing="ij")).permute(2, 1, 0)
        self.register_buffer("pixels", pixels_im.reshape(-1, 2), persistent=False)

        self.opt_depth_only = opt_depth_only
        if opt_depth_only:
            self.register_parameter("depth_log", nn.Parameter(torch.ones_like(depth.log())))
            self.register_buffer("fx", fx, persistent=False)
            self.register_buffer("fy", fy, persistent=False)
            self.register_buffer("cx", cx, persistent=False)
            self.register_buffer("cy", cy, persistent=False)
            self.register_buffer("c2ws", c2ws, persistent=False)
        else:
            assert opt_scales
            pts3d_cam = fast_depthmap_to_pts3d(depth, self.pixels, torch.cat([fx, fy], -1), torch.cat([cx, cy], -1))
            xyz = geotrf(c2ws, pts3d_cam).view(-1, 3)
            self.register_parameter("xyz", nn.Parameter(xyz))

        zeros = torch.zeros_like(fx)
        ones = torch.ones_like(fx)
        K = torch.cat([
            torch.cat([fx, zeros, cx], -1).unsqueeze(1),
            torch.cat([zeros, fy, cy], -1).unsqueeze(1),
            torch.cat([zeros, zeros, ones], -1).unsqueeze(1)
        ], 1)
        self.register_buffer("K", K, persistent=False)

        R = c2ws[:, :3, :3]
        T = c2ws[:, :3, 3:]  # 3 x 1

        w2cs = torch.eye(4, device=R.device).unsqueeze(0).repeat(len(c2ws), 1, 1)
        R_inv = R[:, :3, :3].transpose(1, 2)
        w2cs[:, :3, :3] = R_inv
        w2cs[:, :3, 3:] = -torch.bmm(R_inv, T)
        self.register_buffer("w2cs", w2cs, persistent=False)

        self.tile_size = 16
        self.tile_dim = math.ceil(224 / self.tile_size)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)

    @torch.no_grad()
    def get_psnr(self, chunk_size: int = 10) -> Tuple[List[float], List[float]]:
        xyz = self.get_xyz()
        scales = self.scales

        psnrs = []
        psnrs_i = []

        # for i in range(0, len(self.w2cs), chunk_size):
        #     gt_rgb = self.rgbs[i:i + chunk_size]
        #     splat_results_i = self.render_views_indiv(torch.arange(i, min(len(self.w2cs), i + chunk_size)), xyz, scales)
        #     for result_index in range(len(splat_results_i)):
        #         psnr_i = self.psnr(gt_rgb[result_index], splat_results_i[result_index]).item()
        #         if psnr_i < 20:
        #             self._opacity[224 * 224 * (i + result_index):224 * 224 * (i + result_index + 1)] = 0

        # unfiltered_psnrs = []
        # for i in range(0, len(self.w2cs), chunk_size):
        #     gt_rgb = self.rgbs[i:i + chunk_size]
        #     splat_results, alpha = self.render_views(torch.arange(i, min(len(self.w2cs), i + chunk_size)), xyz, scales)
        #     for result_index in range(len(splat_results)):
        #         unfiltered_psnrs.append(self.psnr(gt_rgb[result_index], splat_results[result_index]).item())
        #
        # sorted, indices = torch.sort(torch.FloatTensor(unfiltered_psnrs))
        # import pdb; pdb.set_trace()
        # for old_psnr, index in zip(sorted, indices):
        #     splat_results, alpha = self.render_views(torch.arange(index, index + 1), xyz, scales)
        #     psnr = self.psnr(self.rgbs[index:index + 1], splat_results).item()
        #     print(old_psnr, psnr)
        #     if psnr < 20:
        #         self._opacity[224 * 224 * index:224 * 224 * (index + 1)] = 0

        for i in range(0, len(self.w2cs), chunk_size):
            splat_results, alpha = self.render_views(torch.arange(i, min(len(self.w2cs), i + chunk_size)), xyz, scales)
            gt_rgb = self.rgbs[i:i + chunk_size]
            splat_results_i = self.render_views_indiv(torch.arange(i, min(len(self.w2cs), i + chunk_size)), xyz, scales)
            for result_index in range(len(splat_results)):
                psnrs.append(self.psnr(gt_rgb[result_index], splat_results[result_index]).item())
                psnrs_i.append(self.psnr(gt_rgb[result_index], splat_results_i[result_index]).item())
                print(psnrs[-1], psnrs_i[-1])
            import pdb;
            pdb.set_trace()

        return psnrs, psnrs_i

    def optimize(self, iters: int, chunk_size: int = 10, verbose: bool = True):
        params = [{"params": self.depth_log if self.opt_depth_only else self.xyz}]
        if self.opt_opacity:
            params.append({"params": self.opacity_logit})

        if self.opt_scales:
            params.append({"params": self.scales_log, "lr": 0.005})

        optimizer = torch.optim.Adam(params, lr=1.6e-4, eps=1e-15)

        with tqdm(total=iters, disable=not verbose) as bar:
            cur_iters = 0
            while cur_iters < iters:
                view_indices = torch.randperm(len(self.w2cs))
                for i in range(0, len(view_indices), chunk_size):
                    t = cur_iters / iters
                    optimizer.param_groups[0]["lr"] = cosine_schedule(t, 1e-2, 1e-4)
                    optimizer.zero_grad()

                    xyz = self.get_xyz()
                    scales = self.scales
                    indices_chunk = view_indices[i:i + chunk_size]
                    splat_results, _ = self.render_views(indices_chunk, xyz, scales)

                    gt_rgb = self.rgbs[indices_chunk]

                    Ll1 = torch.abs(gt_rgb - splat_results).mean()
                    simloss = 1 - self.ssim(gt_rgb.permute(0, 3, 1, 2), splat_results.permute(0, 3, 1, 2))
                    loss = Ll1
                    # loss = 0.8 * Ll1 + 0.2 * simloss
                    loss.backward()
                    optimizer.step()

                    with torch.no_grad():
                        psnr = self.psnr(gt_rgb, splat_results).item()

                    bar.set_postfix_str(f'{psnr=:g} loss={loss:g}')
                    bar.update()

                    cur_iters += 1
                    if cur_iters >= iters:
                        break

    def render_views(self, view_indices: torch.Tensor, xyz: torch.Tensor, scales: torch.Tensor) -> \
            Tuple[
                torch.Tensor, torch.Tensor]:
        w2cs_chunk = self.w2cs[view_indices]
        radii, means2d, depths, conics, compensations = fully_fused_projection(xyz, None, self.quats,
                                                                               scales.expand(-1, 3),
                                                                               w2cs_chunk, self.K[view_indices], 224,
                                                                               224, packed=False)

        tiles_per_gauss, isect_ids, gauss_ids = isect_tiles(
            means2d, radii, depths, self.tile_size, self.tile_dim, self.tile_dim, packed=False
        )

        isect_offsets = isect_offset_encode(isect_ids, len(w2cs_chunk), self.tile_dim, self.tile_dim)

        return rasterize_to_pixels(
            means2d,
            conics,
            self.rgbs.view(1, -1, 3).expand(len(w2cs_chunk), -1, -1),
            self.opacity.squeeze(-1).unsqueeze(0).expand(len(w2cs_chunk), -1),
            224,
            224,
            self.tile_size,
            isect_offsets,
            gauss_ids,
        )

    def render_views_indiv(self, view_indices: torch.Tensor, xyz: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        renderings = []
        num_pixels = 224 * 224
        for view_index in view_indices:
            w2cs_chunk = self.w2cs[view_index:view_index + 1]
            start = max(0, view_index - 3) * num_pixels
            end = (view_index + 3 + 1) * num_pixels
            radii, means2d, depths, conics, compensations = projection(xyz[start:end], None, self.quats[start:end],
                                                                       scales[start:end].expand(-1, 3),
                                                                       w2cs_chunk, self.K[view_index:view_index + 1],
                                                                       224, 224,
                                                                       packed=False)

            tiles_per_gauss, isect_ids, gauss_ids = isect_tiles(
                means2d, radii, depths, self.tile_size, self.tile_dim, self.tile_dim, packed=False
            )

            isect_offsets = isect_offset_encode(isect_ids, len(w2cs_chunk), self.tile_dim, self.tile_dim)

            renderings.append(rasterize_to_pixels(
                means2d,
                conics,
                self.rgbs.view(1, -1, 3)[:, start:end].expand(len(w2cs_chunk), -1, -1),
                self.opacity[start:end].squeeze(-1).unsqueeze(0).expand(len(w2cs_chunk), -1),
                224,
                224,
                self.tile_size,
                isect_offsets,
                gauss_ids,
            )[0])

        return torch.cat(renderings)

    @property
    def opacity(self):
        if self.opt_opacity:
            return torch.sigmoid(self.opacity_logit)
        else:
            return self._opacity

    @property
    def scales(self):
        if self.opt_scales:
            return self.scales_log.exp()
        else:
            return (self.depth_log.exp() / self.fx).view(-1, 1)

    def get_xyz(self):
        if self.opt_depth_only:
            pts3d_cam = fast_depthmap_to_pts3d(self.depth_log.exp(), self.pixels, torch.cat([self.fx, self.fy], -1),
                                               torch.cat([self.cx, self.cy], -1))
            return geotrf(self.c2ws, pts3d_cam).view(-1, 3)
        else:
            return self.xyz
