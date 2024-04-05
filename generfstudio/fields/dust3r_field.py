from copy import deepcopy
from typing import Optional

import numpy as np
import torch
from gsplat import project_gaussians, rasterize_gaussians
from nerfstudio.cameras.cameras import Cameras
from torch import nn
from torch.nn import functional as F

from generfstudio.generfstudio_constants import RGB, FEATURES, ACCUMULATION, ALIGNMENT_LOSS, NEIGHBOR_RESULTS, DEPTH

try:
    from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
    from dust3r.image_pairs import make_pairs
    from dust3r.inference import inference, load_model
except:
    pass


def projection_matrix(znear, zfar, fovx, fovy):
    top = znear * torch.tan(0.5 * fovy)
    bottom = -top
    right = znear * torch.tan(0.5 * fovx)
    left = -right

    zeros = torch.zeros_like(left)
    ones = torch.ones_like(left)

    return torch.cat(
        [
            torch.cat([2 * znear / (right - left), zeros, (right + left) / (right - left), zeros], -1).unsqueeze(1),
            torch.cat([zeros, 2 * znear / (top - bottom), (top + bottom) / (top - bottom), zeros], -1).unsqueeze(1),
            torch.cat([zeros, zeros, torch.full_like(left, (zfar + znear) / (zfar - znear)),
                       torch.full_like(left, -1.0 * zfar * znear / (zfar - znear))], -1).unsqueeze(1),
            torch.cat([zeros, zeros, ones, zeros], -1).unsqueeze(1)
        ],
        1,
    )


class Dust3rField(nn.Module):

    def __init__(
            self,
            checkpoint_path: str,
            in_feature_dim: int = 512,
            out_feature_dim: int = 128,
            image_dim: int = 224,
            project_features: bool = True,
            alignment_iter: int = 300,
            alignment_lr: float = 0.01,
            alignment_schedule: str = "cosine",
            use_confidence_opacity: bool = False,
    ) -> None:
        super().__init__()
        self.model = load_model(checkpoint_path, "cpu")
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.image_dim = image_dim
        self.project_features = project_features
        self.alignment_iter = alignment_iter
        self.alignment_lr = alignment_lr
        self.alignment_schedule = alignment_schedule
        self.use_confidence_opacity = use_confidence_opacity
        self.feature_layer = nn.Linear(in_feature_dim, out_feature_dim)

    @torch.cuda.amp.autocast(enabled=False)
    def splat_gaussians(self, camera: Cameras, xyz: torch.Tensor, scales: torch.Tensor, conf: torch.Tensor,
                        w2c: torch.Tensor, projmat: torch.Tensor, rgbs: torch.Tensor, features: Optional[torch.Tensor],
                        return_depth: bool):
        xys, depths, radii, conics, comp, num_tiles_hit, cov3d = project_gaussians(
            xyz,
            scales,
            1,
            torch.cat([torch.ones_like(conf), torch.zeros_like(xyz)], -1),
            w2c[:3],
            projmat @ w2c,
            camera.fx[0].item(),
            camera.fy[0].item(),
            camera.cx[0].item(),
            camera.cy[0].item(),
            camera.height[0].item(),
            camera.width[0].item(),
            16,
        )

        inputs = [rgbs]
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
            8,
            background=torch.zeros(inputs.shape[-1], device=xys.device),
            return_alpha=True,
        )

        outputs = {RGB: splat_results[..., :3], ACCUMULATION: alpha.unsqueeze(-1)}

        if features is not None:
            outputs[FEATURES] = splat_results[..., 3:3 + features.shape[-1]]

        if return_depth:
            outputs[DEPTH] = splat_results[..., -1:]

        return outputs

    def forward(self, target_cameras: Cameras, neighbor_cameras: Cameras, cond_rgbs: torch.Tensor,
                cond_features: torch.Tensor):
        with torch.no_grad():
            cond_rgbs = F.interpolate(cond_rgbs.view(-1, *cond_rgbs.shape[2:]), self.image_dim, mode="bilinear")
            cond_rgbs = cond_rgbs.view(len(target_cameras), -1, *cond_rgbs.shape[1:])
            cond_rgbs_duster_input = cond_rgbs * 2 - 1

            # bilinear interpolation can be expensive for high-dimensional features
            b, d, h, w = cond_features.shape
            cond_features = cond_features.permute(0, 2, 3, 1)
            cond_features = self.feature_layer(cond_features.view(-1, d)).view(b, h, w, -1)
            cond_features = F.interpolate(cond_features.permute(0, 3, 1, 2), self.image_dim)
            cond_features = cond_features.view(len(target_cameras), -1, *cond_features.shape[1:])

        neighbor_cameras = deepcopy(neighbor_cameras)
        neighbor_cameras.rescale_output_resolution(self.image_dim / neighbor_cameras.height[0, 0].item())

        neighbor_c2ws = torch.eye(4).view(1, 1, 4, 4).repeat(neighbor_cameras.shape[0], neighbor_cameras.shape[1], 1, 1)
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

        fovx = 2 * torch.atan(target_cameras.width / (2 * target_cameras.fx))
        fovy = 2 * torch.atan(target_cameras.height / (2 * target_cameras.fy))
        projmats = projection_matrix(0.001, 1000, fovx, fovy)

        rgbs = []
        features = []
        accumulations = []
        if not self.training:
            depths = []
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

            neighbor_fovx = 2 * torch.atan(neighbor_cameras.width / (2 * neighbor_cameras.fx))
            neighbor_fovy = 2 * torch.atan(neighbor_cameras.height / (2 * neighbor_cameras.fy))
            neighbor_projmats = projection_matrix(0.001, 1000, neighbor_fovx.view(-1, 1),
                                                  neighbor_fovy.view(-1, 1)).view(neighbor_w2cs.shape)

        alignment_loss = 0
        for i in range(len(target_cameras)):
            # Imitate what load_images does. "true_shape" is probably not necessary
            scene_images = [{
                "img": cond_rgbs_duster_input[i, j:j + 1],
                "instance": str(j),
                "idx": j,
                "true_shape": np.int32([cond_rgbs_duster_input[i, j:j + 1].shape[-2:]]),
            } for j in range(len(cond_rgbs_duster_input[i]))]
            pairs = make_pairs(scene_images, scene_graph="complete", prefilter=None, symmetrize=True)
            outputs = inference(pairs, self.model, cond_rgbs.device, batch_size=20)
            scene = global_aligner(outputs, device=cond_rgbs.device, mode=GlobalAlignerMode.PointCloudOptimizer)

            scene.preset_pose(neighbor_c2ws[i])
            scene.preset_focal(neighbor_cameras[i].fx.cpu())  # Assume fx and fy are almost equal

            with torch.enable_grad():
                alignment_loss += scene.compute_global_alignment(init="known_poses", niter=self.alignment_iter,
                                                                 schedule=self.alignment_schedule, lr=self.alignment_lr)
            pts3d = scene.get_pts3d()

            scene_xyz = []
            scene_rgbs = []
            scene_features = []
            scene_scales = []
            if self.use_confidence_opacity:
                scene_opacity = []
            for j in range(len(neighbor_cameras[i])):
                image_xyz = pts3d[j].detach().view(-1, 3)
                image_rgb = cond_rgbs[i, j].permute(1, 2, 0).view(-1, 3)
                image_features = cond_features[i, j].permute(1, 2, 0)
                image_features = image_features.view(-1, image_features.shape[-1])
                pixel_area = neighbor_cameras[i].generate_rays(camera_indices=j).pixel_area.view(-1, 1)

                scene_xyz.append(image_xyz)
                scene_rgbs.append(image_rgb)
                scene_features.append(image_features)

                camera_pos = neighbor_cameras[i, j].camera_to_worlds[:, 3]
                distances = (image_xyz - camera_pos).norm(dim=-1, keepdim=True)
                scene_scales.append(pixel_area * distances)

                if self.use_confidence_opacity:
                    confidence = torch.log(scene.im_conf[j].detach().view(-1, 1))
                    scene_opacity.append(confidence)

            scene_xyz = torch.cat(scene_xyz)
            scene_rgbs = torch.cat(scene_rgbs)
            scene_scales = torch.cat(scene_scales).repeat(1, 3)
            scene_features = torch.cat(scene_features)
            scene_opacity = torch.cat(scene_opacity).clamp_max(1) if self.use_confidence_opacity \
                else torch.ones_like(scene_xyz[..., :1])

            if not self.training:
                scene_neighbor_rgbs = []
                scene_neighbor_depths = []
                scene_neighbor_accumulations = []
                for j in range(len(neighbor_cameras[i])):
                    neighbor_renderings = self.splat_gaussians(neighbor_cameras[i, j], scene_xyz, scene_scales,
                                                               scene_opacity, neighbor_w2cs[i, j],
                                                               neighbor_projmats[i, j], scene_rgbs, None, True)
                    scene_neighbor_rgbs.append(neighbor_renderings[RGB])
                    scene_neighbor_depths.append(neighbor_renderings[DEPTH])
                    scene_neighbor_accumulations.append(neighbor_renderings[ACCUMULATION])

                neighbor_rgbs.append(torch.cat(scene_neighbor_rgbs, 1))
                neighbor_depths.append(torch.cat(scene_neighbor_depths, 1))
                neighbor_accumulations.append(torch.cat(scene_neighbor_accumulations, 1))

            rendering = self.splat_gaussians(target_cameras[i], scene_xyz, scene_scales, scene_opacity, w2cs[i],
                                             projmats[i], scene_rgbs, scene_features, not self.training)

            rgbs.append(rendering[RGB])
            accumulations.append(rendering[ACCUMULATION])
            features.append(rendering[FEATURES])
            if not self.training:
                depths.append(rendering[DEPTH])

        outputs = {
            RGB: torch.stack(rgbs),
            FEATURES: torch.stack(features),
            ACCUMULATION: torch.stack(accumulations),
            ALIGNMENT_LOSS: alignment_loss / len(target_cameras),
        }

        if not self.training:
            outputs[DEPTH] = torch.stack(depths)
            outputs[NEIGHBOR_RESULTS] = {
                RGB: torch.stack(neighbor_rgbs),
                DEPTH: torch.stack(neighbor_depths),
                ACCUMULATION: torch.stack(neighbor_accumulations),
            }

        return outputs
