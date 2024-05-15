import traceback
from pathlib import Path
from typing import Optional, List, Any

import cv2
import numpy as np
import torch
from PIL import Image
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.utils.comms import get_rank

from generfstudio.generfstudio_utils import central_crop_v2

try:
    from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
    from dust3r.model import AsymmetricCroCo3DStereo
    from dust3r.image_pairs import make_pairs
    from dust3r.inference import inference
    from dust3r.utils.image import load_images
except:
    pass


def convert_from_inner(inner_outputs: DataparserOutputs, dest: Path, crop: Optional[int]) -> List[Any]:
    frames = []

    for image_index, image_path in enumerate(inner_outputs.image_filenames):
        camera = inner_outputs.cameras[image_index]
        image = Image.open(image_path)
        K = camera.get_intrinsics_matrices().numpy()
        distortion_params = camera.distortion_params
        distortion_params = np.array(
            [distortion_params[0], distortion_params[1], distortion_params[4], distortion_params[5],
             distortion_params[2], distortion_params[3], 0, 0, ])
        newK, roi = cv2.getOptimalNewCameraMatrix(K, distortion_params, image.size, 0)
        image = cv2.undistort(np.asarray(image), K, distortion_params, None, newK)
        x, y, w, h = roi
        image = Image.fromarray(image[y: y + h, x: x + w])

        image_fx = newK[0, 0]
        image_fy = newK[1, 1]
        image_cx = newK[0, 2]
        image_cy = newK[1, 2]

        if crop is not None:
            cropped_image = central_crop_v2(image)
            image_cx -= ((image.size[0] - cropped_image.size[0]) / 2)
            image_cy -= ((image.size[1] - cropped_image.size[1]) / 2)
            image_cx *= (crop / cropped_image.size[0])
            image_cy *= (crop / cropped_image.size[1])
            image_fx *= (crop / cropped_image.size[0])
            image_fy *= (crop / cropped_image.size[1])
            image = cropped_image.resize((crop, crop), resample=Image.LANCZOS)

        image_path = dest / "images" / image_path.name
        image_path.parent.mkdir(exist_ok=True, parents=True)
        image.save(image_path)

        frames.append({
            "file_path": f"images/{image_path.name}",
            "transform_matrix": camera.camera_to_worlds.tolist(),
            "w": image.size[0],
            "h": image.size[1],
            "fl_x": image_fx,
            "fl_y": image_fy,
            "cx": image_cx,
            "cy": image_cy,
        })

    return frames


def get_xyz_in_camera(inner_outputs: DataparserOutputs, frames: List[Any]):
    points = inner_outputs.metadata["points3D_xyz"]

    c2ws = inner_outputs.cameras.camera_to_worlds
    rot = c2ws[:, :3, :3].transpose(1, 2)  # (B, 3, 3)
    trans = -torch.bmm(rot, c2ws[:, :3, 3:])  # (B, 3, 1)
    w2cs = torch.cat((rot, trans), dim=-1)

    xyz_rot = torch.matmul(w2cs[:, None, :3, :3], points.unsqueeze(-1))[..., 0]
    xyz_in_camera = xyz_rot + w2cs[:, None, :3, 3]

    fx = torch.FloatTensor([x["fl_x"] for x in frames])
    fy = torch.FloatTensor([x["fl_y"] for x in frames])
    cx = torch.FloatTensor([x["cx"] for x in frames])
    cy = torch.FloatTensor([x["cy"] for x in frames])

    # xyz_in_neighbor[..., 1:] *= -1  # RUB to RDF
    # z_pos = xyz_in_neighbor[..., 2]
    # K = torch.eye(3).unsqueeze(0).repeat(len(frames), 1, 1)
    # K[:, 0, 0] = fx
    # K[:, 1, 1] = fy
    # K[:, 0, 2] = cx
    # K[:, 1, 2] = cy
    # uv = torch.matmul(K[:, None, :3, :3], xyz_in_neighbor.unsqueeze(-1))[..., 0]

    uv = -xyz_in_camera[:, :, :2] / xyz_in_camera[:, :, 2:]
    focal = torch.cat([fx.view(-1, 1), -fy.view(-1, 1)], -1)
    uv *= focal.unsqueeze(1)
    center = torch.cat([cx.view(-1, 1), cy.view(-1, 1)], -1)
    uv += center.unsqueeze(1)

    width = torch.IntTensor([x["w"] for x in frames])
    height = torch.IntTensor([x["h"] for x in frames])

    is_valid_x = torch.logical_and(0 <= uv[..., 0], uv[..., 0] < width.unsqueeze(1) - 1)
    is_valid_y = torch.logical_and(0 <= uv[..., 1], uv[..., 1] < height.unsqueeze(1) - 1)
    is_valid_z = xyz_in_camera[:, :, 2] < 0
    is_valid_points = torch.logical_and(torch.logical_and(is_valid_x, is_valid_y), is_valid_z)

    return xyz_in_camera, uv, is_valid_points, width, height


def create_dust3r_poses(model, dust3r_path: Path, image_path: Path, frames: List[Any],
                        frame_intersections: Optional[torch.Tensor], num_neighbors: int = 10, num_iters: int = 5000,
                        lr: float = 0.01, schedule: str = "linear"):
    rank = get_rank()
    device = f"cuda:{rank}"

    with torch.no_grad():
        images = load_images([str(image_path / x["file_path"]) for x in frames], size=224, verbose=False)
        if frame_intersections is not None:
            pairs = []
            for i in range(len(images)):
                _, closest = torch.sort(frame_intersections[i], descending=True)
                for neighbor in closest[:num_neighbors]:
                    if neighbor != i:
                        pairs.append((images[i], images[neighbor]))
        else:
            pairs = make_pairs(images, scene_graph="complete", prefilter=None, symmetrize=True)

        output = inference(pairs, model, device, batch_size=50)
        scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)

        dust3r_path.mkdir(parents=True, exist_ok=True)
        torch.cuda.empty_cache()

    try:
        alignment_loss = scene.compute_global_alignment(init="mst", niter=num_iters, schedule=schedule, lr=lr)
    except:
        traceback.print_exc()
        (dust3r_path / "FAIL").touch()
        return

    with torch.no_grad():
        focals = scene.get_focals()
        pp = scene.get_principal_points()
        im_poses = scene.get_im_poses()
        depth = scene.get_depthmaps(raw=True)

        for frame, depth_map in zip(frames, depth):
            frame_path_png = Path(dust3r_path / frame["file_path"])
            frame_path_png.parent.mkdir(exist_ok=True, parents=True)
            frame_path_pt = frame_path_png.parent / f"{frame_path_png.stem}.pt"
            torch.save(depth_map.cpu().detach().clone(), frame_path_pt)
            # frame_path_gz = frame_path_png.parent / f"{frame_path_png.stem}.npy.gz"
            # f = gzip.GzipFile(str(frame_path_gz), "w")
            # numpy.save(file=f, arr=depth_map.detach().cpu().numpy())
            # f.close()

        pts3d = scene.get_pts3d(raw=True)
        min_bounds = pts3d.view(-1, 3).min(dim=0)[0]
        max_bounds = pts3d.view(-1, 3).max(dim=0)[0]

        torch.save({
            "focals": focals.cpu().detach().clone(),
            "pp": pp.cpu().detach().clone(),
            "poses": im_poses.cpu().detach().clone(),
            "alignment_loss": alignment_loss,
            "num_neighbors": num_neighbors,
            "alignment_iters": num_iters,
            "alignment_lr": lr,
            "alignment_schedule": schedule,
            "near": float(depth.min()),
            "far": float(depth.max()),
            "scene_box": torch.stack([min_bounds, max_bounds]).cpu().detach().clone(),
            "acc_test": True,
        }, dust3r_path / "scene.pt")


        # view1 = torch.IntTensor(output["view1"]["idx"])
        # view2 = torch.IntTensor(output["view2"]["idx"])
        # pred1_conf = torch.FloatTensor(output["pred1"]["conf"]).view(output["pred1"]["conf"].shape[0], -1, 1)
        # pred2_conf = torch.FloatTensor(output["pred2"]["conf"]).view(output["pred2"]["conf"].shape[0], -1, 1)
        # quantiles = torch.FloatTensor([0.1, 0.5, 0.9]).to(pred1_conf.device)
        # torch.save({
        #     "view1": view1.cpu().detach().clone(),
        #     "view2": view2.cpu().detach().clone(),
        #     "pred1_conf_mean": pred1_conf.mean(dim=-1).cpu().detach().clone(),
        #     "pred2_conf_mean": pred2_conf.mean(dim=-1).cpu().detach().clone(),
        # }, dust3r_path / "confs.pt")
