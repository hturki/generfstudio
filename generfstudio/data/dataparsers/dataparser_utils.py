from pathlib import Path
from typing import Optional, Tuple, List, Any

import cv2
import numpy as np
import torch
from PIL import Image
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs

from generfstudio.generfstudio_utils import central_crop_v2


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
            "file_path": str(image_path),
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
