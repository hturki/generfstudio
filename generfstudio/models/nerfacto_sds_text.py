from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Tuple, Type, Optional, Union, Any

import numpy as np
import torch
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.generative.deepfloyd import DeepFloyd
from nerfstudio.generative.positional_text_embeddings import PositionalTextEmbeddings
from nerfstudio.generative.stable_diffusion import StableDiffusion
from torch import Tensor
from torch.nn import Parameter

from nerfstudio.cameras.camera_optimizers import CameraOptimizer, CameraOptimizerConfig
from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.fields.density_fields import HashMLPDensityField
from nerfstudio.fields.nerfacto_field import NerfactoField
from nerfstudio.model_components.losses import (
    MSELoss,
    distortion_loss,
    interlevel_loss,
    orientation_loss,
    pred_normal_loss,
    scale_gradients_by_distance_squared,
)
from nerfstudio.model_components.ray_samplers import ProposalNetworkSampler, UniformSampler
from nerfstudio.model_components.renderers import AccumulationRenderer, DepthRenderer, NormalsRenderer, RGBRenderer
from nerfstudio.model_components.scene_colliders import NearFarCollider
from nerfstudio.model_components.shaders import NormalsShader
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps
import torchvision.transforms as transforms

def randomize_poses(
    size: int,
    resolution: int,
    device: Union[torch.device, str],
    radius_mean: float = 1.0,
    radius_std: float = 0.1,
    central_rotation_range: Tuple[float, float] = (0, 360),
    vertical_rotation_range: Tuple[float, float] = (-90, 0),
    #focal_range: Tuple[float, float] = (0.75, 1.35),
    focal_lengths: Tuple[float, float] = (0.75, 1.35),
    jitter_std: float = 0.01,
    center: Tuple[float, float, float] = (0, 0, 0),
) -> Tuple[Cameras, Tensor, Tensor]:
    """generate random poses from an orbit camera
    Args:
        size: batch size of generated poses.
        device: where to allocate the output.
        radius_mean: mean radius of the orbit camera.
        radius_std: standard deviation of the radius of the orbit camera.
        central_rotation_range: amount that we rotate around the center of the object
        vertical_rotation_range: amount that we allow the cameras to pan up and down from horizontal
        focal_range: focal length range
        jitter_std: standard deviation of the jitter added to the camera position
        center: center of the object
    Return:
        poses: [size, 4, 4]
    """

    vertical_rotation_range = (
        vertical_rotation_range[0] + 90,
        vertical_rotation_range[1] + 90,
    )
    # This is the uniform sample on the part of the sphere we care about where 0 = 0 degrees and 1 = 360 degrees
    sampled_uniform = (
        torch.rand(size) * (vertical_rotation_range[1] - vertical_rotation_range[0]) + vertical_rotation_range[0]
    ) / 180
    vertical_rotation = torch.arccos(1 - 2 * sampled_uniform)
    central_rotation = torch.deg2rad(
        torch.rand(size) * (central_rotation_range[1] - central_rotation_range[0]) + central_rotation_range[0]
    )

    c_cos = torch.cos(central_rotation)
    c_sin = torch.sin(central_rotation)
    v_cos = torch.cos(vertical_rotation)
    v_sin = torch.sin(vertical_rotation)
    zeros = torch.zeros_like(central_rotation)
    ones = torch.ones_like(central_rotation)

    rot_z = torch.stack(
        [
            torch.stack([c_cos, -c_sin, zeros], dim=-1),
            torch.stack([c_sin, c_cos, zeros], dim=-1),
            torch.stack([zeros, zeros, ones], dim=-1),
        ],
        dim=-2,
    )

    rot_x = torch.stack(
        [
            torch.stack([ones, zeros, zeros], dim=-1),
            torch.stack([zeros, v_cos, -v_sin], dim=-1),
            torch.stack([zeros, v_sin, v_cos], dim=-1),
        ],
        dim=-2,
    )

    # Default directions are facing in the -z direction, so origins should face opposite way
    origins = torch.stack([torch.tensor([0, 0, 1])] * size, dim=0)
    origins = (origins * radius_mean) + (origins * (torch.randn((origins.shape)) * radius_std))
    R = torch.bmm(rot_z, rot_x)  # Want to have Rx @ Ry @ origin
    t = (
        torch.bmm(R, origins.unsqueeze(-1))
        + torch.randn((size, 3, 1)) * jitter_std
        + torch.tensor(center)[None, :, None]
    )
    camera_to_worlds = torch.cat([R, t], dim=-1)

    #focals = torch.rand(size) * (focal_range[1] - focal_range[0]) + focal_range[0]

    cameras = Cameras(
        camera_to_worlds=camera_to_worlds,
        fx=focal_lengths[0], #focals * resolution,
        fy=focal_lengths[1], #focals * resolution,
        cx=resolution,
        cy=resolution,
    ).to(device)

    return cameras, torch.rad2deg(vertical_rotation), torch.rad2deg(central_rotation)

def generate_camera_poses_translation(base_pose, num_poses=5, translation_increment=1):
    """
    Generate more camera poses around a given base camera pose by translation.

    Parameters:
    - base_pose: 3x4 numpy array representing the base camera pose (3x3 rotation matrix + translation vector).
    - num_poses: Number of additional camera poses to generate.
    - translation_increment: Increment in translation for each pose.

    Returns:
    - List of 3x4 numpy arrays representing the generated camera poses.
    """
    base_pose = base_pose.cpu().detach().numpy()
    camera_poses = [base_pose]

    # Extract rotation matrix and translation vector from the base pose
    rotation_matrix = base_pose[:, :3]
    translation_vector = base_pose[:, 3]

    # Generate additional camera poses by translating along different axes
    for i in range(1, num_poses + 1):
        # Translate along x-axis
        new_translation_x = translation_vector + np.array([translation_increment, 0, 0])
        new_pose_x = np.column_stack((rotation_matrix, new_translation_x))
        camera_poses.append(new_pose_x)

        # Translate along y-axis
        new_translation_y = translation_vector + np.array([0, translation_increment, 0])
        new_pose_y = np.column_stack((rotation_matrix, new_translation_y))
        camera_poses.append(new_pose_y)

        # Translate along z-axis
        new_translation_z = translation_vector + np.array([0, 0, translation_increment])
        new_pose_z = np.column_stack((rotation_matrix, new_translation_z))
        camera_poses.append(new_pose_z)

    return camera_poses

@dataclass
class NerfactoSDSModelConfig(ModelConfig):
    """Nerfacto Model Config"""

    _target: Type = field(default_factory=lambda: NerfactoSDSModel)
    near_plane: float = 0.05
    """How far along the ray to start sampling."""
    far_plane: float = 1000.0
    """How far along the ray to stop sampling."""
    background_color: Literal["random", "last_sample", "black", "white"] = "last_sample"
    """Whether to randomize the background color."""
    hidden_dim: int = 64
    """Dimension of hidden layers"""
    hidden_dim_color: int = 64
    """Dimension of hidden layers for color network"""
    hidden_dim_transient: int = 64
    """Dimension of hidden layers for transient network"""
    num_levels: int = 16
    """Number of levels of the hashmap for the base mlp."""
    base_res: int = 16
    """Resolution of the base grid for the hashgrid."""
    max_res: int = 2048
    """Maximum resolution of the hashmap for the base mlp."""
    log2_hashmap_size: int = 19
    """Size of the hashmap for the base mlp"""
    features_per_level: int = 2
    """How many hashgrid features per level"""
    num_proposal_samples_per_ray: Tuple[int, ...] = (256, 96)
    """Number of samples per ray for each proposal network."""
    num_nerf_samples_per_ray: int = 48
    """Number of samples per ray for the nerf network."""
    proposal_update_every: int = 5
    """Sample every n steps after the warmup"""
    proposal_warmup: int = 5000
    """Scales n from 1 to proposal_update_every over this many steps"""
    num_proposal_iterations: int = 2
    """Number of proposal network iterations."""
    use_same_proposal_network: bool = False
    """Use the same proposal network. Otherwise use different ones."""
    proposal_net_args_list: List[Dict] = field(
        default_factory=lambda: [
            {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 128, "use_linear": False},
            {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 256, "use_linear": False},
        ]
    )
    """Arguments for the proposal density fields."""
    proposal_initial_sampler: Literal["piecewise", "uniform"] = "piecewise"
    """Initial sampler for the proposal network. Piecewise is preferred for unbounded scenes."""
    interlevel_loss_mult: float = 1.0
    """Proposal loss multiplier."""
    distortion_loss_mult: float = 0.002
    """Distortion loss multiplier."""
    orientation_loss_mult: float = 0.0001
    """Orientation loss multiplier on computed normals."""
    pred_normal_loss_mult: float = 0.001
    """Predicted normal loss multiplier."""
    use_proposal_weight_anneal: bool = True
    """Whether to use proposal weight annealing."""
    use_appearance_embedding: bool = True
    """Whether to use an appearance embedding."""
    use_average_appearance_embedding: bool = True
    """Whether to use average appearance embedding or zeros for inference."""
    proposal_weights_anneal_slope: float = 10.0
    """Slope of the annealing function for the proposal weights."""
    proposal_weights_anneal_max_num_iters: int = 1000
    """Max num iterations for the annealing function."""
    use_single_jitter: bool = True
    """Whether use single jitter or not for the proposal networks."""
    predict_normals: bool = False
    """Whether to predict normals or not."""
    disable_scene_contraction: bool = False
    """Whether to disable scene contraction or not."""
    use_gradient_scaling: bool = False
    """Use gradient scaler where the gradients are lower for points closer to the camera."""
    implementation: Literal["tcnn", "torch"] = "tcnn"
    """Which implementation to use for the model."""
    appearance_embed_dim: int = 32
    """Dimension of the appearance embedding."""
    average_init_density: float = 0.01
    """Average initial density output from MLP. """
    camera_optimizer: CameraOptimizerConfig = field(default_factory=lambda: CameraOptimizerConfig(mode="SO3xR3"))
    """Config of the camera optimizer to use"""

    ########################### SDS (Generfacto) ###################
    taper_range: Tuple[int, int] = (0, 1000)
    """Range of step values for the density tapering"""
    taper_strength: Tuple[float, float] = (1.0, 0.0)
    """Strength schedule of center density"""
    location_based_prompting: bool = True
    """enables location based prompting"""
    interpolated_prompting: bool = False
    """enables interpolated location prompting"""
    positional_prompting: Literal["discrete", "interpolated", "off"] = "off"
    """ how to incorporate position into prompt"""
    prompt: str = "a chair with green cushions in a white background and no floor back view" #a chair in a room" #"a high quality photo" #
    """prompt for stable dreamfusion"""
    top_prompt: str = "" # overhead view"
    """appended to prompt for overhead view"""
    side_prompt: str = "" # side view"
    """appended to prompt for side view"""
    front_prompt: str = "" # front view"
    """appended to prompt for front view"""
    back_prompt: str = "" # back view"
    """appended to prompt for back view"""
    guidance_scale: float = 10
    """guidance scale for sds loss"""
    diffusion_device: Optional[str] = None #"cuda:1"
    """device for diffusion model"""
    diffusion_model: Literal["stablediffusion", "deepfloyd"] = "stablediffusion"
    """diffusion model for SDS loss"""
    sd_version: str = "1-5"
    """model version when using stable diffusion"""
    start_sds_training: int = 100
    """Start sds training after this many iterations"""

class NerfactoSDSModel(Model):
    """Nerfacto model

    Args:
        config: Nerfacto configuration to instantiate model
    """

    config: NerfactoSDSModelConfig

    def __init__(
        self,
        config: NerfactoSDSModelConfig,
        metadata: Dict[str, Any],
        **kwargs,
    ) -> None:
        self.prompt = config.prompt
        self.cur_prompt = config.prompt
        self.sd_version = config.sd_version
        self.grad_scaler = kwargs["grad_scaler"]

        self.guidance_scale = config.guidance_scale
        self.top_prompt = config.top_prompt
        self.side_prompt = config.side_prompt
        self.back_prompt = config.back_prompt
        self.front_prompt = config.front_prompt
        self.train_sds = False
        self.train_with_sds = False
        self.iteration_index = 0

        self.diffusion_device = (
            torch.device(kwargs["device"]) if config.diffusion_device is None else torch.device(config.diffusion_device)
        )

        self.test_cameras = metadata.get('test_cameras', None)
        self.test_cameras.rescale_output_resolution(scaling_factor=1.0 / 8)
        super().__init__(config=config, **kwargs)

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        ##################################################
        if self.config.diffusion_model == "stablediffusion":
            self._diffusion_model = StableDiffusion(self.diffusion_device, version=self.sd_version)
        elif self.config.diffusion_model == "deepfloyd":
            self._diffusion_model = DeepFloyd(self.diffusion_device)

        self.text_embeddings = PositionalTextEmbeddings(
            base_prompt=self.cur_prompt,
            top_prompt=self.cur_prompt + self.top_prompt,
            side_prompt=self.cur_prompt + self.side_prompt,
            back_prompt=self.cur_prompt + self.back_prompt,
            front_prompt=self.cur_prompt + self.front_prompt,
            diffusion_model=self._diffusion_model,
            positional_prompting=self.config.positional_prompting,
        )
        ###################################################

        if self.config.disable_scene_contraction:
            scene_contraction = None
        else:
            scene_contraction = SceneContraction(order=float("inf"))

        appearance_embedding_dim = self.config.appearance_embed_dim if self.config.use_appearance_embedding else 0

        # Fields
        self.field = NerfactoField(
            self.scene_box.aabb,
            hidden_dim=self.config.hidden_dim,
            num_levels=self.config.num_levels,
            max_res=self.config.max_res,
            base_res=self.config.base_res,
            features_per_level=self.config.features_per_level,
            log2_hashmap_size=self.config.log2_hashmap_size,
            hidden_dim_color=self.config.hidden_dim_color,
            hidden_dim_transient=self.config.hidden_dim_transient,
            spatial_distortion=scene_contraction,
            num_images=self.num_train_data,
            use_pred_normals=self.config.predict_normals,
            use_average_appearance_embedding=self.config.use_average_appearance_embedding,
            appearance_embedding_dim=appearance_embedding_dim,
            average_init_density=self.config.average_init_density,
            implementation=self.config.implementation,
        )

        self.camera_optimizer: CameraOptimizer = self.config.camera_optimizer.setup(
            num_cameras=self.num_train_data, device="cpu"
        )
        self.density_fns = []
        num_prop_nets = self.config.num_proposal_iterations
        # Build the proposal network(s)
        self.proposal_networks = torch.nn.ModuleList()
        if self.config.use_same_proposal_network:
            assert len(self.config.proposal_net_args_list) == 1, "Only one proposal network is allowed."
            prop_net_args = self.config.proposal_net_args_list[0]
            network = HashMLPDensityField(
                self.scene_box.aabb,
                spatial_distortion=scene_contraction,
                **prop_net_args,
                average_init_density=self.config.average_init_density,
                implementation=self.config.implementation,
            )
            self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for _ in range(num_prop_nets)])
        else:
            for i in range(num_prop_nets):
                prop_net_args = self.config.proposal_net_args_list[min(i, len(self.config.proposal_net_args_list) - 1)]
                network = HashMLPDensityField(
                    self.scene_box.aabb,
                    spatial_distortion=scene_contraction,
                    **prop_net_args,
                    average_init_density=self.config.average_init_density,
                    implementation=self.config.implementation,
                )
                self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for network in self.proposal_networks])

        # Samplers
        def update_schedule(step):
            return np.clip(
                np.interp(step, [0, self.config.proposal_warmup], [0, self.config.proposal_update_every]),
                1,
                self.config.proposal_update_every,
            )

        # Change proposal network initial sampler if uniform
        initial_sampler = None  # None is for piecewise as default (see ProposalNetworkSampler)
        if self.config.proposal_initial_sampler == "uniform":
            initial_sampler = UniformSampler(single_jitter=self.config.use_single_jitter)

        self.proposal_sampler = ProposalNetworkSampler(
            num_nerf_samples_per_ray=self.config.num_nerf_samples_per_ray,
            num_proposal_samples_per_ray=self.config.num_proposal_samples_per_ray,
            num_proposal_network_iterations=self.config.num_proposal_iterations,
            single_jitter=self.config.use_single_jitter,
            update_sched=update_schedule,
            initial_sampler=initial_sampler,
        )

        # Collider
        self.collider = NearFarCollider(near_plane=self.config.near_plane, far_plane=self.config.far_plane)

        # renderers
        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer(method="median")
        self.renderer_expected_depth = DepthRenderer(method="expected")
        self.renderer_normals = NormalsRenderer()

        # shaders
        self.normals_shader = NormalsShader()

        # losses
        self.rgb_loss = MSELoss()
        self.step = 0
        # metrics
        from torchmetrics.functional import structural_similarity_index_measure
        from torchmetrics.image import PeakSignalNoiseRatio
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        self.step = 0

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        param_groups["proposal_networks"] = list(self.proposal_networks.parameters())
        param_groups["fields"] = list(self.field.parameters())
        self.camera_optimizer.get_param_groups(param_groups=param_groups)
        return param_groups

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = []
        if self.config.use_proposal_weight_anneal:
            # anneal the weights of the proposal network before doing PDF sampling
            N = self.config.proposal_weights_anneal_max_num_iters

            def set_anneal(step):
                # https://arxiv.org/pdf/2111.12077.pdf eq. 18
                self.step = step
                train_frac = np.clip(step / N, 0, 1)
                self.step = step

                def bias(x, b):
                    return b * x / ((b - 1) * x + 1)

                anneal = bias(train_frac, self.config.proposal_weights_anneal_slope)
                self.proposal_sampler.set_anneal(anneal)

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=set_anneal,
                )
            )
            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=self.proposal_sampler.step_cb,
                )
            )

            def taper_density(step: int):
                self.density_strength = np.interp(step, self.config.taper_range, self.config.taper_strength)

            def start_training_sds(step: int):
                self.train_sds = True

            def set_even_iteration(step: int):
                if step % 2 == 0:
                    self.train_with_sds = True
                else:
                    self.train_with_sds = False

            def set_iteration(step: int):
                self.iteration_index = step

            callbacks += [
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    func=taper_density,
                    update_every_num_iters=1,
                ),
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    iters=(self.config.start_sds_training,),
                    func=start_training_sds,
                ),
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=set_even_iteration,
                ),
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=set_iteration,
                ),
            ]
        return callbacks

    def get_outputs_for_SDS(self, camera_ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        """Takes in camera parameters and computes the output of the model.

        Args:
            camera_ray_bundle: ray bundle to calculate outputs over
        """
        input_device = camera_ray_bundle.directions.device
        num_rays_per_chunk = self.config.eval_num_rays_per_chunk
        image_height, image_width = camera_ray_bundle.origins.shape[:2]
        num_rays = len(camera_ray_bundle)
        outputs_lists = defaultdict(list)
        for i in range(0, num_rays, num_rays_per_chunk):
            start_idx = i
            end_idx = i + num_rays_per_chunk
            ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
            # move the chunk inputs to the model device
            ray_bundle = ray_bundle.to(self.device)
            outputs = self.forward(ray_bundle=ray_bundle)
            for output_name, output in outputs.items():  # type: ignore
                if not isinstance(output, torch.Tensor) or output_name != 'rgb':
                    # TODO: handle lists of tensors as well
                    continue
                # move the chunk outputs from the model device back to the device of the inputs.
                outputs_lists[output_name].append(output.to(input_device))
        outputs = {}
        for output_name, outputs_list in outputs_lists.items():
            outputs[output_name] = torch.cat(outputs_list).view(image_height, image_width, -1)  # type: ignore
        return outputs

    def get_outputs(self, ray_bundle: RayBundle):
        # apply the camera optimizer pose tweaks
        if self.training:
            self.camera_optimizer.apply_to_raybundle(ray_bundle)
        ray_samples: RaySamples
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        field_outputs = self.field.forward(ray_samples, compute_normals=self.config.predict_normals)
        if self.config.use_gradient_scaling:
            field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)

        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        with torch.no_grad():
            depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        expected_depth = self.renderer_expected_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "expected_depth": expected_depth,
        }

        if self.config.predict_normals:
            normals = self.renderer_normals(normals=field_outputs[FieldHeadNames.NORMALS], weights=weights)
            pred_normals = self.renderer_normals(field_outputs[FieldHeadNames.PRED_NORMALS], weights=weights)
            outputs["normals"] = self.normals_shader(normals)
            outputs["pred_normals"] = self.normals_shader(pred_normals)
        # These use a lot of GPU memory, so we avoid storing them for eval.
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

        if self.training and self.config.predict_normals:
            outputs["rendered_orientation_loss"] = orientation_loss(
                weights.detach(), field_outputs[FieldHeadNames.NORMALS], ray_bundle.directions
            )

            outputs["rendered_pred_normal_loss"] = pred_normal_loss(
                weights.detach(),
                field_outputs[FieldHeadNames.NORMALS].detach(),
                field_outputs[FieldHeadNames.PRED_NORMALS],
            )

        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])
        return outputs

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = {}
        gt_rgb = batch["image"].to(self.device)  # RGB or RGBA image
        gt_rgb = self.renderer_rgb.blend_background(gt_rgb)  # Blend if RGBA
        predicted_rgb = outputs["rgb"]
        metrics_dict["psnr"] = self.psnr(predicted_rgb, gt_rgb)

        if self.training:
            metrics_dict["distortion"] = distortion_loss(outputs["weights_list"], outputs["ray_samples_list"])

        self.camera_optimizer.get_metrics_dict(metrics_dict)
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = {}
        image = batch["image"].to(self.device)
        pred_rgb, gt_rgb = self.renderer_rgb.blend_background_for_loss_computation(
            pred_image=outputs["rgb"],
            pred_accumulation=outputs["accumulation"],
            gt_image=image,
        )

        loss_dict["rgb_loss"] = self.rgb_loss(gt_rgb, pred_rgb)
        if self.training:
            loss_dict["interlevel_loss"] = self.config.interlevel_loss_mult * interlevel_loss(
                outputs["weights_list"], outputs["ray_samples_list"]
            )
            assert metrics_dict is not None and "distortion" in metrics_dict
            loss_dict["distortion_loss"] = self.config.distortion_loss_mult * metrics_dict["distortion"]
            if self.config.predict_normals:
                # orientation loss for computed normals
                loss_dict["orientation_loss"] = self.config.orientation_loss_mult * torch.mean(
                    outputs["rendered_orientation_loss"]
                )

                # ground truth supervision for normals
                loss_dict["pred_normal_loss"] = self.config.pred_normal_loss_mult * torch.mean(
                    outputs["rendered_pred_normal_loss"]
                )
            # Add loss from camera optimizer
            self.camera_optimizer.get_loss_dict(loss_dict)


            ###############################################################################
            ############################ SDS on test images ##################################
            ###### find spatially annealed cameras ####
            # origins = poses[..., :3, 3]
            # mean_origin = origins.mean(axis=0)
            # poses[..., :3, 3] = origins - mean_origin
            eval_cameras = self.test_cameras.to(self.device)
            # # train_cameras = batch["train_cameras"]
            # origin = torch.zeros(3).cuda()
            # for i in range(len(eval_cameras)):
            #     origin = torch.add(origin, eval_cameras.camera_to_worlds[i][:, 3])
            # # for i in range(len(train_cameras)):
            # #     origin = torch.add(origin, train_cameras.camera_to_worlds[i][:, 3])
            #
            # origin = torch.divide(origin, float(len(eval_cameras)))
            # # origin = torch.divide(origin, float(len(train_cameras) + len(eval_cameras)))
            # # dist=0
            # # for i in range(len(eval_cameras)):    dist += torch.sqrt(torch.sum(torch.pow(torch.subtract(origin, eval_cameras.camera_to_worlds[i][:, 3]), 2), dim=0))
            # # for i in range(len(train_cameras)):    dist += torch.sqrt(torch.sum(torch.pow(torch.subtract(origin, eval_cameras.camera_to_worlds[i][:, 3]), 2), dim=0))
            # # print("Mean radius: ",dist/float(len(train_cameras)+len(eval_cameras)))
            # f1 = eval_cameras.fx[0]
            # f2 = eval_cameras.fy[0]
            image_size = 100
            # rand_cams, vert, hori = randomize_poses(size=1, resolution=int(image_size), device=self.device,
            #                                         radius_mean=0.85, radius_std=0.1,
            #                                         central_rotation_range=(0, 360), vertical_rotation_range=(-90, 0),
            #                                         # (-90, 0),
            #                                         focal_lengths=(f1, f2), jitter_std=0.01,
            #                                         center=(origin[0], origin[1], origin[2]), )
            if self.prompt != self.cur_prompt:
                self.cur_prompt = self.prompt
                self.text_embeddings.update_prompt(
                    base_prompt=self.cur_prompt,
                    top_prompt=self.cur_prompt + self.top_prompt,
                    side_prompt=self.cur_prompt + self.side_prompt,
                    back_prompt=self.cur_prompt + self.back_prompt,
                    front_prompt=self.cur_prompt + self.front_prompt,
                )

            text_embedding = self.text_embeddings.get_text_embedding(
                vertical_angle=torch.zeros(1, 1), horizontal_angle=torch.zeros(1, 1)  # batch["vertical"], horizontal_angle=batch["central"]
            )

            # rand = 1  # random.choice((0,1))
            # if rand == 0:    eval_cameras = rand_cams
            # if rand == 1:    eval_cameras = eval_cameras

            # print(eval_cameras.camera_to_worlds.shape) #,eval_cameras.fx,eval_cameras.cx)
            batch_size = 1
            train_output_batch = torch.zeros((batch_size, 3, image_size, image_size))
            for i1 in range(batch_size):
                camera_idx = np.random.randint(len(eval_cameras))
                # camera_idx = self.iteration_index % len(eval_cameras)
                """
                ########## randomize focal length ######
                rand=random.choice((0.25,0.5)) #0.25,0.5)) #,0.75)) #,1.0)) #1,2,4,8))
                eval_fx=eval_cameras.fx/rand;eval_fy=eval_cameras.fy/rand;
                eval_cameras.fx=eval_fx;eval_cameras.fy=eval_fy;
                """
                ######### translate and rotate #########
                # base_camera_pose=eval_cameras.camera_to_worlds[camera_idx]
                # generated_poses = generate_camera_poses_translation(base_camera_pose, num_poses=5, translation_increment=0.1)
                # rand=random.choice((0,1,2,3))
                # eval_cameras.camera_to_worlds[camera_idx,:,:] = torch.cuda.FloatTensor(generated_poses[rand])
                """
                rotation_matrix=eval_cameras.camera_to_worlds[camera_idx,:3,:3].cpu().detach().numpy()
                angle_degrees = random.randint(-30, 30);angle_radians = angle_degrees * (np.pi / 180)
                R_y = np.array([[np.cos(angle_radians), 0, np.sin(angle_radians)],[0, 1, 0],
                                [-np.sin(angle_radians), 0, np.cos(angle_radians)]])
                R_x = np.array([[np.cos(angle_radians), np.sin(angle_radians), 0],
                                [-np.sin(angle_radians), np.cos(angle_radians), 0],[0, 0, 1]])
                rotation_matrix = np.dot(rotation_matrix, R_y)
                eval_cameras.camera_to_worlds[camera_idx,:3,:3]=torch.cuda.FloatTensor(rotation_matrix)
                """
                ########################################
                camera_ray_bundle = eval_cameras.generate_rays(camera_indices=camera_idx, obb_box=None)
                # print(camera_ray_bundle.origins.shape)
                # outputs = self.get_outputs_for_camera_ray_bundle(camera_ray_bundle)

                outputs = self.get_outputs_for_SDS(camera_ray_bundle)
                # print(outputs["rgb"].shape)
                # from torchvision import transforms
                # from PIL import Image
                # pilImg = transforms.ToPILImage()(outputs["rgb"].permute(2, 0, 1)) #rgb.reshape(64, 64, 3).permute(2, 0, 1)[:,:,:])
                # pilImg.save('output2.jpg')
                train_output = outputs["rgb"].permute(2, 0, 1)
                train_output = train_output[None, :, :, :]
                transform = transforms.RandomCrop((image_size, image_size))  # Specify the desired output size
                params = transform.get_params(train_output, output_size=(image_size, image_size))
                train_output = transforms.functional.crop(train_output, *params)
                train_output_batch[i1:i1 + 1, :, :, :] = train_output
                # print(train_output.shape)
            ################ Uncertainty ##########################
            # """
            # with torch.no_grad():
            #     threshold=3.5
            #     frame2 = ndimage.minimum_filter(outputs["entropy"].cpu().detach().numpy(), size=11) #threshold_with_max_kernel(frame2,10,3)
            #     frame2 = (frame2 < threshold)
            #     frame2=frame2.astype(np.uint8)
            #     #print(frame2.max(),frame2.min())
            #
            #     kernelSizes = [(3, 3), (5, 5), (7, 7), (9, 9), (11,11)]
            #     enlargement_radius = 11  # Adjust this value as needed
            #     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * enlargement_radius + 1, 2 * enlargement_radius + 1))
            #     frame2 = cv2.dilate(frame2, kernel)
            #     for kernelSize in kernelSizes:
            #         kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
            #         frame2 = cv2.morphologyEx(frame2, cv2.MORPH_CLOSE, kernel)
            #     frame4 = cv2.merge([frame2, frame2, frame2])*255;#print(frame2.dtype,frame1.dtype);#frame2=frame2.uint8()
            #     frame3 = cv2.merge([frame2, frame2, frame2])*1.0;
            #     #a=np.array(frame3);print(np.max(a),a.shape)
            #     #mask_image = Image.fromarray(np.array(cv2.resize(frame2, (1024,1024), interpolation = cv2.INTER_AREA)))
            #     mask_image = torch.tensor(np.array(frame3))
            #     from PIL import Image
            #     frame = Image.fromarray(np.array(frame4))
            #     frame.save('output.jpg',"JPEG")
            #     mask_image = transforms.functional.crop(mask_image.permute(2, 0, 1), *params)
            #     mask_image = mask_image[None,:,:,:]
            # """
            ##################################################
            # print("WWWWWWWWW",outputs["rgb"].shape)
            #########################################################################

            # print("BBBB",train_output.shape,mask_image.shape)
            # train_output = transform(train_output)

            # train_output=torch.nn.functional.interpolate(train_output, size=(512, 512), mode='bicubic', align_corners=False)
            # print(train_output.shape)

            # self._diffusion_model.auto_encoder = self._diffusion_model.auto_encoder.to(self.diffusion_device)
            # torch.cuda.empty_cache()

            sds_loss = self._diffusion_model.sds_loss(
                text_embedding.to(self.diffusion_device),
                train_output_batch.to(self.diffusion_device),
                guidance_scale=int(self.guidance_scale),
                grad_scaler=self.grad_scaler,
                # mask_image=mask_image,
                # iteration=self.iteration_index,
            )

            if 1:  # self.train_with_sds==True: #1: #self.train_sds:
                loss_dict["sds_loss"] = 0.0000001 * sds_loss.to(self.device)  ### one after other (1000 it) training
                # print(loss_dict["sds_loss"], self.rgb_loss(gt_rgb, pred_rgb))
            """
            if self.training:
                loss_dict["distortion_loss"] = self.config.distortion_loss_mult * distortion_loss(
                    outputs["weights_list"], outputs["ray_samples_list"]
                )
                loss_dict["interlevel_loss"] = self.config.interlevel_loss_mult * interlevel_loss(
                    outputs["weights_list"], outputs["ray_samples_list"]
                )
            """

        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        gt_rgb = batch["image"].to(self.device)
        predicted_rgb = outputs["rgb"]  # Blended with background (black if random background)
        gt_rgb = self.renderer_rgb.blend_background(gt_rgb)
        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
        )

        combined_rgb = torch.cat([gt_rgb, predicted_rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        gt_rgb = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
        predicted_rgb = torch.moveaxis(predicted_rgb, -1, 0)[None, ...]

        psnr = self.psnr(gt_rgb, predicted_rgb)
        ssim = self.ssim(gt_rgb, predicted_rgb)
        lpips = self.lpips(gt_rgb, predicted_rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        images_dict = {"img": combined_rgb, "accumulation": combined_acc, "depth": combined_depth}

        for i in range(self.config.num_proposal_iterations):
            key = f"prop_depth_{i}"
            prop_depth_i = colormaps.apply_depth_colormap(
                outputs[key],
                accumulation=outputs["accumulation"],
            )
            images_dict[key] = prop_depth_i

        return metrics_dict, images_dict
