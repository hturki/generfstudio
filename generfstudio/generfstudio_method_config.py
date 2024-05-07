"""
Generfstudio data configuration file.
"""
import faulthandler
import signal

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanagerConfig, FullImageDatamanager
from nerfstudio.data.datamanagers.parallel_datamanager import ParallelDataManagerConfig, ParallelDataManager
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.plugins.types import MethodSpecification

from generfstudio.adamw_optimizer_config import AdamWOptimizerConfig
from generfstudio.data.datamanagers.neighboring_views_datamanager import NeighboringViewsDatamanagerConfig, \
    NeighboringViewsDatamanager
from generfstudio.data.datamanagers.union_datamanager import UnionDatamanagerConfig
from generfstudio.data.dataparsers.dtu_dataparser import DTUDataParserConfig
from generfstudio.data.dataparsers.sds_dataparser import SDSDataParserConfig
from generfstudio.data.datasets.masked_dataset import MaskedDataset
from generfstudio.data.datasets.neighboring_views_dataset import NeighboringViewsDataset
from generfstudio.models.mv_diffusion import MVDiffusionConfig
from generfstudio.models.nerfacto_sds import NerfactoSDSModelConfig
from generfstudio.models.pixelnerf import PixelNeRFModelConfig
from generfstudio.models.splatfacto_sds import SplatfactoSDSModelConfig
from generfstudio.zero_redundancy_optimizer_config import ZeroRedundancyOptimizerConfig

faulthandler.register(signal.SIGUSR1)

nerfacto_sds_method = MethodSpecification(
    config=TrainerConfig(
        method_name="nerfacto-sds",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=30000,
        mixed_precision=True,
        pipeline=VanillaPipelineConfig(
            datamanager=ParallelDataManagerConfig(
                _target=ParallelDataManager[MaskedDataset],
                dataparser=SDSDataParserConfig(),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
            ),
            model=NerfactoSDSModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                camera_optimizer=CameraOptimizerConfig(mode="off"),
                disable_scene_contraction=True,
                use_appearance_embedding=False,
                background_color="white"
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=5000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description='Nerfacto with SDS loss',
)

splatfacto_sds_method = MethodSpecification(
    config=TrainerConfig(
        method_name="splatfacto-sds",
        steps_per_eval_image=100,
        steps_per_eval_batch=0,
        steps_per_save=2000,
        steps_per_eval_all_images=1000,
        max_num_iterations=30000,
        # mixed_precision=True,
        mixed_precision=False,
        # gradient_accumulation_steps={"camera_opt": 100},
        pipeline=VanillaPipelineConfig(
            datamanager=FullImageDatamanagerConfig(
                _target=FullImageDatamanager[MaskedDataset],
                dataparser=SDSDataParserConfig(),
                cache_images_type="uint8",
            ),
            model=SplatfactoSDSModelConfig(
                random_init=True,
                background_color="white"
            ),
        ),
        optimizers={
            "xyz": {
                "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1.6e-6,
                    max_steps=30000,
                ),
            },
            "features_dc": {
                "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
                "scheduler": None,
            },
            "features_rest": {
                "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
                "scheduler": None,
            },
            "opacity": {
                "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
                "scheduler": None,
            },
            "scaling": {
                "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
                "scheduler": None,
            },
            "rotation": {"optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15), "scheduler": None},
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description='Splatfacto with SDS loss',
)

depth_gaussians_method = MethodSpecification(
    config=TrainerConfig(
        method_name="depth-gaussians",
        steps_per_eval_image=100,
        steps_per_eval_batch=0,
        steps_per_save=2000,
        steps_per_eval_all_images=1000,
        max_num_iterations=30000,
        # mixed_precision=True,
        mixed_precision=False,
        # gradient_accumulation_steps={"camera_opt": 100},
        pipeline=VanillaPipelineConfig(
            datamanager=FullImageDatamanagerConfig(
                _target=FullImageDatamanager[MaskedDataset],
                dataparser=SDSDataParserConfig(),
                cache_images_type="uint8",
            ),
            model=SplatfactoSDSModelConfig(
                random_init=True,
                background_color="white"
            ),
        ),
        optimizers={
            "xyz": {
                "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1.6e-6,
                    max_steps=30000,
                ),
            },
            "features_dc": {
                "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
                "scheduler": None,
            },
            "features_rest": {
                "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
                "scheduler": None,
            },
            "opacity": {
                "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
                "scheduler": None,
            },
            "scaling": {
                "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
                "scheduler": None,
            },
            "rotation": {"optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15), "scheduler": None},
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description='Depth Gaussians',
)

pixelnerf_method = MethodSpecification(
    config=TrainerConfig(
        method_name="pixel-nerf",
        steps_per_eval_image=1000,
        steps_per_eval_batch=0,
        steps_per_save=2000,
        steps_per_eval_all_images=1000000,
        max_num_iterations=1000001,
        mixed_precision=True,
        pipeline=VanillaPipelineConfig(
            datamanager=NeighboringViewsDatamanagerConfig(
                _target=NeighboringViewsDatamanager[NeighboringViewsDataset],
                dataparser=DTUDataParserConfig(scene_id=None, auto_orient=False, crop=None),
                rays_per_image=128,
            ),
            model=PixelNeRFModelConfig(),
        ),
        optimizers={
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-15),
                "scheduler": None,
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description='PixelNeRF',
)

mv_diffusion_method = MethodSpecification(
    config=TrainerConfig(
        method_name="mv-diffusion",
        steps_per_eval_image=10,
        steps_per_eval_batch=0,
        steps_per_save=1000,
        steps_per_eval_all_images=1000000,
        max_num_iterations=101,
        mixed_precision=True,
        log_gradients=False,
        gradient_accumulation_steps={
            "cond_encoder": 4,
            "fields": 4
        },
        pipeline=VanillaPipelineConfig(
            datamanager=NeighboringViewsDatamanagerConfig(
                _target=NeighboringViewsDatamanager[NeighboringViewsDataset],
                neighboring_views_size=3,
                image_batch_size=64,
                dataparser=DTUDataParserConfig(scene_id=None, auto_orient=True),
            ),
            model=MVDiffusionConfig(),
        ),
        optimizers={
            "cond_encoder": {
                "optimizer": AdamWOptimizerConfig(lr=1e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_pre_warmup=1e-10, warmup_steps=100),
            },
            "fields": {
                "optimizer": AdamWOptimizerConfig(lr=1e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_pre_warmup=1e-10, warmup_steps=100),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description='Training a multi-view diffusion model',
)

mv_diffusion_union_method = MethodSpecification(
    config=TrainerConfig(
        method_name="mv-diffusion-union",
        steps_per_eval_image=1000,
        steps_per_eval_batch=0,
        steps_per_save=2000,
        steps_per_eval_all_images=1000000,
        max_num_iterations=1000001,
        mixed_precision=True,
        log_gradients=False,
        gradient_accumulation_steps={
            "cond_encoder": 64,
            "fields": 64
        },
        pipeline=VanillaPipelineConfig(
            datamanager=UnionDatamanagerConfig(
                inner=NeighboringViewsDatamanagerConfig(
                    _target=NeighboringViewsDatamanager[NeighboringViewsDataset],
                    neighboring_views_size=3,
                    image_batch_size=4,
                ),
            ),
            model=MVDiffusionConfig(),
        ),
        optimizers={
            "cond_encoder": {
                "optimizer": AdamWOptimizerConfig(lr=1e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_pre_warmup=1e-10, warmup_steps=100),
            },
            "fields": {
                "optimizer": AdamWOptimizerConfig(lr=1e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_pre_warmup=1e-10, warmup_steps=100),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description='Training a multi-view diffusion model on all datasets',
)

mv_diffusion_ddp_method = MethodSpecification(
    config=TrainerConfig(
        method_name="mv-diffusion-ddp",
        steps_per_eval_image=1600,
        steps_per_eval_batch=0,
        steps_per_save=2000,
        steps_per_eval_all_images=1000000,
        max_num_iterations=1000001,
        mixed_precision=True,
        log_gradients=False,
        gradient_accumulation_steps={
            "cond_encoder": 8,
            "fields": 8
        },
        pipeline=VanillaPipelineConfig(
            datamanager=NeighboringViewsDatamanagerConfig(
                _target=NeighboringViewsDatamanager[NeighboringViewsDataset],
                neighboring_views_size=3,
                image_batch_size=32,
                dataparser=DTUDataParserConfig(scene_id=None, auto_orient=True),
            ),
            model=MVDiffusionConfig(),
        ),
        optimizers={
            "cond_encoder": {
                "optimizer": ZeroRedundancyOptimizerConfig(lr=1e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_pre_warmup=1e-10, warmup_steps=100),
            },
            "fields": {
                "optimizer": ZeroRedundancyOptimizerConfig(lr=1e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_pre_warmup=1e-10, warmup_steps=100),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description='Training a multi-view diffusion model with DDP (8 GPUs)',
)

mv_diffusion_union_ddp_method = MethodSpecification(
    config=TrainerConfig(
        method_name="mv-diffusion-union-ddp",
        steps_per_eval_image=1600,
        steps_per_eval_batch=0,
        steps_per_save=2000,
        steps_per_eval_all_images=1000000,
        max_num_iterations=1000001,
        mixed_precision=True,
        log_gradients=False,
        gradient_accumulation_steps={
            "cond_encoder": 8,
            "fields": 8
        },
        pipeline=VanillaPipelineConfig(
            datamanager=UnionDatamanagerConfig(
                inner=NeighboringViewsDatamanagerConfig(
                    _target=NeighboringViewsDatamanager[NeighboringViewsDataset],
                    neighboring_views_size=3,
                    image_batch_size=32,
                ),
            ),
            model=MVDiffusionConfig(),
        ),
        optimizers={
            "cond_encoder": {
                "optimizer": ZeroRedundancyOptimizerConfig(lr=1e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_pre_warmup=1e-10, warmup_steps=100),
            },
            "fields": {
                "optimizer": ZeroRedundancyOptimizerConfig(lr=1e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_pre_warmup=1e-10, warmup_steps=100),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description='Training a multi-view diffusion model on all datasets with DDP (8 GPUs)',
)
