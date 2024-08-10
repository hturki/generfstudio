"""
Generfstudio data configuration file.
"""
import faulthandler
import signal

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.plugins.types import MethodSpecification

from generfstudio.adamw_optimizer_config import AdamWOptimizerConfig
from generfstudio.data.datamanagers.neighboring_views_datamanager import NeighboringViewsDatamanagerConfig, \
    NeighboringViewsDatamanager
from generfstudio.data.datamanagers.union_datamanager import UnionDatamanagerConfig
from generfstudio.data.datamanagers.updating_full_images_datamanager import UpdatingFullImageDatamanagerConfig
from generfstudio.data.dataparsers.depth_providing_dataparser import DepthProvidingDataParserConfig
from generfstudio.data.dataparsers.dl3dv_dataparser import DL3DVDataParserConfig
from generfstudio.data.datasets.neighboring_views_dataset import NeighboringViewsDataset
from generfstudio.models.depth_gaussians import DepthGaussiansModelConfig
from generfstudio.models.rgbd_diffusion import RGBDDiffusionConfig
from generfstudio.models.rgbd_diffusion_if import RGBDDiffusionIFConfig
from generfstudio.zero_redundancy_optimizer_config import ZeroRedundancyOptimizerConfig

faulthandler.register(signal.SIGUSR1)

depth_gaussians_method = MethodSpecification(
    config=TrainerConfig(
        method_name="depth-gaussians",
        steps_per_eval_image=1000,
        steps_per_eval_batch=0,
        steps_per_save=2000,
        steps_per_eval_all_images=1000,
        max_num_iterations=30000,
        mixed_precision=False,
        pipeline=VanillaPipelineConfig(
            datamanager=UpdatingFullImageDatamanagerConfig(
                dataparser=DepthProvidingDataParserConfig(),
                cache_images_type="uint8",
            ),
            model=DepthGaussiansModelConfig(),
        ),
        optimizers={
            "means": {
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
            "opacities": {
                "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
                "scheduler": None,
            },
            "scales": {
                "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
                "scheduler": None,
            },
            "quats": {"optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15), "scheduler": None},
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=5e-7, max_steps=30000, warmup_steps=1000, lr_pre_warmup=0
                ),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description='Depth Gaussians',
)

rgbd_diffusion_method = MethodSpecification(
    config=TrainerConfig(
        method_name="rgbd-diffusion",
        steps_per_eval_image=1000,
        steps_per_eval_batch=0,
        steps_per_save=1000,
        steps_per_eval_all_images=999000000,
        max_num_iterations=1000001,
        mixed_precision=True,
        log_gradients=False,
        gradient_accumulation_steps={
            "cond_encoder": 4,
            "fields": 4
        },
        pipeline=VanillaPipelineConfig(
            datamanager=NeighboringViewsDatamanagerConfig(
                _target=NeighboringViewsDatamanager[NeighboringViewsDataset],
                views_size_train=1,
                batch_size=64,
                dataparser=DL3DVDataParserConfig(),
            ),
            model=RGBDDiffusionConfig(),
        ),
        optimizers={
            "fields": {
                "optimizer": AdamWOptimizerConfig(lr=1e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_pre_warmup=1e-10, warmup_steps=100),
            },
        },
        vis="wandb",
    ),
    description='Training a multi-view RGBD diffusion model',
)

rgbd_diffusion_ddp_method = MethodSpecification(
    config=TrainerConfig(
        method_name="rgbd-diffusion-ddp",
        steps_per_eval_image=1000,
        steps_per_eval_batch=0,
        steps_per_save=1000,
        steps_per_eval_all_images=999000000,
        max_num_iterations=1000001,
        mixed_precision=True,
        log_gradients=False,
        gradient_accumulation_steps={
            "fields": 1
        },
        pipeline=VanillaPipelineConfig(
            datamanager=NeighboringViewsDatamanagerConfig(
                _target=NeighboringViewsDatamanager[NeighboringViewsDataset],
                views_size_train=1,
                batch_size=256,
                dataparser=DL3DVDataParserConfig(),
            ),
            model=RGBDDiffusionConfig(),
        ),
        optimizers={
            "fields": {
                "optimizer": ZeroRedundancyOptimizerConfig(lr=1e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_pre_warmup=1e-10, warmup_steps=100),
            },
        },
        vis="wandb",
    ),
    description='Training a multi-view RGBD diffusion model',
)

rgbd_diffusion_union_method = MethodSpecification(
    config=TrainerConfig(
        method_name="rgbd-diffusion-union",
        steps_per_eval_image=1,
        steps_per_eval_batch=0,
        steps_per_save=1000,
        steps_per_eval_all_images=999000000,
        max_num_iterations=1000001,
        mixed_precision=True,
        log_gradients=False,
        gradient_accumulation_steps={
            "fields": 4
        },
        pipeline=VanillaPipelineConfig(
            datamanager=UnionDatamanagerConfig(
                inner=NeighboringViewsDatamanagerConfig(
                    _target=NeighboringViewsDatamanager[NeighboringViewsDataset],
                    views_size_train=1,
                    batch_size=64,
                ),
            ),
            model=RGBDDiffusionConfig(),
        ),
        optimizers={
            "fields": {
                "optimizer": AdamWOptimizerConfig(lr=1e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_pre_warmup=1e-10, warmup_steps=100),
            },
        },
        vis="wandb",
    ),
    description='Training a multi-view RGBD diffusion model',
)

rgbd_diffusion_union_ddp_method = MethodSpecification(
    config=TrainerConfig(
        method_name="rgbd-diffusion-union-ddp",
        steps_per_eval_image=1000,
        steps_per_eval_batch=0,
        steps_per_save=1000,
        steps_per_eval_all_images=999000000,
        max_num_iterations=1000001,
        mixed_precision=True,
        log_gradients=False,
        gradient_accumulation_steps={
            "fields": 1
        },
        pipeline=VanillaPipelineConfig(
            datamanager=UnionDatamanagerConfig(
                inner=NeighboringViewsDatamanagerConfig(
                    _target=NeighboringViewsDatamanager[NeighboringViewsDataset],
                    views_size_train=1,
                    batch_size=256,
                ),
            ),
            model=RGBDDiffusionConfig(),
        ),
        optimizers={
            "fields": {
                "optimizer": ZeroRedundancyOptimizerConfig(lr=1e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_pre_warmup=1e-10, warmup_steps=100),
            },
        },
        vis="wandb",
    ),
    description='Training a multi-view RGBD diffusion model',
)

rgbd_diffusion_if_method = MethodSpecification(
    config=TrainerConfig(
        method_name="rgbd-diffusion-if",
        steps_per_eval_image=1000,
        steps_per_eval_batch=0,
        steps_per_save=1000,
        steps_per_eval_all_images=999000000,
        max_num_iterations=1000001,
        mixed_precision=True,
        log_gradients=False,
        gradient_accumulation_steps={
            "fields": 16
        },
        pipeline=VanillaPipelineConfig(
            datamanager=NeighboringViewsDatamanagerConfig(
                _target=NeighboringViewsDatamanager[NeighboringViewsDataset],
                views_size_train=1,
                batch_size=16,
                dataparser=DL3DVDataParserConfig(),
            ),
            model=RGBDDiffusionIFConfig(),
        ),
        optimizers={
            "fields": {
                "optimizer": AdamWOptimizerConfig(lr=1e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_pre_warmup=1e-10, warmup_steps=100),
            },
        },
        vis="wandb",
    ),
    description='Training a multi-view RGBD diffusion model',
)

rgbd_diffusion_if_union_method = MethodSpecification(
    config=TrainerConfig(
        method_name="rgbd-diffusion-if-union",
        steps_per_eval_image=1000,
        steps_per_eval_batch=0,
        steps_per_save=1000,
        steps_per_eval_all_images=999000000,
        max_num_iterations=1000001,
        mixed_precision=True,
        log_gradients=False,
        gradient_accumulation_steps={
            "fields": 16
        },
        pipeline=VanillaPipelineConfig(
            datamanager=UnionDatamanagerConfig(
                inner=NeighboringViewsDatamanagerConfig(
                    _target=NeighboringViewsDatamanager[NeighboringViewsDataset],
                    views_size_train=1,
                    batch_size=16,
                ),
            ),
            model=RGBDDiffusionIFConfig(),
        ),
        optimizers={
            "fields": {
                "optimizer": AdamWOptimizerConfig(lr=1e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_pre_warmup=1e-10, warmup_steps=100),
            },
        },
        vis="wandb",
    ),
    description='Training a multi-view RGBD diffusion model',
)

rgbd_diffusion_if_union_ddp_method = MethodSpecification(
    config=TrainerConfig(
        method_name="rgbd-diffusion-if-union-ddp",
        steps_per_eval_image=1000,
        steps_per_eval_batch=0,
        steps_per_save=1000,
        steps_per_eval_all_images=999000000,
        max_num_iterations=1000001,
        mixed_precision=True,
        log_gradients=False,
        gradient_accumulation_steps={
            "fields": 4
        },
        pipeline=VanillaPipelineConfig(
            datamanager=UnionDatamanagerConfig(
                inner=NeighboringViewsDatamanagerConfig(
                    _target=NeighboringViewsDatamanager[NeighboringViewsDataset],
                    views_size_train=1,
                    batch_size=64,
                ),
            ),
            model=RGBDDiffusionIFConfig(),
        ),
        optimizers={
            "fields": {
                "optimizer": ZeroRedundancyOptimizerConfig(lr=1e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_pre_warmup=1e-10, warmup_steps=100),
            },
        },
        vis="wandb",
    ),
    description='Training a multi-view RGBD diffusion model',
)
