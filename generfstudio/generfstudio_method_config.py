"""
Generfstudio data configuration file.
"""
from dataclasses import field

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanagerConfig, FullImageDatamanager
from nerfstudio.data.datamanagers.parallel_datamanager import ParallelDataManagerConfig, ParallelDataManager
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.plugins.types import MethodSpecification

from generfstudio.data.dataparsers.sds_dataparser import SDSDataParserConfig
from generfstudio.data.datasets.masked_dataset import MaskedDataset
from generfstudio.models.nerfacto_sds import NerfactoSDSModelConfig
from generfstudio.models.splatfacto_sds import SplatfactoSDSModelConfig

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
            # "camera_opt": {
            #     "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
            #     "scheduler": ExponentialDecaySchedulerConfig(lr_final=5e-5, max_steps=30000),
            # },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description='Splatfacto with SDS loss',
)
