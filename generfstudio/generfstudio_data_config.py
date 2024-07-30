"""
Generfstudio dataparsers configuration file.
"""
from nerfstudio.plugins.registry_dataparser import DataParserSpecification

from generfstudio.data.dataparsers.co3d_dataparser import CO3DDataParserConfig
from generfstudio.data.dataparsers.depth_providing_dataparser import DepthProvidingDataParserConfig
from generfstudio.data.dataparsers.dl3dv_dataparser import DL3DVDataParserConfig
from generfstudio.data.dataparsers.dtu_dataparser import DTUDataParserConfig
from generfstudio.data.dataparsers.mipnerf_dataparser import MipNerf360DataParserConfig
from generfstudio.data.dataparsers.mvimgnet_dataparser import MVImgNetDataParserConfig
from generfstudio.data.dataparsers.objaverse_xl_dataparser import ObjaverseXLDataParserConfig
from generfstudio.data.dataparsers.r10k_dataparser import R10KDataParserConfig
from generfstudio.data.dataparsers.sds_dataparser import SDSDataParserConfig

co3d_dataparser = DataParserSpecification(config=CO3DDataParserConfig())
depth_providing_dataparser = DataParserSpecification(config=DepthProvidingDataParserConfig())
dl3dv_dataparser = DataParserSpecification(config=DL3DVDataParserConfig())
dtu_dataparser = DataParserSpecification(config=DTUDataParserConfig())
mipnerf_dataparser = DataParserSpecification(config=MipNerf360DataParserConfig())
mvimgnet_dataparser = DataParserSpecification(config=MVImgNetDataParserConfig())
oxl_dataparser = DataParserSpecification(config=ObjaverseXLDataParserConfig())
r10k_dataparser = DataParserSpecification(config=R10KDataParserConfig())
