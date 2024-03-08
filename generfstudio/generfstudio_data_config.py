"""
Generfstudio dataparsers configuration file.
"""
from nerfstudio.plugins.registry_dataparser import DataParserSpecification

from generfstudio.data.dataparsers.dtu_dataparser import DTUDataParserConfig
from generfstudio.data.dataparsers.r10k_dataparser import R10KDataParserConfig
from generfstudio.data.dataparsers.sds_dataparser import SDSDataParserConfig

dtu_dataparser = DataParserSpecification(config=DTUDataParserConfig())
r10k_dataparser = DataParserSpecification(config=R10KDataParserConfig())
sds_dataparser = DataParserSpecification(config=SDSDataParserConfig())
