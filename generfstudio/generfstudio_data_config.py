"""
Generfstudio dataparsers configuration file.
"""
from nerfstudio.plugins.registry_dataparser import DataParserSpecification

from generfstudio.data.dataparsers.dtu_dataparser import DTUDataParserConfig
from generfstudio.data.dataparsers.sds_dataparser import SDSDataParserConfig

sds_dataparser = DataParserSpecification(config=SDSDataParserConfig())
dtu_dataparser = DataParserSpecification(config=DTUDataParserConfig())