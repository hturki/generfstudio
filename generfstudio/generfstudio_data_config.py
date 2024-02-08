"""
Generfstudio dataparsers configuration file.
"""
from nerfstudio.plugins.registry_dataparser import DataParserSpecification

from generfstudio.data.dataparsers.sds_dataparser import SDSDataParserConfig

sds_dataparser = DataParserSpecification(config=SDSDataParserConfig())