from typing import Dict

from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.datasets.base_dataset import InputDataset

from generfstudio.generfstudio_constants import NEIGHBORING_VIEW_INDICES


class NeighboringViewsDataset(InputDataset):
    def __init__(self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1.0):
        super().__init__(dataparser_outputs, scale_factor)
        self.neighboring_views = self.metadata[NEIGHBORING_VIEW_INDICES]

    def get_metadata(self, data: Dict) -> Dict:
        metadata = super().get_metadata(data)

        metadata[NEIGHBORING_VIEW_INDICES] = self.neighboring_views[data["image_idx"]]

        # metadata[NEIGHBORING_VIEW_INDICES] = torch.LongTensor(self.neighboring_views[data["image_idx"]])

        return metadata
