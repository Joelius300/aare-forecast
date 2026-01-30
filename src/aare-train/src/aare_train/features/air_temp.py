from aare_train.features.base.single_field_feature import SingleFieldFeature
from aare_train.fetching.remote_existenz_store import FieldRequest


class AirTemp(SingleFieldFeature):
    NAME = "tt"

    def __init__(self, loc: str | int):
        super().__init__(self.loc_name(loc), FieldRequest("smn", "tt", "1h", "mean", loc))
