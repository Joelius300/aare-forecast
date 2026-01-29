from aare.features.base.single_field_feature import SingleFieldFeature
from aare.fetching.remote_existenz_store import FieldRequest


class Rainfall(SingleFieldFeature):
    NAME = "rr"

    def __init__(self, loc: str | int):
        super().__init__(self.loc_name(loc), FieldRequest("smn", "rr", "1h", "sum", loc))
