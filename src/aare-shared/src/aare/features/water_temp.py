from typing import override
from aare.features.base.single_field_feature import SingleFieldFeature
from aare.remote_existenz_store import FieldRequest


class WaterTemp(SingleFieldFeature):
    NAME = "temp"

    def __init__(self, loc: str | int):
        super().__init__(self.loc_name(loc), FieldRequest("hydro", "temperature", "1h", "mean", loc))
