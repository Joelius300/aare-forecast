from aare_train.features.base.single_field_feature import SingleFieldFeature
from aare_influx.field_request import FieldRequest


class WaterTemp(SingleFieldFeature):
    NAME = "temp"

    def __init__(self, loc: str | int):
        super().__init__(self.loc_name(loc), FieldRequest("hydro", "temperature", "1h", "first", loc))
