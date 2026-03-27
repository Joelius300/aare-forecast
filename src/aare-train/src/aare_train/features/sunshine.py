from aare_train.features.base.single_field_feature import SingleFieldFeature
from aare_influx.field_request import FieldRequest


class Sunshine(SingleFieldFeature):
    NAME = "ss"

    def __init__(self, loc: str | int):
        super().__init__(self.loc_name(loc), FieldRequest("smn", "ss", "1h", "sum", loc))
