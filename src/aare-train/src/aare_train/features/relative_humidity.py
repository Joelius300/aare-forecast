from aare_train.features.base.single_field_feature import SingleFieldFeature
from aare_influx.field_request import FieldRequest


class RelativeHumidity(SingleFieldFeature):
    NAME = "rh"

    def __init__(self, loc: str | int):
        super().__init__(self.loc_name(loc), FieldRequest("smn", "rh", "1h", "mean", loc))
