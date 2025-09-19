from aare.features.base.single_field_feature import SingleFieldFeature
from aare.remote_existenz_store import FieldRequest


class WaterTempBern(SingleFieldFeature):
    NAME = "temp_bern"
    FIELD = FieldRequest.from_str("hydro/temperature:mean_1h@bern")

    def __init__(self):
        super().__init__(self.NAME, self.FIELD)
