from aare.features.base.single_field_feature import SingleFieldFeature
from aare.remote_existenz_store import FieldRequest


class SunshineBern(SingleFieldFeature):
    NAME = "ss_bern"
    FIELD = FieldRequest.from_str("smn/ss:sum_1h@bern")

    def __init__(self):
        super().__init__(self.NAME, self.FIELD)
