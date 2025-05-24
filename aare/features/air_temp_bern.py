import pandas as pd

from aare.features.base.single_field_feature import SingleFieldFeature
from aare.preparation import interpolate_aare_temp
from aare.remote_existenz_store import FieldRequest


class AirTempBern(SingleFieldFeature):
    NAME = "tt_bern"
    FIELD = FieldRequest.from_str("smn/tt:mean_1h@bern")

    def __init__(self):
        super().__init__(self.NAME, self.FIELD)

    def cleanup(self, df: pd.DataFrame) -> pd.DataFrame:
        # TODO currently using the same settings as the water temp imputation (!)
        df = interpolate_aare_temp(df, drop_filled=True, columns=self.field.name)

        return df
