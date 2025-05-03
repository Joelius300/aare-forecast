import pandas as pd
from darts import TimeSeries

from aare.features.feature import Feature
from aare.preparation import interpolate
from aare.remote_existenz_store import FieldRequest
from aare.utils import to_ts


class AirTempBern(Feature):
    NAME = "tt_bern"
    FIELD = FieldRequest.from_str("smn/tt:mean_1h@bern")

    def __init__(self):
        super().__init__(self.NAME, self.FIELD)

    def cleanup(self, df: pd.DataFrame) -> pd.DataFrame:
        # TODO currently using the same settings as the water temp imputation (!)
        df = interpolate(df, drop_filled=True, columns=self.FIELD.name)

        return df

    def transform(self, df: pd.DataFrame) -> TimeSeries:
        return to_ts(df, col=self.FIELD.name).with_columns_renamed([self.FIELD.name], [self.NAME])
