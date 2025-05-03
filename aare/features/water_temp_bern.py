import pandas as pd
from darts import TimeSeries

from aare.features.feature import Feature
from aare.preparation import remove_outliers, interpolate
from aare.remote_existenz_store import FieldRequest
from aare.utils import to_ts


class WaterTempBern(Feature):
    # TODO before doing more features, create some base class(es) to simplify
    NAME = "temp_bern"
    FIELD = FieldRequest.from_str("hydro/temperature:mean_1h@bern")

    def __init__(self):
        super().__init__(self.NAME, self.FIELD)

    def cleanup(self, df: pd.DataFrame) -> pd.DataFrame:
        # TODO move these functions into here, or at least somewhere else, because they are
        # purely for the (bern) water temperature
        df = remove_outliers(df, self.FIELD.name)
        df = interpolate(df, drop_filled=True, columns=self.FIELD.name)

        return df

    def transform(self, df: pd.DataFrame) -> TimeSeries:
        return to_ts(df, col=self.FIELD.name).with_columns_renamed([self.FIELD.name], [self.NAME])
