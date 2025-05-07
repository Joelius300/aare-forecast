import pandas as pd

from aare.features.base.single_field_feature import SingleFieldFeature
from aare.preparation import remove_outliers, interpolate
from aare.remote_existenz_store import FieldRequest


class WaterTempBern(SingleFieldFeature):
    NAME = "temp_bern"
    FIELD = FieldRequest.from_str("hydro/temperature:mean_1h@bern")

    def __init__(self):
        super().__init__(self.NAME, self.FIELD)

    def cleanup(self, df: pd.DataFrame) -> pd.DataFrame:
        # TODO move these functions into here, or at least somewhere else, because they are
        # purely for the (bern) water temperature
        df = remove_outliers(df, self.field.name)
        df = interpolate(df, drop_filled=True, columns=self.field.name)

        return df
