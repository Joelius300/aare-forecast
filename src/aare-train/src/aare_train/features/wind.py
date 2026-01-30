from typing import override
from aare_train.features.base.feature import Feature
from aare_train.preparation import interpolate_continuous, remove_outliers, remove_period
from aare_train.fetching.remote_existenz_store import FieldRequest
from aare_train.darts_utils import to_ts
from darts import TimeSeries
import numpy as np
from pandas import DataFrame


class Wind(Feature):
    NAME = "wind"

    def __init__(self, loc: str):
        self.dir_field = FieldRequest("smn", "dd", "1h", "mean", loc)
        self.mag_field = FieldRequest("smn", "ff", "1h", "mean", loc)
        super().__init__(self.loc_name(loc), [self.dir_field, self.mag_field])
        self.loc = loc

    @override
    def cleanup(self, df: DataFrame) -> DataFrame:
        df = remove_outliers(df, 0, 360, col=self.dir_field.name)
        df = remove_outliers(df, 0, 99999, col=self.mag_field.name)

        # really weird constant dd_bern before a gap in January 2009
        remove_period(df, "01.01.2009T12:00:00Z", "06.01.2009T00:00:00Z", self.dir_field.name)

        df = interpolate_continuous(
            df,
            linear_gap_bound=5,
            cubic_gap_bound=10,
            median_gap_bound=None,
            drop_filled=True,
            columns=self.mag_field.name,
        )

        # since direction is much harder to interpolate, we only do 3hours linearly.
        # then the interpolation of the magnitude is much less relevant because we need both anyway.
        df = interpolate_continuous(
            df,
            linear_gap_bound=3,
            cubic_gap_bound=None,
            median_gap_bound=None,
            drop_filled=True,
            columns=self.dir_field.name,
        )

        return df

    @override
    def transform(self, df: DataFrame) -> TimeSeries:
        dir = df[self.dir_field.name]
        mag = df[self.mag_field.name]

        y_name = f"wind_y_{self.loc}"
        x_name = f"wind_x_{self.loc}"
        df[y_name] = np.sin(dir) * mag
        df[x_name] = np.cos(dir) * mag

        return to_ts(df, [x_name, y_name])
