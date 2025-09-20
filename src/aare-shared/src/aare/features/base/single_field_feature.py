import logging

import pandas as pd
from darts import TimeSeries

from aare.features.base.feature import Feature
from aare.preparation import remove_outliers, interpolate_continuous
from aare.remote_existenz_store import FieldRequest
from aare.utils import to_ts

logger = logging.getLogger(__name__)


class SingleFieldFeature(Feature):
    def __init__(self, name: str, field: FieldRequest):
        super().__init__(name, field)

    @property
    def field(self) -> FieldRequest:
        return self.required_fields[0]

    def transform(self, df: pd.DataFrame) -> TimeSeries:
        return to_ts(df, col=self.field.name).with_columns_renamed([self.field.name], [self.name])

    def remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers according to the 'outliers' config if it's set."""
        if self.params is None or "outliers" not in self.params or (params := self.params["outliers"]) is None:
            logger.debug(f"Not removing outliers on '{self.name}' because 'outliers' config is not set.")
            return df

        return remove_outliers(
            df, params["low_cutoff"], params["high_cutoff"], params["diff_threshold"], col=self.field.name
        )

    def interpolate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Interpolate according to the 'interpolate' config if it's set."""
        if self.params is None or "interpolate" not in self.params or (params := self.params["interpolate"]) is None:
            logger.debug(f"Not interpolating '{self.name}' because 'interpolate' config is not set.")
            return df

        return interpolate_continuous(
            df, params["linear_gap_bound"], params["cubic_gap_bound"], drop_filled=True, columns=self.field.name
        )

    def cleanup(self, df: pd.DataFrame) -> pd.DataFrame:
        """Default implementation removes outliers and interpolates according to feature config."""
        df = self.remove_outliers(df)
        df = self.interpolate(df)

        return df
