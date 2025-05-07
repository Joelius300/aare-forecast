import pandas as pd
from darts import TimeSeries

from aare.features.base.feature import Feature
from aare.remote_existenz_store import FieldRequest
from aare.utils import to_ts


class SingleFieldFeature(Feature):
    def __init__(self, name: str, field: FieldRequest):
        super().__init__(name, field)

    @property
    def field(self) -> FieldRequest:
        return self.required_fields[0]

    def transform(self, df: pd.DataFrame) -> TimeSeries:
        return to_ts(df, col=self.field.name).with_columns_renamed([self.field.name], [self.name])
