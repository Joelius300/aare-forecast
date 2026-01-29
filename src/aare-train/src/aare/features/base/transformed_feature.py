from typing import Callable, cast

import pandas as pd
from darts import TimeSeries
from darts.dataprocessing.transformers import Mapper

from aare.features.base.feature import Feature

MapperFuncType = Mapper | Callable[[TimeSeries], TimeSeries]


class TransformedFeature(Feature):
    """Transformed version of existing feature, e.g. for squaring."""

    @classmethod
    def base_name(cls) -> str:
        return "TRANSFORMED"

    def __init__(self, base_feature: Feature, suffix: str, transformer: MapperFuncType):
        super().__init__(base_feature.name + suffix, base_feature.required_fields)
        self.base_feature = base_feature
        self._transformer = transformer
        self._suffix = suffix

    def _apply_transformation(self, ts: TimeSeries) -> TimeSeries:
        if isinstance(self._transformer, Mapper):
            return cast(TimeSeries, self._transformer.transform(ts))

        return self._transformer(ts)

    def cleanup(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.base_feature.cleanup(df)

    def transform(self, df: pd.DataFrame) -> TimeSeries:
        ts = self.base_feature.transform(df)
        transformed = self._apply_transformation(ts)

        comp = transformed.components
        renamed = transformed.with_columns_renamed(comp.tolist(), (comp + self._suffix).tolist())

        return renamed
