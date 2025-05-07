from typing import Callable, cast

import pandas as pd
from darts import TimeSeries
from darts.dataprocessing.transformers import Mapper

from aare.features.base.feature import Feature


class TransformedFeature(Feature):
    """Transformed version of existing feature, e.g. for squaring."""

    def __init__(self, base_feature: Feature, suffix: str, transformer: Mapper | Callable[[TimeSeries], TimeSeries]):
        super().__init__(base_feature.name + suffix, base_feature.required_fields)
        self.base_feature = base_feature
        self._transformer = transformer

    def _apply_transformation(self, ts: TimeSeries) -> TimeSeries:
        if isinstance(self._transformer, Mapper):
            return cast(TimeSeries, self._transformer.transform(ts))

        return self._transformer(ts)

    def cleanup(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.base_feature.cleanup(df)

    def transform(self, df: pd.DataFrame) -> TimeSeries:
        ts = self.base_feature.transform(df)
        transformed = self._apply_transformation(ts)

        return transformed
