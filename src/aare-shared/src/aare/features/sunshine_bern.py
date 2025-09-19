import pandas as pd

from aare.features.base.single_field_feature import SingleFieldFeature
from aare.remote_existenz_store import FieldRequest
from aare.utils import fill_with_hard_limit


class SunshineBern(SingleFieldFeature):
    NAME = "ss_bern"
    FIELD = FieldRequest.from_str("smn/ss:sum_1h@bern")

    def __init__(self):
        super().__init__(self.NAME, self.FIELD)

    def cleanup(self, df: pd.DataFrame) -> pd.DataFrame:
        limit_key = "median_gap_bound"
        if self.params is None or (limit := self.params.get("custom", {}).get(limit_key)) is None:
            raise ValueError(f"'{limit_key}' not set correctly in '{self.name}' feature config")

        df_i = fill_with_hard_limit(
            df,
            limit=limit,
            fill_method="median",
            columns=[self.field.name],
        )
        df[self.field.name] = df_i[self.field.name]

        return df
