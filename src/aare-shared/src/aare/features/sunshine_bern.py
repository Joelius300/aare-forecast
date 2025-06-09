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
        df_i = fill_with_hard_limit(
            df,
            limit=10,  # todo parametrize in params.yaml
            fill_method="median",
            columns=[self.field.name],
        )
        df[self.field.name] = df_i[self.field.name]

        return df
