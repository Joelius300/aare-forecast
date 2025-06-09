import pandas as pd

from aare.features.base.single_field_feature import SingleFieldFeature
from aare.preparation import remove_period, remove_outliers, interpolate_continuous
from aare.remote_existenz_store import FieldRequest


class FlowBern(SingleFieldFeature):
    NAME = "flow_bern"
    FIELD = FieldRequest.from_str("hydro/flow:mean_1h@bern")

    def __init__(self):
        super().__init__(self.NAME, self.FIELD)

    def _remove_faulty_periods(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # seems to be the only part where the cleanup fails and interpolation makes everything worse
        remove_period(df, "23.05.2015T12:00:00Z", "26.05.2015T16:00:00Z", self.field.name)

        # these are very picky, I just saw them when fixing the big one above, but I can't fix thing like this
        #  during inference, so I'll leave it as is.
        # remove_period(df, "29.06.2015T00:00:00Z", "29.06.2015T05:00:00Z", self.field.name)
        # remove_period(df, "29.06.2015T07:00:00Z", "29.06.2015T11:00:00Z", self.field.name)
        # these don't get removed?!?
        # remove_period(df, "01.07.2015T10:00:00Z", "02.07.2015T00:00:00Z", self.field.name)
        # remove_period(df, "02.07.2015T07:00:00Z", "02.07.2015T11:00:00Z", self.field.name)
        # On 08.05.2016 there's another fuckup

        return df

    def cleanup(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._remove_faulty_periods(df)
        # TODO move these into params.yaml
        df = remove_outliers(df, 30, 99999, 14.5, self.field.name)
        df = interpolate_continuous(df, 3, 23, drop_filled=True, columns=self.field.name)

        return df
