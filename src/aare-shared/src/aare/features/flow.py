from aare.constants import LOC_BERN
import pandas as pd

from aare.features.base.single_field_feature import SingleFieldFeature
from aare.preparation import remove_period
from aare.remote_existenz_store import FieldRequest


class Flow(SingleFieldFeature):
    NAME = "flow"

    def __init__(self, loc: str | int):
        super().__init__(self.loc_name(loc), FieldRequest("hydro", "flow", "1h", "mean", loc))

    def _remove_faulty_periods(self, df: pd.DataFrame) -> pd.DataFrame:
        # the periods below are for just for bern. once we want flow from other locations, rethink this.
        if self.field.location != LOC_BERN:
            return df

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
        df = self.remove_outliers(df)
        df = self.interpolate(df)

        return df
