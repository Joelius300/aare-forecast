from typing import cast

import httpx
import numpy as np
import pandas as pd

from aare.constants import TIME
from lib.external_sources.external_source import ExternalSource


class MeteoTestSource(ExternalSource):
    """Fetch forecasts from Meteotest (internal Meteotest service)"""

    def __init__(self, url: str, locations: list[str]):
        self.url = url
        self.locations = locations

    def fetch(self) -> pd.DataFrame:
        # can be made async later
        r = httpx.get(self.url)
        r.raise_for_status()

        body = r.json()
        mos = body["payload"]["mos"]

        dfs = []
        for loc in self.locations:
            df = self._to_df(mos[loc])
            df["location"] = loc
            dfs.append(df)

        df = pd.concat(dfs, axis="index", ignore_index=True)

        return df

    def prepare(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        df = raw_data.drop("run_ts", axis=1)
        df["location"] = df["location"].str.lower()

        # setting index to (time, location) then unstacking [location] is the same as
        # pivoting with index="time", columns="location" and values = {all other columns}
        df = cast(pd.DataFrame, df.set_index(["time", "location"]).unstack())
        # after unstacking (or pivoting, doesn't matter) the columns will be a MultiIndex with
        # the first level the original name of the col (e.g. 'tt') and the second the location (e.g. 'bern')
        # so they have to be combined into a single combined name
        df.columns = df.columns.map(lambda x: f"{x[0]}_{x[1]}")
        # currently, the index is the time col and named 'time', but we need parity so it must be an extra col '_time'
        df = df.reset_index(names=TIME)

        return df

    def _to_df(self, data: dict):
        df = pd.DataFrame.from_dict(data, orient="index", dtype=np.float32)
        # timestamps from meteotest are naive but should be interpreted as UTC
        df.index = pd.to_datetime(df.index).tz_localize("UTC")
        df = df.reset_index(names="time")

        return df
