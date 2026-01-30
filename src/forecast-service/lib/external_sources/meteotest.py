from collections.abc import Sequence
import logging
from typing import cast, Any, override

import httpx
import numpy as np
import pandas as pd

from aare.constants import TIME
from lib.external_sources.external_source import ExternalSource

logger = logging.getLogger(__name__)


class MeteoTestSource(ExternalSource):
    """Fetch forecasts from Meteotest (internal Meteotest service)"""

    # dict with name translation from meteotest names to our internal names (see locations.py).
    # if an name is explicitly mapped to None, it does not have a mapping (yet).
    NAME_TRANSLATIONS = {
        "AARAU": None,
        "BERN": "BERN",
        "BRIENZ": "BRNZ",
        "BRUGG": "BRGG",
        "BIEL": "BIEL",
        "OLTEN": None,
        "RINGGENBERG": "INT",
        "SOLOTHURN": None,
        "THUN": "THUN",
        "BIELERSEE": None,  # this is in the middle of the lake; probably colder than Hagneck, but closer than Biel.
    }

    def __init__(self, url: str, locations: Sequence[str]):
        self.url: str = url
        self.locations: Sequence[str] = locations

        invalid_locs = set(locations) - set(self.NAME_TRANSLATIONS.keys())
        if invalid_locs:
            raise ValueError("Passed unknown/unsupported locations: " + ", ".join(invalid_locs))

    @override
    async def fetch(self) -> pd.DataFrame:
        async with httpx.AsyncClient() as client:
            r = await client.get(self.url)
            r.raise_for_status()

        body = r.json()
        mos = body["payload"]["mos"]

        dfs: list[pd.DataFrame] = []
        for loc in self.locations:
            if loc not in mos:
                logger.warning(f"Attempted to get MeteoTest location '{loc}', but it was not in the response!")
                continue

            df = self._to_df(mos[loc])
            df["location"] = loc
            dfs.append(df)

        df = pd.concat(dfs, axis="index", ignore_index=True)

        return df

    @override
    def prepare(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        df = raw_data.drop("run_ts", axis=1)
        loc_with_mapping = [loc for loc, mapping in self.NAME_TRANSLATIONS.items() if mapping is not None]
        df = df[df["location"].isin(loc_with_mapping)]  # don't prepare locations we can't use (yet)
        # replace locations with internal names and make them lowercase
        df["location"] = df["location"].map(self.NAME_TRANSLATIONS).str.lower()

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

    @staticmethod
    def _to_df(data: dict[str, Any]):
        df = pd.DataFrame.from_dict(data, orient="index", dtype=np.float32)
        # timestamps from meteotest are naive but should be interpreted as UTC
        df.index = pd.to_datetime(df.index).tz_localize("UTC")
        df = df.reset_index(names="time")

        return df
