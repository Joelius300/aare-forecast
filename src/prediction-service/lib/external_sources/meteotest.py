import numpy as np
import pandas as pd
import httpx


class MeteoTestSource:
    """Fetch predictions from Meteotest (internal Meteotest service)"""

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

    def _to_df(self, data: dict):
        df = pd.DataFrame.from_dict(data, orient="index", dtype=np.float32)
        df.index = pd.to_datetime(df.index)
        df = df.reset_index(names="time")

        return df
