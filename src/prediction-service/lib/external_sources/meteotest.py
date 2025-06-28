import numpy as np
import pandas as pd
import httpx


class MeteoTestSource:
    """Fetch predictions from Meteotest (internal Meteotest service)"""
    def __init__(self, url: str):
        self.url = url
        
    def fetch(self) -> pd.DataFrame:
        # can be made async later
        r = httpx.get(self.url)
        r.raise_for_status()
        
        body = r.json()
        mos = body["payload"]["mos"]
        
        # TODO make generic solution aligned with the names in the locations module
        bern = mos["BERN"]
        thun = mos["THUN"]

        bern_df = self._to_df(bern)
        thun_df = self._to_df(thun)
        
        df = pd.merge(bern_df, thun_df, left_index=True, right_index=True, suffixes=("bern", "thun"))
        
        return df
        

    def _to_df(self, data: dict):
        df = pd.DataFrame.from_dict(data, orient="index", dtype=np.float32)
        df.index = pd.to_datetime(df.index)
        df = df.reset_index(names="time")
        
        return df
