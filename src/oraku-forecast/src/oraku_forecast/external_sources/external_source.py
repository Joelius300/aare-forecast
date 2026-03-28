from abc import ABC, abstractmethod

import pandas as pd


class ExternalSource(ABC):
    @abstractmethod
    async def fetch(self) -> pd.DataFrame:
        """Fetch the data into a semi-raw format suitable for storage."""
        pass

    @abstractmethod
    def prepare(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Transform the data from the semi-raw data as it's stored into a usable format like queried from influx."""
        pass
