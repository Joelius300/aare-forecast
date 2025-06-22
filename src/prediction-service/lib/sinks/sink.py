from abc import ABC, abstractmethod

import pandas as pd


# maybe combine this with the fetching methods used at inference, i.e. one interface for insertion, one for fetching
# because timescale will need both while influxdb only implements fetching.
class Sink(ABC):
    @abstractmethod
    def persist(self, data: pd.DataFrame) -> None:
        """Persist the data to this store."""
        pass