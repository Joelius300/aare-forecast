from abc import ABC, abstractmethod

import pandas as pd


class ExternalSource(ABC):
    @abstractmethod
    def fetch(self) -> pd.DataFrame:
        pass