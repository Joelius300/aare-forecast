"""
General Idea:

A feature is a transformation of 1 or more fields from the InfluxDB (and maybe other sources later on).
Most features are just directly the field, so that's the default case.
Features will be registered with a good string name so they can be used easily for hyperparameter tuning and
testing out different models with different inputs easily. Don't overdo it as long as we don't need it yet, just
implement the things we can use directly. At this point, this is just non-linear transformations of 1 influxdb field.
Also, data cleanup is very important so removing outliers and doing imputation must also be part of this.
Adding a property with the availability could also be helpful (e.g. bern_temp is 2001, but bern_tt is 2013).
Maybe this structure could be helpful later for inference when data has to be pulled from other sources, but you'll
probably want to reinvent things then anyway.

Example:
    feature_set:
      - bern_temp
      - bern_tt
      - bern_tt_log
      - bern_tt_cube
"""

from abc import ABC, abstractmethod
import pandas as pd
from darts import TimeSeries

from aare.remote_existenz_store import FieldRequest


class Feature(ABC):
    def __init__(self, name: str, required_fields: FieldRequest | list[FieldRequest]):
        self._name = name
        self._required_fields = required_fields if isinstance(required_fields, list) else [required_fields]

    @property
    def name(self) -> str:
        """The name for this feature. Allowed: [a-z0-9_]"""
        return self._name

    @property
    def required_fields(self) -> list[FieldRequest]:
        """The fields that must be fetched from the datasource to construct the feature."""
        return self._required_fields

    @abstractmethod
    def cleanup(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean up the raw data needed for the feature. Includes imputation.
        Must not manipulate the original input df. The input df may contain more
        columns than needed. The output df must have the needed columns, but doesn't
        have to have the same columns as the input. The frequency / number of rows must
        not be changed in this transformation.
        """
        pass

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> TimeSeries:
        """
        Transform the original, cleaned data into the feature TimeSeries.
        Functional transformations like taking the root should be done here.
        """
        pass

    def make(self, df: pd.DataFrame) -> TimeSeries:
        """Makes the feature by first calling cleanup, then transform."""
        return self.transform(self.cleanup(df))
