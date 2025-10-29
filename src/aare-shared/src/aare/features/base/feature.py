from abc import ABC, abstractmethod
import pandas as pd
from darts import TimeSeries

from aare.params import read_params
from aare.remote_existenz_store import FieldRequest


class Feature(ABC):
    """
    A feature is a transformation of 1 or more fields from a source like InfluxDB or an external source.
    Most features are just a single field from influx, see SingleFieldFeature for that.
    Features handle cleanup and preparation of the relevant data.
    Features will be registered with a good string name so they can be used easily for hyperparameter tuning and
    testing out different models with different inputs easily.

    Example:
        feature_set:
        - bern_temp
        - bern_tt
        - bern_tt_log
        - bern_tt_cube
    """

    def __init__(self, name: str, required_fields: FieldRequest | list[FieldRequest]):
        # name is kept generic since not all features are necessarily bound to a location.
        # however, most features will be [bound to a location] so their names should reflect that,
        # e.g. temp_bern and temp_thun should be different instances but the same class,
        # unless they require different cleanup etc. -> IDEA: use the same class but when looking for
        # cleanup params allow 'temp_bern' to override values of 'temp', then you don't need extra classes
        self._name = name
        self._required_fields = required_fields if isinstance(required_fields, list) else [required_fields]
        feature_params = read_params()["features"]

        self.base_params = feature_params.get(self.base_name(), {})
        """Params specified in the features section of params.yaml under the base_name of this feature (e.g. tt)"""
        self.params = self.base_params | feature_params.get(name, {})
        """Params specified in the features section of params.yaml under the actual of this feature (e.g. tt_bern) combined with the base_params."""

    @classmethod
    def base_name(cls) -> str:
        """
        The base name of this feature without location identifier or similar.
        Reads the 'NAME' constant if a class if defined. Allowed: [a-z0-9] (no underscore)
        """
        name = getattr(cls, "NAME", None)
        if not name:
            raise ValueError(f"The feature '{cls}' does not override base_name or specify a 'NAME' constant.")

        return name

    @classmethod
    def loc_name(cls, loc: str | int):
        """Return the base name together with a location, since this is often how you want to identify."""
        return cls.base_name() + f"_{loc}"

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
