import itertools
from typing import Optional
import darts
from darts import TimeSeries
from darts.utils.missing_values import extract_subseries
from darts.utils.ts_utils import retain_period_common_to_all

import pandas as pd

from aare.features.base.feature import Feature
from aare.params import SplitParams
from aare.preparation import resample
from aare.remote_existenz_store import RemoteExistenzStore


class FeatureSet:
    def __init__(
        self,
        targets: Feature | list[Feature],
        *,
        past: Optional[list[Feature]] = None,
        future: Optional[list[Feature]] = None,
        split_params: SplitParams,
    ):
        self._store = RemoteExistenzStore()  # TODO decouple
        self._targets: list[Feature] = targets if isinstance(targets, list) else [targets]
        self._past = past
        self._future = future
        self._train_split = split_params["train_split"]
        self._val_split = split_params["val_split"]
        self._test_split = split_params["test_split"]

    @property
    def _all_features(self):
        return itertools.chain(
            self._targets, (self._past if self._past else []), (self._future if self._future else [])
        )

    @property
    def _all_fields(self):
        return list(set(field for feature in self._all_features for field in feature.required_fields))

    def _fetch_all(self, period: str | tuple[str, str]) -> pd.DataFrame:
        return self._store.query(period, self._all_fields)

    def _prepare(
        self, df: pd.DataFrame
    ) -> tuple[list[TimeSeries], Optional[list[TimeSeries]], Optional[list[TimeSeries]]]:
        df = resample(df)

        # make all the features and combine them into a single wide series
        # to be able to extract all subseries by removing any missing values.
        data = [f.make(df) for f in self._all_features]
        data = retain_period_common_to_all(data)
        data = darts.concatenate(data, axis="component")
        data = extract_subseries(data, mode="any")
        data = [sub for sub in data if len(sub) > 0]  # wild that this is needed

        # then reconstruct the splits
        targets = [part[[f.name for f in self._targets]] for part in data]
        pc = [part[[f.name for f in self._past]] for part in data] if self._past else None
        fc = [part[[f.name for f in self._future]] for part in data] if self._future else None

        # now targets, pc and fc all have the same number of subseries, all aligned (same time period), and no nans.
        return targets, pc, fc

    def get_train(self):
        return self._prepare(self._fetch_all((self._train_split, self._val_split)))

    def get_val(self):
        return self._prepare(self._fetch_all((self._val_split, self._test_split)))

    def get_test(self):
        return self._prepare(self._fetch_all(self._test_split))  # no upper bound ( = now() )
