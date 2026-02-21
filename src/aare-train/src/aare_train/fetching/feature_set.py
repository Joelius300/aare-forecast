import itertools
from typing import Optional
import darts
from darts import TimeSeries
from darts.utils.missing_values import extract_subseries
from darts.utils.ts_utils import retain_period_common_to_all

import pandas as pd

from aare_train.features.base.feature import Feature
from aare_train.params import SplitParams
from aare_train.preparation import resample
from aare_train.fetching.remote_existenz_store import RemoteExistenzStore


# TODO it would be very nice to have the option (maybe per feature) to
#  avoid removing so much data as that fragments the training and val data.
#  Instead it would fill the gaps up to a certain size with -99 and create a companion feature
#  indicating whether it was filled in or not.
#  Maybe extra feature could be to fill in spots where maximum X out of Y features are NaN,
#  if more are NaN, a split is needed.
class FeatureSet:
    def __init__(
        self,
        targets: Feature | list[Feature],
        *,
        past: list[Feature] | None = None,
        future: list[Feature] | None = None,
        split_params: SplitParams | None = None,
    ):
        if not targets:
            raise ValueError("Must provide targets")

        if isinstance(past, list) and not past:
            raise ValueError("Provide None or a non-empty list for past features")

        if isinstance(future, list) and not future:
            raise ValueError("Provide None or a non-empty list for future features")

        self._store = RemoteExistenzStore()  # TODO decouple
        self._targets: list[Feature] = targets if isinstance(targets, list) else [targets]
        self._past = past
        self._future = future
        self._split_params = split_params

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
        self, df: pd.DataFrame, min_len: int = 1
    ) -> tuple[list[TimeSeries], Optional[list[TimeSeries]], Optional[list[TimeSeries]]]:
        assert min_len >= 1, f"invalid min_len {min_len}"
        df = resample(df)

        # make all the features and combine them into a single wide series
        # to be able to extract all subseries by removing any missing values.

        # every element of these arrays is a single TimeSeries instance with one or more components, potentially with NaNs
        targets = [f.make(df) for f in self._targets]
        pc = [f.make(df) for f in self._past] if self._past else None
        fc = [f.make(df) for f in self._future] if self._future else None

        # flat list of component names per feature type (target, past, future)
        target_comp = [c for feat_ts in targets for c in feat_ts.components]
        pc_comp = [c for feat_ts in pc for c in feat_ts.components] if pc else None
        fc_comp = [c for feat_ts in fc for c in feat_ts.components] if fc else None

        # components is a list of TimeSeries with unique components but (approximately) the same time periods
        components = targets + (pc or []) + (fc or [])
        # align times globally (only start and end, gaps are ignored)
        components = retain_period_common_to_all(components)

        # make a single TimeSeries with all components of all features
        wide = darts.concatenate(components, axis="component")

        # split into a list of TimeSeries (subseries) with all the same components but sliced to be non-overlapping periods without NaNs
        subs = extract_subseries(wide, mode="any")

        # make sure all splits contain at least min_len data points
        # also fixes the very weird case where splitting on gaps results in 0-length subseries ?!
        subs = [sub for sub in subs if len(sub) >= min_len]

        # reconstruct target, pc and fc so that there is one list per target/pc/fc with aligned TimeSeries that
        # all contain all components of all features of that type (one feature may contain multiple components)
        targets = [part[target_comp] for part in subs]
        pc = [part[pc_comp] for part in subs] if pc_comp else None
        fc = [part[fc_comp] for part in subs] if fc_comp else None

        # now targets, pc and fc all have the same number of subseries, all aligned (same time period), and no nans.
        return targets, pc, fc

    def get(self, start: str, end: str | None, min_len: int = 1):
        """
        Get clean non-null data for the specified period split into subseries of at least min_len points.
        If end is None, will get all data up to now.
        """
        return self._prepare(self._fetch_all((start, end) if end else start), min_len)

    def get_train(self, min_len: int = 1):
        """Get clean non-null training data split into subseries of at least min_len points."""
        if not self._split_params:
            raise ValueError("'train' data is not defined if split params aren't provided.")

        return self.get(self._split_params["train_split"], self._split_params["val_split"], min_len)

    def get_val(self, min_len: int = 1):
        """Get clean non-null validation data split into subseries of at least min_len points."""
        if not self._split_params:
            raise ValueError("'val' data is not defined if split params aren't provided.")

        return self.get(self._split_params["val_split"], self._split_params["test_split"], min_len)

    def get_test(self, end: str | None = None, min_len: int = 1):
        """
        Get clean non-null test data split into subseries of at least min_len points.
        Returns all data up to now unless you specify an end time.
        """
        if not self._split_params:
            raise ValueError("'test' data is not defined if split params aren't provided.")

        return self.get(self._split_params["test_split"], end, min_len)
