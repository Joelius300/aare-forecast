from datetime import datetime, timedelta
from typing import cast, Optional

import darts
import pandas as pd
from darts import TimeSeries

from aare.compat.types import ExtremeLags, DataTransformers
from aare.constants import TIME
from aare.feature_identifiers import FeatureIdentifiers

from aare.features.base.feature import Feature
from aare.features.registry import FEATURES
from aare.preparation import resample
from aare.remote_existenz_store import RemoteExistenzStore
from lib.data.inference_data import InferenceData
from lib.external_sources.translations import MEAS_TRANS


def _get_features_and_fields(feature_ids: list[str]):
    features = [FEATURES[f] for f in feature_ids]
    fields = list(set(field for f in features for field in f.required_fields))

    return features, fields


def _prepare_series(features: list[Feature], df: pd.DataFrame) -> TimeSeries:
    df = resample(df)
    tss = [f.make(df) for f in features]
    # need a single TimeSeries with all relevant components for inference
    series = darts.concatenate(tss, axis="component")
    if not series.gaps(mode="any").empty:
        raise ValueError("There are (unfillable) gaps, cannot make a prediction!")

    return series


EXTRA_PAST_HOURS = 2


# TODO refactor this a bit to reduce duplications and function size
#  This function has some overlap with FeatureSet, but not sure if it can be consolidated.
def get_inference_data(
    features: FeatureIdentifiers,
    extreme_lags: ExtremeLags,
    external_data: dict[str, pd.DataFrame],
    run_ts: datetime,
) -> InferenceData:
    """Fetch required data from internal influx store and combine with external data to get inference data."""
    influx_store = RemoteExistenzStore()  # TODO eww, need to remove these coupling

    # targets can always be taken from influx because they are either in the past or predicted AR
    target_features, target_fields = _get_features_and_fields(features["targets"])
    # reminder to self: the min in min_target_lag means it's a lower value and further back,
    # it's NOT the minimum it will look back (it's actually the maximum/furthest it will look back)
    min_target_lag = extreme_lags[0]
    assert min_target_lag is not None and min_target_lag < 0, f"Invalid min_target_lag (for us): {min_target_lag}"

    # take data from further in the past to make sure we get all the required data, darts handles that
    hours_back = abs(min_target_lag) + EXTRA_PAST_HOURS
    period = run_ts - timedelta(hours=hours_back), run_ts  # start, end
    target_df = influx_store.query(period, target_fields)
    target = _prepare_series(target_features, target_df)

    if "past" in features and features["past"] is not None:
        # would need to take from influx for the past extreme_lags[2] hours.
        # Reminder: past covariates are known in the same time-span as the target itself
        # by definition, past cov cannot be known into the future, so fetching from external source
        # would make them a future cov (the weird actual/forecast mixing we're doing currently makes this
        # a bit more confusing to understand). If the model is AR and has an output chunk len < our desired horizon,
        # then we would need the _past_ covariate ALSO into the future, which is nonsense. So either use future cov,
        # use a model that outputs enough data points that no AR is needed, or turn past cov into an extra AR target.
        raise NotImplementedError("Currently support for past covariates.")

    future = None
    if "future" in features and features["future"] is not None:
        cols = []
        future_features, future_fields = _get_features_and_fields(features["future"])
        # take the data from the external sources for the future data points
        for field in future_fields:
            assert field.measurement in MEAS_TRANS, (
                f"Measurement of required field '{field}' ({field.measurement}) has no translation!"
            )
            source_name = MEAS_TRANS[field.measurement]
            # set index here so it is included in the series and kept after concatenation
            df = external_data[source_name].set_index(TIME)
            cols.append(df[field.name])

        future_df_future: pd.DataFrame = pd.concat(cols, axis=1).reset_index(names=TIME)
        # discard data from external sources that are in the past since we can fetch more
        # accurate data from influx directly. At least for our current sources.
        future_df_future = cast(pd.DataFrame, future_df_future[future_df_future[TIME] >= run_ts])

        # must combine that future data with past data, if the model uses it
        min_future_lag = extreme_lags[4]
        assert min_future_lag is not None, f"Invalid min_future_lag (for us): {min_future_lag}"
        if min_future_lag >= 0:
            # no need to look into the past and fetch from influx, it only uses future future cov vals
            future_df = future_df_future
        else:
            # it also uses past future cov values, so we need to fetch from influx
            hours_back = abs(min_future_lag) + EXTRA_PAST_HOURS
            period = run_ts - timedelta(hours=hours_back), run_ts  # start, end
            future_df_past = influx_store.query(period, future_fields)
            future_df_past = resample(future_df_past)  # remove trailing 08:40 data point (see example below)
            future_df = pd.concat([future_df_past, future_df_future], axis=0, ignore_index=True)

            # Influx also returns a data point at the very end that's basically at run_ts. To remove it, we
            # can either just resample future_df_past (see above), or we could merge it together with the
            # future prediction and take the mean. Here it's important that the data point at 01:00 only contains data
            # from before (<= 01:00) and 08:45 is merged with 09:00, so closed and label must be set to 'right'.
            # THE REASON I'M JUST DROPPING INSTEAD OF MERGING is that at 08:50 it would probably increase accuracy
            # but at 08:10 it might decrease it because it pulls it to the earlier hour. For the sake of transparency
            # and simplicity, we just ignore the last "partial" data point from influx.
            # > should read 1h from config, if you do this
            # future_df = future_df.set_index(TIME).resample("1h", closed="right", label="right").mean().reset_index()

        future = _prepare_series(future_features, future_df)

    return {
        "series": target,
        "future_covariates": future,
    }


def scale_inference_data(data: InferenceData, scalers: Optional[DataTransformers]) -> InferenceData:
    """Scale the inference data, if scalers are provided."""
    if not scalers:
        return data

    scaled = {}

    # if scalers are provided, it's expected that all targets and covariates have a scaler
    for key in data.keys():
        # just to make type checkers happy
        assert isinstance(data, dict)
        assert isinstance(scalers, dict)
        assert key in scalers, f"Scalers dict was provided, but for '{key}' there wasn't one"
        scaled[key] = scalers[key].transform(data[key])

    return cast(InferenceData, scaled)
