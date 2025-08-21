import datetime
import logging
from typing import TypedDict, NotRequired, Optional, cast

import darts
import pandas as pd
import psycopg
from darts import TimeSeries
from darts.dataprocessing import Pipeline
from darts.dataprocessing.transformers import InvertibleDataTransformer
from darts.models.forecasting.forecasting_model import GlobalForecastingModel
from psycopg_pool import ConnectionPool

from aare.compat.types import ExtremeLags
from aare.constants import TIME
from aare.feature_identifiers import FeatureIdentifiers
from aare.features.base.feature import Feature
from aare.features.registry import FEATURES
from aare.preparation import resample
from aare.remote_existenz_store import RemoteExistenzStore
from aare.storage.model import load_model
from lib.external_sources.external_source import ExternalSource
from lib.external_sources.registry import SourceRegistry, Sources
from lib.external_sources.translations import MEAS_TRANS
from lib.persistence.tables.prediction import PredictionTable
from lib.persistence.timescale_table import TimescaleTable

# dict/None = don't know the type yet :)
logger = logging.getLogger(__name__)


class InferenceData(TypedDict):
    # no batch support (or necessity)
    # not provided is the same as None here, that's not always the case (!)
    series: TimeSeries  # some models allow covariate-only pred, but we'll never use it so non-nullable
    past_covariates: NotRequired[Optional[TimeSeries]]
    future_covariates: NotRequired[Optional[TimeSeries]]


def persist_metadata(run_ts: datetime.datetime):
    # store version (?), features, etc.
    raise NotImplementedError()


def _fetch_cache_external(name: str, source: ExternalSource, table: TimescaleTable, run_ts: datetime.datetime):
    # fetch, cache in db and then transform and return data from the source
    table.ensure_table_exists()
    df = source.fetch()

    logger.debug(f"Fetched {len(df)} rows from {name}")
    df["run_ts"] = run_ts

    table.insert(df)
    logger.debug(f"Inserted {len(df)} rows into {table.table_name}")

    prepared = source.prepare(df)

    return prepared


def load_external_data(sources: Sources, run_ts: datetime.datetime) -> dict[str, pd.DataFrame]:
    # pull from all registered external sources and store to db cache

    # can be parallelized later, or at least async
    return {
        source_name: _fetch_cache_external(source_name, **source, run_ts=run_ts)
        for source_name, source in sources.items()
    }


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
    run_ts: datetime.datetime,
) -> InferenceData:
    influx_store = RemoteExistenzStore()  # eww, need to remove these coupling

    # targets can always be taken from influx because they are either in the past or predicted AR
    target_features, target_fields = _get_features_and_fields(features["targets"])
    # reminder to self: the min in min_target_lag means it's a lower value and further back,
    # it's NOT the minimum it will look back (it's actually the maximum/furthest it will look back)
    min_target_lag = extreme_lags[0]
    assert min_target_lag is not None and min_target_lag < 0, f"Invalid min_target_lag (for us): {min_target_lag}"

    # take data from further in the past to make sure we get all the required data, I think darts handles that
    # TODO set the period end to the run_ts so it's reproducible and not bound to some "now" implementation (!)
    target_df = influx_store.query(f"{min_target_lag - EXTRA_PAST_HOURS}h", target_fields)
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
            future_df_past = influx_store.query(f"{min_future_lag - EXTRA_PAST_HOURS}h", future_fields)
            future_df_past = resample(future_df_past)  # remove trailing 08:40 data point (see below)
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


def predict(
    model: GlobalForecastingModel, data: InferenceData, target_scaler: Optional[InvertibleDataTransformer | Pipeline]
) -> pd.DataFrame:
    # use the data to predict the coming temperature
    # TODO read args from some config yaml
    args = dict(n=96)
    if model.supports_probabilistic_prediction:
        args["num_samples"] = 128

    pred = model.predict(**data, **args)  # not sure why pyright is mad here  # pyright: ignore [reportArgumentType]
    if not isinstance(pred, TimeSeries):
        raise ValueError(f"Model returned '{type(pred)}' instead of TimeSeries.")

    if pred.is_stochastic:
        raise ValueError("Model returned a stochastic prediction; not supported yet")

    if target_scaler:
        assert isinstance(target_scaler, (InvertibleDataTransformer, Pipeline))
        pred = target_scaler.inverse_transform(pred)
        assert isinstance(pred, TimeSeries), "Not a TimeSeries anymore after inverse transform"

    return pred.to_dataframe().reset_index(names="time")


def persist_prediction(run_ts: datetime.datetime, prediction: pd.DataFrame, table: TimescaleTable):
    table.ensure_table_exists()
    to_store = prediction.copy()
    to_store["run_ts"] = run_ts
    table.insert(to_store)


if __name__ == "__main__":
    logging.basicConfig(level="DEBUG")
    run_ts = datetime.datetime.now(datetime.UTC)
    model_meta, model, scalers = load_model(name="LR", version="dev")
    print(model.extreme_lags)

    # could also use NullConnectionPool because we don't really need pooling atm.
    # with this config, it opens a connection immediately and keeps it open/ready.
    # TODO read from config/env
    conn_pool = ConnectionPool(
        "host=127.0.0.1 dbname=aare_oraku user=postgres password=password",
        min_size=1,
        connection_class=psycopg.Connection,
    )
    with conn_pool:
        sources = SourceRegistry().configure_sources(conn_pool)

        # persist_metadata(run_ts)
        external_data = load_external_data(sources, run_ts)

        data = get_inference_data(model_meta["features"], model.extreme_lags, external_data, run_ts)

        if scalers is not None:
            # if scalers are provided, it's expected that all targets and covariates have a scaler
            for key in data.keys():
                # just to make type checkers happy
                assert isinstance(data, dict)
                assert isinstance(scalers, dict)
                assert key in scalers, f"Scalers dict was provided, but for '{key}' there wasn't one"
                data[key] = scalers[key].transform(data[key])

        prediction = predict(model, data, scalers.get("series") if scalers else None)
        persist_prediction(run_ts, prediction, PredictionTable(conn_pool))

        print("fini")
