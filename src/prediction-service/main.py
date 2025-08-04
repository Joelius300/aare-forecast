import datetime
import logging
from typing import TypedDict, NotRequired

import darts
import pandas as pd
import psycopg
from darts import TimeSeries
from darts.models.forecasting.forecasting_model import GlobalForecastingModel
from psycopg_pool import ConnectionPool

from aare.compat.types import ExtremeLags
from aare.feature_identifiers import FeatureIdentifiers
from aare.features.registry import FEATURES
from aare.preparation import resample
from aare.remote_existenz_store import RemoteExistenzStore
from aare.storage.model import load_model
from lib.external_sources.external_source import ExternalSource

from lib.external_sources.registry import SourceRegistry, Sources
from lib.persistance.timescale_table import TimescaleTable


# dict/None = don't know the type yet :)
logger = logging.getLogger(__name__)


class InferenceData(TypedDict):
    # no batch support (or necessity)
    series: NotRequired[TimeSeries]
    past_covariates: NotRequired[TimeSeries]
    future_covariates: NotRequired[TimeSeries]


def persist_metadata(run_ts: datetime.datetime):
    # store version (?), features, etc.
    raise NotImplementedError()


def _fetch_cache_external(name: str, source: ExternalSource, table: TimescaleTable, run_ts: datetime.datetime):
    # fetch, cache in db and then return data from the source
    table.ensure_table_exists()
    df = source.fetch()

    logger.debug(f"Fetched {len(df)} rows from {name}")
    df["run_ts"] = run_ts

    table.insert(df)
    logger.debug(f"Inserted {len(df)} rows into {table.table_name}")

    return df


def load_external_data(sources: Sources, run_ts: datetime.datetime) -> dict[str, pd.DataFrame]:
    # pull from all registered external sources and store to db cache

    # can be parallelized later, or at least async
    return {
        source_name: _fetch_cache_external(source_name, **source, run_ts=run_ts)
        for source_name, source in sources.items()
    }


# TODO refactor this a bit to reduce duplications and function size
#  This function has some overlap with FeatureSet, but not sure if it can be consolidated.
def get_inference_data(
    features: FeatureIdentifiers, extreme_lags: ExtremeLags, external_data: dict[str, pd.DataFrame]
) -> InferenceData:
    # get actual features from the registry
    # take what you can from influx, the rest must be in external_data
    influx_store = RemoteExistenzStore()

    # targets can always be taken from influx because they are either in the past or predicted AR
    target_features = [FEATURES[f] for f in features["targets"]]
    target_fields = [field for f in target_features for field in f.required_fields]

    max_target_lag = extreme_lags[0]

    assert max_target_lag is not None and max_target_lag < 0, f"Invalid max_target_lag (for us): {max_target_lag}"

    # take data from further in the past to make sure we get all the required data, I think darts handles that
    target_df = influx_store.query(f"{max_target_lag - 2}h", target_fields)
    target_df = resample(target_df)
    targets = [f.make(target_df) for f in target_features]
    # need a single TimeSeries with all relevant components for inference
    target = darts.concatenate(targets, axis="component")
    if not target.gaps(mode="any").empty:
        raise ValueError("There are (unfillable) gaps in the target variable, cannot make a prediction!")

    if "past" in features:
        # would need to take from influx for the past extreme_lags[2] hours.
        # Reminder: past covariates are known in the same time-span as the target itself
        # by definition, past cov cannot be known into the future, so fetching from external source
        # would make them a future cov (the weird actual/forecast mixing we're doing currently makes this
        # a bit more confusing to understand). If the model is AR and has an output chunk len < our desired horizon,
        # then we would need the _past_ covariate ALSO into the future, which is nonsense. So either use future cov,
        # use a model that outputs enough data points that no AR is needed, or turn past cov into an extra AR target.
        raise NotImplementedError("Currently support for past covariates.")

    if "future" in features:
        # TODO implement :)
        #  Currently, I think you could just build a df by taking all the pandas series out of the dataframes
        #  of the external sources until you have everything you need to make all features (i.e. all required
        #  fields are in a df), then from f.make on it's the same as the target var. Of course refactor into func.
        pass

    return {
        "series": target,
        # TODO
    }


def predict(model: GlobalForecastingModel, data: InferenceData) -> pd.DataFrame:
    # use the data to predict the coming temperature
    # TODO read args from some config yaml
    args = dict(n=96, num_samples=128)

    pred = model.predict(**data, **args)  # not sure why pyright is mad here  # pyright: ignore [reportArgumentType]
    if not isinstance(pred, TimeSeries):
        raise ValueError(f"Model returned '{type(pred)}' instead of TimeSeries.")

    if pred.is_stochastic:
        raise ValueError("Model returned a stochastic prediction; not supported yet")

    return pred.to_dataframe().reset_index(names="time")


def persist_prediction(run_ts: datetime.datetime, prediction: pd.DataFrame, table: TimescaleTable):
    # store prediction to db
    to_store = prediction.copy()
    to_store["run_ts"] = run_ts
    table.insert(to_store)


if __name__ == "__main__":
    logging.basicConfig(level="DEBUG")
    run_ts = datetime.datetime.now(datetime.UTC)
    # model, features = load_model()

    # could also use NullConnectionPool because we don't really need pooling atm.
    # with this config, it opens a connection immediately and keeps it open/ready.
    conn_pool = ConnectionPool(
        "host=127.0.0.1 dbname=aare_oraku user=postgres password=password",
        min_size=1,
        connection_class=psycopg.Connection,
    )
    with conn_pool:
        sources = SourceRegistry().configure_sources(conn_pool)

        # persist_metadata(run_ts)
        external_data = load_external_data(sources, run_ts)

        model_meta, model, scalers = load_model(name="LR", version="dev")
        data = get_inference_data(model_meta["features"], model.extreme_lags, external_data)

        # TODO scale
        # prediction = predict(model, data)
        # persist_prediction(run_ts, prediction, PredictionTable(conn_pool))

        print("fini")
