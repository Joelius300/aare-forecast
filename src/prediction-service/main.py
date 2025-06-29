import datetime
from typing import TypedDict, NotRequired

import pandas as pd
from darts import TimeSeries
from darts.models.forecasting.forecasting_model import GlobalForecastingModel
from psycopg_pool import ConnectionPool

from aare.utils import find_project_root
from lib import hello
from lib.external_sources.external_source import ExternalSource

from lib.external_sources.registry import SourceRegistry, Sources
from lib.persistance.tables.prediction import PredictionTable
from lib.persistance.timescale_table import TimescaleTable


# dict/None = don't know the type yet :)


class InferenceData(TypedDict):
    # no batch support (or necessity)
    series: NotRequired[TimeSeries]
    past_covariates: NotRequired[TimeSeries]
    future_covariates: NotRequired[TimeSeries]


class FeatureIds(TypedDict):
    """Reference features from the registry for past and future covariates."""

    # no need for target at the moment
    past: list[str]
    future: list[str]


def load_model():
    # load model (from pickle)
    # load feature set from yaml
    pass


def persist_metadata(run_ts: datetime.datetime):
    # store version (?), features, etc.
    pass


def _fetch_cache_external(source: ExternalSource, table: TimescaleTable):
    df = source.fetch()
    table.insert(df)

    return df


def load_external_data(sources: Sources) -> dict[str, pd.DataFrame]:
    # pull from all registered external sources and store to db cache

    # can be parallelized later, or at least async
    return {source_name: _fetch_cache_external(**source) for source_name, source in sources.items()}


def get_inference_data(features: FeatureIds, external_data: dict[str, pd.DataFrame]) -> InferenceData:
    # get actual features from the registry
    # take what you can from influx, the rest must be in external_data
    pass


def predict(model: GlobalForecastingModel, data: InferenceData) -> pd.DataFrame:
    # use the data to predict the coming temperature
    # TODO read args from some config yaml
    args = dict(n=96, num_samples=128)

    pred = model.predict(**data, **args)  # not sure why pyright is mad here
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
    hello()
    print(f"Project Root: {find_project_root()}")

    run_ts = datetime.datetime.now(datetime.UTC)
    model, features = load_model()

    # could also use NullConnectionPool because we don't really need pooling atm.
    # with this config, it opens a connection immediately and keeps it open/ready.
    conn_pool = ConnectionPool("host=127.0.0.1 dbname=aare_oraku user=postgres password=password", min_size=1)
    with conn_pool:
        sources = SourceRegistry().configure_sources(conn_pool)

        # persist_metadata(run_ts)
        external_data = load_external_data(sources)
        data = get_inference_data(features, external_data)

        prediction = predict(model, data)
        persist_prediction(run_ts, prediction, PredictionTable(conn_pool))
