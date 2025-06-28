import datetime
from typing import TypedDict, NotRequired

import pandas as pd
from darts import TimeSeries
from darts.models.forecasting.forecasting_model import GlobalForecastingModel
from psycopg_pool import ConnectionPool

from aare.utils import find_project_root
from lib import hello
from lib.sinks.sink import Sink
from lib.sinks.timescale import TimescaleSink


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


def fetch_external_data() -> dict:
    # pull from all registered external sources

    # ACTUALLY need to decided if we want to fetch _all_ data even if we're not using it for the model (not in features)
    # because that would build a dataset of data we can use later for services that don't offer historical forecasts
    # like the flow prediction of BAFU. -> yes do that :)
    pass


def get_inference_data(features: FeatureIds, external_data: dict) -> InferenceData:
    # get actual features from the registry
    # take what you can from influx, the rest must be in external_data
    pass


def persist_data(run_ts: datetime.datetime, data: dict):
    # store all the data together with the run_ts as identification
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


def persist_prediction(run_ts: datetime.datetime, prediction: pd.DataFrame, sink: Sink):
    # store prediction to db
    to_store = prediction.copy()
    to_store["run_ts"] = run_ts
    pred_cols = ["run_ts", "time", "temp_bern"]
    if len(to_store.columns) != len(pred_cols):
        raise ValueError("Unexpected number of columns in prediction")

    to_store = to_store[pred_cols]
    sink.persist("forecast", to_store)


if __name__ == "__main__":
    hello()
    print(f"Project Root: {find_project_root()}")

    run_ts = datetime.datetime.now(datetime.UTC)
    model, features = load_model()

    # could also use NullConnectionPool because we don't really need pooling atm.
    # with this config, it opens a connection immediately and keeps it open/ready.
    conn_pool = ConnectionPool("host=127.0.0.1 dbname=aare_oraku user=postgres password=password", min_size=1)
    with conn_pool:
        # todo fix crazy psycopg typing
        pg_sink = TimescaleSink(conn_pool)  # pyright: ignore [reportArgumentType]

        # persist_metadata(run_ts)
        external_data = fetch_external_data()
        persist_data(run_ts, external_data)
        data = get_inference_data(features, external_data)

        prediction = predict(model, data)
        persist_prediction(run_ts, prediction, pg_sink)
