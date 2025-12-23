# ruff: noqa: E402
# ignore 'imports not at top of file' for this file
from datetime import UTC, datetime

from aare.logging import setup_logging

import_start_ts = datetime.now(UTC)

import logging
from typing import Optional

import configargparse
import pandas as pd
import psycopg
from darts import TimeSeries
from darts.dataprocessing import Pipeline
from darts.dataprocessing.transformers import InvertibleDataTransformer
from darts.models.forecasting.forecasting_model import GlobalForecastingModel
from psycopg_pool import ConnectionPool

from aare.compat.types import DataTransformers
from aare.storage.metadata import AareModel
from aare.storage.model import load_model
from aare.params import set_params_file
from lib.data.compile_data import get_inference_data, scale_inference_data
from lib.data.inference_data import InferenceData
from lib.external_sources.external_source import ExternalSource
from lib.external_sources.registry import SourceRegistry, Sources
from lib.persistence.tables.forecast import ForecastTable
from lib.persistence.tables.forecast_meta import ForecastMetaTable
from lib.persistence.timescale_table import TimescaleTable

logger = logging.getLogger(__name__)


def _fetch_cache_external(name: str, source: ExternalSource, table: TimescaleTable, run_ts: datetime):
    """fetch, cache in db and then transform and return data from the source"""
    table.ensure_table_exists()
    df = source.fetch()

    logger.debug(f"Fetched {len(df)} rows from {name}")
    df["run_ts"] = run_ts

    table.insert(df)
    logger.debug(f"Inserted {len(df)} rows into {table.table_name}")

    prepared = source.prepare(df)

    return prepared


def load_external_data(sources: Sources, run_ts: datetime) -> dict[str, pd.DataFrame]:
    """pull from all registered external sources and store to db cache"""

    # can be parallelized later, or at least async
    return {
        source_name: _fetch_cache_external(source_name, **source, run_ts=run_ts)
        for source_name, source in sources.items()
    }


def predict(
    model: GlobalForecastingModel,
    data: InferenceData,
    target_scaler: Optional[InvertibleDataTransformer | Pipeline],
    horizon: int,
    num_samples: int,
) -> pd.DataFrame:
    """use the fetched data to predict the future temperature"""
    args = dict(n=horizon)
    if model.supports_probabilistic_prediction:
        args["num_samples"] = num_samples

    pred = model.predict(**data, **args)  # not sure why pyright is mad here  # pyright: ignore [reportArgumentType]
    if not isinstance(pred, TimeSeries):
        raise ValueError(f"Model returned '{type(pred)}' instead of TimeSeries.")

    if pred.is_stochastic:
        raise ValueError("Model returned a stochastic forecast; not supported yet")

    if target_scaler:
        assert isinstance(target_scaler, (InvertibleDataTransformer, Pipeline))
        pred = target_scaler.inverse_transform(pred)
        assert isinstance(pred, TimeSeries), "Not a TimeSeries anymore after inverse transform"

    return pred.to_dataframe().reset_index(names="time")


def persist_forecast(run_ts: datetime, forecast: pd.DataFrame, table: TimescaleTable):
    table.ensure_table_exists()
    to_store = forecast.copy()
    to_store["run_ts"] = run_ts
    table.insert(to_store)


def make_forecast(
    run_ts: datetime,
    model_meta: AareModel,
    model: GlobalForecastingModel,
    scalers: Optional[DataTransformers],
    conn_pool: ConnectionPool,
    horizon: int,
    num_samples: int,
):
    # configure and pull external sources
    sources = SourceRegistry().configure_sources(conn_pool)
    external_data = load_external_data(sources, run_ts)

    # compile inference data data from internal (influx) and external data
    data = get_inference_data(model_meta["features"], model.extreme_lags, external_data, run_ts)
    data = scale_inference_data(data, scalers)

    # actually make and store forecast with loaded model
    forecast = predict(model, data, scalers.get("series") if scalers else None, horizon, num_samples)
    persist_forecast(run_ts, forecast, ForecastTable(conn_pool))


def get_args():
    p = configargparse.ArgParser(auto_env_var_prefix="oraku_", default_config_files=["./dev_config.yaml"])
    p.add_argument(
        "-c", "--connection-string", required=True, type=str, help="Connection string for the postgres database"
    )
    p.add_argument("-m", "--model-path", required=True, type=str, help="Path to the model meta file (json)")
    p.add_argument("-n", "--horizon", default=96, type=int, help="Number of hours to forecast into the future")
    p.add_argument("--num-samples", default=128, type=int, help="Number of samples to take for probabilistic forecasts")
    p.add_argument("--logging-level", default="INFO", type=str, help="Logging level for logging module")
    p.add_argument("--loki-url", default=None, type=str, help="Base URL for the loki instance")
    p.add_argument("--loki-password", default=None, type=str, help="Password for the 'loki' user in loki")

    return p.parse_args()


def main():
    args = get_args()
    setup_logging(args.logging_level, args.loki_url, args.loki_password, "aare-oraku-service")

    run_ts = datetime.now(UTC)

    # load model and all required accessories into memory
    model_meta, model, scalers = load_model(args.model_path)
    # set params file for read_params to the one that was used when training the model
    set_params_file(model_meta["params_path"])

    # could also use NullConnectionPool because we don't really need pooling atm.
    # with this config, it opens a connection immediately and keeps it open/ready.
    conn_pool = ConnectionPool(
        args.connection_string,
        min_size=1,
        connection_class=psycopg.Connection,
    )
    with conn_pool:
        metadata_table = ForecastMetaTable(conn_pool)
        metadata_table.ensure_table_exists()
        metadata_table.insert_metadata(run_ts, model_meta, args)

        # noinspection PyBroadException
        status = "success"
        error = None
        try:
            # do the hard part
            make_forecast(run_ts, model_meta, model, scalers, conn_pool, args.horizon, args.num_samples)
        except Exception as e:
            # only catches error during fetching and forecasting, mostly because fetching has external factors.
            # issues with the database or loading the model will only be visible in the app/container logs.
            status = "failure"
            error = str(e)
            logger.exception("Couldn't finish the forecast run.", exc_info=True)

        finished_at = datetime.now(UTC)

        metadata_table.update_metadata(run_ts, status, error, finished_at)

    logger.info(f"Finished run in {finished_at - run_ts} (+ {run_ts - import_start_ts} imports)")


if __name__ == "__main__":
    main()
