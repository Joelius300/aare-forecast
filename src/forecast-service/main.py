# ruff: noqa: E402
# ignore 'imports not at top of file' for this file

from datetime import UTC, datetime

import_start_ts = datetime.now(UTC)

from collections.abc import Awaitable
import logging
import asyncio

import uvloop
import pandas as pd
from darts import TimeSeries
from darts.dataprocessing import Pipeline
from darts.dataprocessing.transformers import InvertibleDataTransformer
from darts.models.forecasting.forecasting_model import GlobalForecastingModel
from psycopg import AsyncConnection
from psycopg.rows import TupleRow
from psycopg_pool import AsyncConnectionPool

from aare_logging.logging import setup_logging
from aare_timescale.timescale_table import TimescaleTable
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
from lib.args import get_args

logger = logging.getLogger(__name__)


async def _fetch_cache_external(name: str, source: ExternalSource, table: TimescaleTable, run_ts: datetime):
    """fetch, cache in db and then transform and return data from the source"""
    await table.ensure_table_exists()
    df = await source.fetch()

    logger.debug(f"Fetched {len(df)} rows from {name}")
    df["run_ts"] = run_ts

    await table.insert(df)
    logger.debug(f"Inserted {len(df)} rows into {table.table_name}")

    prepared = source.prepare(df)

    return prepared


async def load_external_data(sources: Sources, run_ts: datetime) -> dict[str, pd.DataFrame]:
    """pull from all registered external sources and store to db cache"""
    source_names: list[str] = []
    tasks: list[Awaitable[pd.DataFrame | BaseException]] = []
    for source_name, source in sources.items():
        source_names.append(source_name)
        tasks.append(_fetch_cache_external(source_name, **source, run_ts=run_ts))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    external_data: dict[str, pd.DataFrame] = {}
    for source_name, result in zip(source_names, results):
        if isinstance(result, BaseException):
            logger.error(f"Could not fetch and store '{source_name}': {result}")
            continue

        assert isinstance(result, pd.DataFrame), "result is not a dataframe and not an exception"
        external_data[source_name] = result

    return external_data


def predict(
    model: GlobalForecastingModel,
    data: InferenceData,
    target_scaler: InvertibleDataTransformer | Pipeline | None,
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


async def persist_forecast(run_ts: datetime, forecast: pd.DataFrame, table: TimescaleTable):
    await table.ensure_table_exists()
    to_store = forecast.copy()
    to_store["run_ts"] = run_ts
    await table.insert(to_store)


async def make_forecast(
    run_ts: datetime,
    model_meta: AareModel,
    model: GlobalForecastingModel,
    scalers: DataTransformers | None,
    conn_pool: AsyncConnectionPool,
    horizon: int,
    num_samples: int,
):
    # configure and pull external sources
    sources = SourceRegistry().configure_sources(conn_pool)
    external_data = await load_external_data(sources, run_ts)

    # compile inference data data from internal (influx) and external data
    data = get_inference_data(model_meta["features"], model.extreme_lags, external_data, run_ts)
    data = scale_inference_data(data, scalers)

    # actually make and store forecast with loaded model
    forecast = predict(model, data, scalers.get("series") if scalers else None, horizon, num_samples)
    await persist_forecast(run_ts, forecast, ForecastTable(conn_pool))


async def main():
    args = get_args()
    setup_logging(args.logging_level, args.loki_url, args.loki_password, "aare-oraku-service")

    run_ts = datetime.now(UTC)

    # load model and all required accessories into memory
    model_meta, model, scalers = load_model(args.model_path)
    # set params file for read_params to the one that was used when training the model
    set_params_file(model_meta["params_path"])

    # could also use AsyncNullConnectionPool because we probably don't really need pooling atm.
    # with this config, it always keeps one connection open/ready and could/would use more if multiple are need at once.
    conn_pool = AsyncConnectionPool(
        args.connection_string,
        min_size=1,  # keep one open at all times
        max_size=4,
        # shouldn't need more workers to manage those connections (big default on min_size, num_workers, ..)
        num_workers=1,
        connection_class=AsyncConnection[TupleRow],  # needed to make pyright happy, but is already the default
    )
    async with conn_pool:
        metadata_table = ForecastMetaTable(conn_pool)
        await metadata_table.ensure_table_exists()
        await metadata_table.insert_metadata(run_ts, model_meta, args)

        # noinspection PyBroadException
        status = "success"
        error = None
        try:
            # do the hard part
            await make_forecast(run_ts, model_meta, model, scalers, conn_pool, args.horizon, args.num_samples)
        except Exception as e:
            # only catches error during fetching and forecasting, mostly because fetching has external factors.
            # issues with the database or loading the model will only be visible in the app/container logs.
            status = "failure"
            error = str(e)
            logger.exception("Couldn't finish the forecast run.", exc_info=True)

        finished_at = datetime.now(UTC)

        await metadata_table.update_metadata(run_ts, status, error, finished_at)

    logger.info(f"Finished run in {finished_at - run_ts} (+ {run_ts - import_start_ts} imports)")


if __name__ == "__main__":
    uvloop.run(main())
