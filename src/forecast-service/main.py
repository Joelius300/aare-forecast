# ruff: noqa: E402
# ignore 'imports not at top of file' for this file

from datetime import UTC, datetime

# run duration of the service should ignore the time needed to import libraries, but we still want to log it
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
from aare_train.compat.types import DataTransformers
from aare_train.storage.metadata import AareModel
from aare_train.storage.model import load_model
from aare_train.params import set_params_file
from lib.data.compile_data import get_inference_data, scale_inference_data
from lib.data.inference_data import InferenceData
from lib.external_sources.external_source import ExternalSource
from lib.external_sources.registry import SourceRegistry, Sources
from lib.persistence.tables.forecast import ForecastTable
from lib.persistence.tables.forecast_meta import ForecastMetaTable
from lib.args import parse_cli_args, CliArgs
from lib.version import __version__

logger = logging.getLogger(__name__)


async def main():
    args = parse_cli_args()
    setup_logging(args.logging_level, args.loki_url, args.loki_password, "aare-oraku-service")

    run_ts = datetime.now(UTC)

    # load model and all required accessories into memory
    model_meta, model, scalers = load_model(args.model_path)
    # set params file for read_params to the one that was used when training the model
    set_params_file(model_meta["params_path"])

    async with await init_db_pool(args) as conn_pool:
        metadata_table = ForecastMetaTable(conn_pool)
        await metadata_table.ensure_table_exists()
        await metadata_table.insert_metadata(run_ts, model_meta, args, __version__)

        status = "success"
        error = None
        try:
            # do the hard part :)
            await make_forecast(run_ts, model_meta, model, scalers, conn_pool, args.horizon, args.num_samples)
        except Exception as e:
            # only catches errors during fetching and forecasting, mostly because fetching has external factors.
            # issues with the database or loading the model will only be visible in the app/container logs.
            status = "failure"
            error = str(e)
            logger.exception("Couldn't finish the forecast run.", exc_info=True)

        finished_at = datetime.now(UTC)

        await metadata_table.update_metadata(run_ts, status, error, finished_at)

    logger.info(f"Finished run in {finished_at - run_ts} (+ {run_ts - import_start_ts} imports)")


async def init_db_pool(args: CliArgs) -> AsyncConnectionPool:
    # TODO consolidate with init_db_pool of api into aare-timescale package
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

    return conn_pool


async def make_forecast(
    run_ts: datetime,
    model_meta: AareModel,
    model: GlobalForecastingModel,
    scalers: DataTransformers | None,
    conn_pool: AsyncConnectionPool,
    horizon: int,
    num_samples: int,
):
    """Pull external sources, feed them to the model, and persist the returned forecast."""
    # configure and pull external sources
    sources = SourceRegistry().configure_sources(conn_pool)
    external_data = await load_external_data(sources, run_ts)

    # compile inference data data from internal (influx) and external data
    data = get_inference_data(model_meta["features"], model.extreme_lags, external_data, run_ts)
    data = scale_inference_data(data, scalers)

    # actually make and store forecast with loaded model
    forecast = predict(model, data, scalers.get("series") if scalers else None, horizon, num_samples)
    await persist_forecast(run_ts, forecast, ForecastTable(conn_pool))


async def load_external_data(sources: Sources, run_ts: datetime) -> dict[str, pd.DataFrame]:
    """
    Pull from all registered external sources, persist their data in the database in the intermediate format, then
    prepare the outputs so they're aligned with our models (naming and such) and return that.
    """
    source_names: list[str] = []
    fetch_tasks: list[Awaitable[pd.DataFrame | BaseException]] = []

    for source_name, source in sources.items():
        source_names.append(source_name)
        fetch_tasks.append(_fetch_cache_external(source_name, **source, run_ts=run_ts))

    # fetching external sources and persisting them can all be done concurrently, await all together
    results = await asyncio.gather(*fetch_tasks, return_exceptions=True)

    external_data: dict[str, pd.DataFrame] = {}
    for source_name, result in zip(source_names, results):
        if isinstance(result, BaseException):
            if isinstance(result, BaseExceptionGroup):  # TaskGroup wraps error(s) in exception group
                error = f"{result}: {' | '.join(str(e) for e in result.exceptions)}"
            else:
                error = str(result)

            logger.error(f"Could not fetch and store '{source_name}': {error}", exc_info=result)
            continue

        assert isinstance(result, pd.DataFrame), "result is neither a dataframe nor an exception"
        external_data[source_name] = result

    return external_data


async def _fetch_cache_external(name: str, source: ExternalSource, table: TimescaleTable, run_ts: datetime):
    """Fetch data, cache it in the db, then transform and return it ready for the model."""
    async with asyncio.TaskGroup() as tg:  # similar to asyncio.gather, just nicer syntax for this use-case
        fetch_task = tg.create_task(source.fetch())  # fetch from external source
        tg.create_task(table.ensure_table_exists())  # create table if needed (without waiting for fetch task)

    # task is 100% already finished here, so call result() directly instead of awaiting it
    df = fetch_task.result()

    # Ps. for myself regarding how async works and compares to C#: coroutines in Python are cold tasks, they are only
    # started once something awaits them. In contrast, C# tasks are always hot, so the second you call a function, it will
    # run everything up the first await. In Python, if you just call an async function, nothing inside of it will run.
    # After playing around, this seems to even be true with create_task, but slightly different. create_task schedules
    # the coroutine as the next task and as soon as a context switch occurs, e.g. because some other task is awaited,
    # this scheduled task will be started while the other one is waiting. In this case here, the first context switch
    # after scheduling happens when the async context manager is exited, so then all scheduled tasks are run concurrently.
    # IIRC you could use asyncio.eager_task_factory for a "hot task" like behavior.

    logger.debug(f"Fetched {len(df)} rows from {name}")
    df["run_ts"] = run_ts

    await table.insert(df)
    logger.debug(f"Inserted {len(df)} rows into {table.table_name}")

    prepared = source.prepare(df)

    return prepared


def predict(
    model: GlobalForecastingModel,
    data: InferenceData,
    target_scaler: InvertibleDataTransformer | Pipeline | None,
    horizon: int,
    num_samples: int,
) -> pd.DataFrame:
    """Use the loaded model and fetched data to predict the future temperature."""
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
    """Store the forecast in the timescale database."""
    await table.ensure_table_exists()
    to_store = forecast.copy()
    to_store["run_ts"] = run_ts
    await table.insert(to_store)


if __name__ == "__main__":
    uvloop.run(main())
