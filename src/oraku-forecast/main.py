# ruff: noqa: E402
# ignore 'imports not at top of file' for this file

from datetime import UTC, datetime
from pathlib import Path

from psycopg import sql


# run duration of the service should ignore the time needed to import packages, but we still want to log it
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
from psycopg_pool import AsyncConnectionPool

from aare_logging.logging import setup_logging
from aare_timescale.timescale_table import TimescaleTable
from aare_timescale.postgres import init_db_pool
from aare_train.compat.types import DataTransformers
from aare_train.storage.metadata import AareModel
from aare_train.storage.model import load_model
from aare_train.params import set_params_file
from oraku_forecast.data.compile_data import get_inference_data, scale_inference_data
from oraku_forecast.data.inference_data import InferenceData
from oraku_forecast.external_sources.external_source import ExternalSource
from oraku_forecast.external_sources.registry import SourceRegistry, Sources
from oraku_forecast.persistence.tables.forecast import ForecastTable
from oraku_forecast.persistence.tables.forecast_meta import ForecastMetaTable
from oraku_forecast.args import parse_cli_args, CliArgs
from oraku_forecast.version import __version__

logger = logging.getLogger(__name__)


async def main_wrapped():
    args = parse_cli_args()
    setup_logging(args.logging_level, args.loki_url, args.loki_password, "aare-oraku-forecast")

    try:
        await main_unwrapped(args)
    except Exception as e:
        # unless loki is the problem, log the error so it's easier to see what failed
        try:
            logger.error("Forecast run failed catastrophically", exc_info=e)
        except:  # noqa: E722
            pass

        # reraise for the app/container to crash
        raise


async def main_unwrapped(args: CliArgs):
    simulation = False
    now = datetime.now(UTC)
    if args.simulate_runts:
        simulation = True
        logger.info("Simulating runs using existing external (=covariate) data.")

    # load model and all required accessories into memory
    model_meta, model, scalers = load_model(args.model_path)
    # set params file for read_params to the one that was used when training the model
    set_params_file(model_meta["params_path"])
    logger.info(f"Loaded model, scalers and params from '{args.model_path}'")

    async with init_db_pool(args.connection_string) as conn_pool:
        # initialize metadata store on db if not in a simulation. could be extended to log simulation metadata too.
        metadata_table = ForecastMetaTable(conn_pool) if not simulation else None
        if metadata_table is not None:
            await metadata_table.ensure_table_exists()

        if not simulation:
            run_ts = now
            # insert partial run metadata record
            if metadata_table is not None:
                await metadata_table.insert_metadata(run_ts, model_meta, args, __version__)

            # create forecast and catch errors (should not raise)
            forecast, error, status = await forecast_once(
                args, conn_pool, model, model_meta, run_ts, scalers, use_cached_external=False
            )

            # persist forecast if successful
            if forecast is not None:
                assert error is None
                await persist_forecast_db(forecast, ForecastTable(conn_pool))

            # complete run metadata with finish time and success status
            finished_at = datetime.now(UTC)
            if metadata_table is not None:
                await metadata_table.update_metadata(run_ts, status, error, finished_at)
        else:
            assert args.simulate_runts, "simulation without simulated run_ts??"
            forecasts: list[pd.DataFrame] = []
            for run_ts in args.simulate_runts:
                # use cached covariates for simulating forecast
                forecast, error, _ = await forecast_once(
                    args, conn_pool, model, model_meta, run_ts, scalers, use_cached_external=True
                )
                if forecast is not None:
                    assert error is None
                    forecasts.append(forecast)
                else:
                    logger.warning(f"Could not simulate forecast for run_ts '{run_ts}': {error}")

            # combine all forecasts into a single table
            if not forecasts:
                raise ValueError("Could not simulate a single of the provided run_ts.")

            forecast = pd.concat(forecasts)

            # persist to a single parquet file
            persist_forecast_file(now, forecast, model_meta)

    logger.info(f"Finished run in {datetime.now(UTC) - now} (+ {now - import_start_ts} imports)")


async def forecast_once(
    args: CliArgs,
    conn_pool: AsyncConnectionPool,
    model: GlobalForecastingModel,
    model_meta: AareModel,
    run_ts: datetime,
    scalers: DataTransformers | None,
    use_cached_external: bool,
):
    status = "success"
    error = None
    forecast = None

    try:
        # calculate forecast with loaded model
        forecast = await make_forecast(
            run_ts,
            model_meta,
            model,
            scalers,
            conn_pool,
            args.horizon,
            args.num_samples,
            use_cached_external,
        )

        forecast["run_ts"] = run_ts
    except Exception as e:
        # only catches errors during fetching and forecasting, mostly because fetching has external factors.
        # issues with the database or loading the model will only be visible in the app/container logs.
        # the error here is intended to be stored in the model run metadata.
        status = "failure"
        error = str(e)
        logger.exception("Couldn't finish the forecast run.", exc_info=True)

    return forecast, error, status


async def make_forecast(
    run_ts: datetime,
    model_meta: AareModel,
    model: GlobalForecastingModel,
    scalers: DataTransformers | None,
    conn_pool: AsyncConnectionPool,
    horizon: int,
    num_samples: int,
    use_cached_external: bool,
):
    """Pull external sources, feed them to the model, and return forecast."""
    # configure and pull external sources
    sources = SourceRegistry.configure_sources(conn_pool)
    external_data = await load_external_data(sources, run_ts, use_cached_external)

    # compile inference data from internal (influx) and external data
    data = get_inference_data(model_meta["features"], model.extreme_lags, external_data, run_ts)
    data = scale_inference_data(data, scalers)

    # actually make forecast with loaded model
    return predict(model, data, scalers.get("series") if scalers else None, horizon, num_samples)


async def load_external_data(sources: Sources, run_ts: datetime, use_cached: bool) -> dict[str, pd.DataFrame]:
    """
    Pull from all registered external sources, persist their data in the database in the intermediate format, then
    prepare the outputs so they're aligned with our models (naming and such) and return that.
    """
    fetch_tasks: list[Awaitable[pd.DataFrame | BaseException]] = []

    for source_name, source in sources.items():
        if not use_cached:
            fetch_tasks.append(_fetch_and_cache_external(source_name, **source, run_ts=run_ts))
        else:
            fetch_tasks.append(_fetch_external_from_cache(source["table"], run_ts))

    # fetching external sources and persisting them can all be done concurrently, await all together
    results = await asyncio.gather(*fetch_tasks, return_exceptions=True)

    external_data: dict[str, pd.DataFrame] = {}
    for (source_name, source), result in zip(sources.items(), results):
        if isinstance(result, BaseException):
            if isinstance(result, BaseExceptionGroup):  # TaskGroup wraps error(s) in exception group
                error = f"{result}: {' | '.join(str(e) for e in result.exceptions)}"
            else:
                error = str(result)

            logger.error(f"Could not fetch and store '{source_name}': {error}", exc_info=result)
            continue

        assert isinstance(result, pd.DataFrame), "result is neither a dataframe nor an exception"

        # prepare external data before returning it
        prepared = source["source"].prepare(result)
        external_data[source_name] = prepared

    return external_data


async def _fetch_and_cache_external(name: str, source: ExternalSource, table: TimescaleTable, run_ts: datetime):
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

    return df


async def _fetch_external_from_cache(table: TimescaleTable, run_ts: datetime):
    # currently has a 1 sec tolerance on the run_ts. sweet spot between convenience (not specifying milliseconds)
    # and accuracy (runs are never started this close to each other so collision shouldn't happen).
    return await table.select(
        # postgres does not allow abs() on type INTERVAL so use between
        where=sql.SQL("""("run_ts" - {}) BETWEEN interval '-1 second' AND interval '1 second'""").format(run_ts)
    )


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


async def persist_forecast_db(forecast: pd.DataFrame, table: TimescaleTable):
    """Store the forecast in the timescale database."""
    await table.ensure_table_exists()
    await table.insert(forecast)


def persist_forecast_file(actual_now: datetime, forecast: pd.DataFrame, model_meta: AareModel):
    dir = Path("simulated_runs")
    dir.mkdir(parents=True, exist_ok=True)
    forecast.to_parquet(
        dir
        / (
            f"{actual_now.astimezone().replace(microsecond=0, tzinfo=None).isoformat()}"
            "_"
            f"{model_meta['name']}-{model_meta['version']}.parquet"
        ),
        index=False,
    )


if __name__ == "__main__":
    uvloop.run(main_wrapped())
