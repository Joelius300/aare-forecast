import logging
from contextlib import asynccontextmanager
from typing import Optional, Annotated

import pandas as pd
import psycopg_pool
import pytz
from fastapi import FastAPI, HTTPException, Depends, Query
from datetime import datetime, UTC


from lib.settings import Settings
from lib.postgres import select_predictions, get_model_info
from lib.dto import PredictionPayload, PredictionMetadata

# pydantic(-settings) doesn't work well with static type checkers. there's a plugin for mypy but not pyright.
# noinspection PyArgumentList
settings = Settings()  # pyright: ignore[reportCallIssue]
if settings.default_horizon > settings.maximum_horizon:
    raise ValueError("default_horizon cannot be larger than maximum_horizon")

logging.basicConfig(level=settings.logging_level)

tz = pytz.timezone(settings.time_zone)

PREPARE_THRESHOLD = 0  # prepare every query the first time it's executed -> not sure if this works correctly with copy
pool = psycopg_pool.AsyncConnectionPool(
    settings.connection_string,
    open=False,
    min_size=1,  # keep one open at all times
    max_size=4,
    num_workers=1,  # shouldn't need more workers to manage those connections (default is on min_size, num_workers, etc.)
    kwargs=dict(prepare_threshold=PREPARE_THRESHOLD),  # kwargs are passed to the connection
)


# setup lifespan to initialize and cleanup the psycopg connection pool
@asynccontextmanager
async def lifespan(_app: FastAPI):
    await pool.open()
    yield
    await pool.close()


app = FastAPI(lifespan=lifespan)


async def get_conn():
    """Open connection in form of a generator to be used with FastAPI DI (Depends) -> open, return(yield), close."""
    async with pool.connection() as conn:
        yield conn


@app.get("/config")
def get_config():
    """Get the config the API is running with. Things like maximum_prediction_age, timezone, etc."""
    # TODO implement
    return {}


API_DESC = (
    "Get the latest predictions made before the specified time, or the most recent predictions if not specified.\n"
    "If the timestamp is specified without a timezone, it is interpreted as the timezone specified in /config "
    f"(currently '{settings.time_zone}').\n"
    "You may optionally specify a horizon if you want determinism or do not want the default.\n"
)
FROM_API_DESC = (
    "Timestamp in the format YYYY-MM-DDThh:mm:ssZ. "
    "Use 'Z' for UTC or url-encode the timestamp to use a plus. "
    f"If no timezone is specified, it is interpreted as {settings.time_zone}!"
)


@app.get("/predictions", description=API_DESC)
async def get_predictions(
    from_: Annotated[Optional[datetime], Query(alias="from", description=FROM_API_DESC)] = None,
    horizon: Annotated[Optional[int], Query(gt=0, le=settings.maximum_horizon)] = settings.default_horizon,
    model_info=False,
    conn=Depends(get_conn),
) -> PredictionPayload:
    if horizon is None:
        horizon = settings.default_horizon

    if horizon > settings.maximum_horizon:
        raise HTTPException(
            400, f"Cannot request a horizon larger than the maximum horizon of {settings.maximum_horizon}"
        )

    now = datetime.now(UTC)
    if from_ is None:
        from_ = now
    elif from_.tzinfo is None:
        from_ = from_.astimezone(tz)

    if from_ > now:
        raise HTTPException(
            400,
            "Cannot request predictions made in the future. "
            "If you want the latest predictions (furthest into the future), omit the 'from' parameter.",
        )

    df = await select_predictions(conn, from_, settings.maximum_prediction_age, horizon)
    if df.empty:
        # instead of an error, return an empty response.
        return PredictionPayload(
            time=[],
            temp_bern=[],
            metadata=PredictionMetadata(
                run_ts=None,
                model=None,
            ),
        )

    run_ts_unique = df["run_ts"].unique()
    assert len(run_ts_unique) == 1, "somehow fetched more than one run, currently not supported so it shouldn't happen"
    run_ts = run_ts_unique.item()
    run_ts = datetime.fromisoformat(run_ts).astimezone(tz)

    model = await get_model_info(conn, run_ts) if model_info else None

    # if speed is important, it's probably faster to use .dt.strftime() to convert pd.Timestamp directly to str
    times = pd.to_datetime(df["time"]).dt.tz_convert(tz).apply(pd.Timestamp.to_pydatetime)

    return PredictionPayload(
        time=times.to_list(),
        temp_bern=df["temp_bern"].to_list(),
        metadata=PredictionMetadata(
            run_ts=run_ts,
            model=model,
        ),
    )
