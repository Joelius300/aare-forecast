import logging
from contextlib import asynccontextmanager
from typing import Optional, Annotated

import pandas as pd
import psycopg_pool
import pytz
from fastapi import FastAPI, HTTPException, Depends, Query
from datetime import datetime, UTC


from lib.settings import Settings
from lib.postgres import select_forecasts, get_model_info
from lib.dto import ForecastPayload, ForecastMetadata, Config

# pydantic(-settings) doesn't work well with static type checkers. there's a plugin for mypy but not pyright.
# noinspection PyArgumentList
settings = Settings()  # pyright: ignore[reportCallIssue]
if settings.default_horizon > settings.maximum_horizon:
    raise ValueError("default_horizon cannot be larger than maximum_horizon")

logging.basicConfig(level=settings.logging_level)

tz = pytz.timezone(settings.timezone)

PREPARE_THRESHOLD = 0  # prepare every query the first time it's executed -> not sure if this works correctly with copy
pool = psycopg_pool.AsyncConnectionPool(
    settings.connection_string,
    open=False,
    min_size=1,  # keep one open at all times
    max_size=4,
    num_workers=1,  # shouldn't need more workers to manage those connections (big default on min_size, num_workers, ..)
    kwargs=dict(prepare_threshold=PREPARE_THRESHOLD),  # kwargs are passed to the connection
)


# setup lifespan to initialize and cleanup the psycopg connection pool
@asynccontextmanager
async def lifespan(_app: FastAPI):
    await pool.open()
    yield
    await pool.close()


app = FastAPI(lifespan=lifespan)


async def open_db():
    """Open connection in form of a generator to be used with FastAPI DI (Depends) -> open, return(yield), close."""
    async with pool.connection() as conn:
        yield conn


# Yes, using async for non-async methods is better in FastAPI (except if there is blocking IO in the function)
@app.get("/config")
async def get_config() -> Config:
    """Get the config the API is running with. Things like maximum_forecast_age, timezone, etc."""
    return Config(
        timezone=settings.timezone,
        maximum_forecast_age=settings.maximum_forecast_age,
        default_horizon=settings.default_horizon,
        maximum_horizon=settings.maximum_horizon,
    )


@app.get("/")
async def get_index() -> str:
    return "«Bitte anthropomorphisier mi nid, i bi doch nume chli fancy Math u Statistik», seit ds Oraku"


API_DESC = (
    "Get the latest forecasts made before the specified time, or the most recent forecasts if not specified.\n"
    "If the timestamp is specified without a timezone, it is interpreted as the timezone specified in /config "
    f"(currently '{settings.timezone}').\n"
    "You may optionally specify a horizon in hours if you want determinism or do not want the default.\n"
    "'last_updated' is the exact timestamp when the returned forecast was made. It must be between the specified "
    f"time ('from') and {settings.maximum_forecast_age} before that. If no forecast was made in that timeframe, "
    f"an empty response is returned where 'last_updated' is null."
)
FROM_API_DESC = (
    "Timestamp in the format YYYY-MM-DDThh:mm:ssZ. "
    "Use 'Z' for UTC or url-encode the timestamp to use a plus. "
    f"If no timezone is specified, it is interpreted as {settings.timezone}!"
)
HORIZON_API_DESC = "Number of steps (hours) the forecast should contain (24 = one day forecast)"
MODEL_INFO_API_DESC = "Set to true if you want information on the model that was used to make the returned forecast"


@app.get("/forecasts", description=API_DESC)
async def get_forecasts(
    from_: Annotated[Optional[datetime], Query(alias="from", description=FROM_API_DESC)] = None,
    horizon: Annotated[
        int, Query(gt=0, le=settings.maximum_horizon, description=HORIZON_API_DESC)
    ] = settings.default_horizon,
    model_info: Annotated[bool, Query(description=MODEL_INFO_API_DESC)] = False,
    conn=Depends(open_db),
) -> ForecastPayload:
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
            "Cannot request forecasts made in the future. "
            "If you want the latest forecasts (furthest into the future), omit the 'from' parameter.",
        )

    df = await select_forecasts(conn, from_, settings.maximum_forecast_age, horizon)
    if df.empty:
        # instead of an error, return an empty response.
        return ForecastPayload(
            time=[],
            temp_bern=[],
            metadata=ForecastMetadata(
                last_updated=None,
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

    return ForecastPayload(
        time=times.to_list(),
        temp_bern=df["temp_bern"].to_list(),
        metadata=ForecastMetadata(
            last_updated=run_ts,
            model=model,
        ),
    )
