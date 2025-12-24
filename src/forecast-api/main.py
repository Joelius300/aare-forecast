import logging
from contextlib import asynccontextmanager
from enum import StrEnum
from typing import Annotated

import pandas as pd
import uvicorn
from psycopg import AsyncConnection
import psycopg_pool
import pytz
from fastapi import FastAPI, HTTPException, Depends, Header, Query, Response
from datetime import datetime, timedelta, UTC

from aare.logging import setup_logging
from lib.client_caching import get_last_modified, set_client_caching, response_still_fresh
from lib.latest_cache import LatestCache
from lib.oraku_settings import OrakuSettings
from lib.postgres import select_forecasts, get_model_info
from lib.dto import (
    ForecastPayload,
    ForecastMetadata,
    Config,
    ForecastDataFormat,
    ForecastColumnData,
    ForecastRowData,
    Health,
)

# pydantic(-settings) doesn't work well with static type checkers. there's a plugin for mypy but not pyright.
# noinspection PyArgumentList
settings = OrakuSettings()  # pyright: ignore[reportCallIssue]

setup_logging(
    settings.logging_level,
    settings.loki_url,
    settings.loki_password,
    "aare-oraku-api",
    # TODO think about these again, at least the healthchecks probably shouldn't be in this.
    ["uvicorn.access", "uvicorn.error"],
)

logger = logging.getLogger(__name__)

# dynamically create enum from specified supported cities. enums give automatic input validation and nicer swagger docs.
CityEnum = StrEnum("CityEnum", settings.available_cities)

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


# the server side cache is only for the default request, so default city, default horizon and no or very recent 'from'
default_latest_cache = LatestCache(
    timedelta(seconds=settings.expected_interval_sec), timedelta(seconds=settings.cache_tolerance_sec)
)


async def fetch_forecast(
    conn: AsyncConnection, from_: datetime, horizon: int, city: CityEnum | str
) -> tuple[datetime | None, pd.DataFrame]:
    df = await select_forecasts(conn, from_, settings.maximum_forecast_age, horizon, city)
    if df.empty:
        return None, df

    # if speed is important, it's probably faster to use .dt.strftime() to convert pd.Timestamp directly to str
    df["time"] = pd.to_datetime(df["time"]).dt.tz_convert(tz).apply(pd.Timestamp.to_pydatetime)

    run_ts_unique = df["run_ts"].unique()
    assert len(run_ts_unique) == 1, "fetched more than one run, currently not supported so it shouldn't happen"
    run_ts = run_ts_unique.item()
    run_ts = datetime.fromisoformat(run_ts).astimezone(tz)

    return run_ts, df


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


@app.get("/forecast", description=API_DESC, response_model=ForecastPayload)
async def get_forecasts(
    conn: Annotated[AsyncConnection, Depends(open_db)],
    response: Response,
    if_modified_since: Annotated[str | None, Header()] = None,
    from_: Annotated[datetime | None, Query(alias="from", description=FROM_API_DESC)] = None,
    horizon: Annotated[
        int, Query(gt=0, le=settings.maximum_horizon, description=HORIZON_API_DESC)
    ] = settings.default_horizon,
    city: CityEnum = CityEnum(settings.default_city),  # cannot disable jetbrains warning here, but it works :)
    model_info: Annotated[bool, Query(description=MODEL_INFO_API_DESC)] = False,
    format: ForecastDataFormat = ForecastDataFormat.COLUMN,
) -> ForecastPayload | Response:
    if horizon > settings.maximum_horizon:
        raise HTTPException(
            400, f"Cannot request a horizon larger than the maximum horizon of {settings.maximum_horizon}"
        )

    fetching_latest = False
    now = datetime.now(UTC)
    if from_ is None:
        from_ = now
        fetching_latest = True
    elif from_.tzinfo is None:
        from_ = from_.astimezone(tz)

    if from_ > now:
        raise HTTPException(
            400,
            "Cannot request forecasts made in the future. "
            + "If you want the latest forecasts (furthest into the future), omit the 'from' parameter.",
        )

    # it's a cacheable request:
    # if no 'from' was passed at all (fetching latest),
    # or it's a time very close to now (less than configured cache tolerance)
    # BUT in that case the requested time must be later than the last cache update,
    # otherwise we would return data that is too new.
    ss_cacheable = fetching_latest or (
        from_ > now - default_latest_cache.tolerance
        and (default_latest_cache.last_updated is None or from_ >= default_latest_cache.last_updated)
    )

    # additionally, the cache must only be used when the request is all default parameters!
    # it would be possible to support all horizons shorter than what's stored in the cache, but prob not worth it.
    ss_cacheable = ss_cacheable and horizon == settings.default_horizon and city == settings.default_city

    if ss_cacheable and default_latest_cache.fresh:
        run_ts, df = default_latest_cache.last_updated, default_latest_cache.data
        assert df is not None and run_ts is not None, "df or run_ts were None from cache!"
        logger.debug("[server-side cache] hit cache in forecast endpoint")
    else:
        run_ts, df = await fetch_forecast(conn, from_, horizon, city)
        logger.debug("[server-side cache] had to fetch in forecast endpoint")

        if ss_cacheable and run_ts is not None:
            logger.debug("[server-side cache] updated cache in forecast endpoint")
            default_latest_cache.update(run_ts, df)

    if run_ts is None:
        # instead of an error, return an empty response.
        return ForecastPayload(
            data=ForecastColumnData() if format == ForecastDataFormat.COLUMN else ForecastRowData(),
            metadata=ForecastMetadata(
                last_updated=None,
                model=None,
                city=city,
                format=format,
            ),
        )

    if response_still_fresh(if_modified_since, run_ts):
        # Not Modified, saves bandwidth, but we still had to fetch the data.
        # Note, if we didn't set no-cache here, it would assume that the data was REFRESHED and the max-age from the
        # original 200 response is active again. We don't want that, it should revalidate everytime after we send a 304
        # until new data arrives and the next 200 response sets the max-age again.
        return Response(
            status_code=304, headers={"Cache-Control": "no-cache", "Last-Modified": get_last_modified(run_ts)}
        )

    model = await get_model_info(conn, run_ts) if model_info else None

    if fetching_latest:
        # if the client didn't set a 'from' param, we can use client-side caching. see comments in function.
        set_client_caching(
            response, now, run_ts, default_latest_cache.expected_interval, default_latest_cache.tolerance
        )

    return ForecastPayload(
        data=ForecastColumnData(time=df["time"].to_list(), temp=df["temp"].to_list())
        if format == ForecastDataFormat.COLUMN
        else ForecastRowData.model_validate(df[["time", "temp"]].to_dict(orient="records")),
        metadata=ForecastMetadata(
            last_updated=run_ts,
            model=model,
            city=city,
            format=format,
        ),
    )


# Yes, using async for non-async methods is better in FastAPI (except if there is blocking IO in the function)
@app.get("/config")
async def get_config() -> Config:
    """Get the config the API is running with. Things like maximum_forecast_age, timezone, etc."""
    return Config(
        timezone=settings.timezone,
        maximum_forecast_age=settings.maximum_forecast_age,
        default_horizon=settings.default_horizon,
        maximum_horizon=settings.maximum_horizon,
        available_cities=settings.available_cities,
    )


@app.get("/")
async def get_index() -> str:
    return "«Bitte anthropomorphisier mi nid, i bi doch nume chli fancy Math u Statistik», seit ds Oraku"


@app.get("/health")
async def health() -> Health:
    # in the best case, the cache is still fresh, and we're sure (enough) that we're up to date.
    # this needs to be revisited once more than one location is supported.
    if default_latest_cache.fresh:
        logger.debug("[server-side cache] hit cache in health endpoint")
        return Health(status="OK", age=int(default_latest_cache.age.total_seconds()))

    # if the cache is stale, we need to fetch from the database.
    # in case the database has issues, this will cause an exception and the health endpoint will return a 500 status.
    async with pool.connection() as conn:
        last_updated, latest_df = await fetch_forecast(
            conn, datetime.now(UTC), settings.default_horizon, city=settings.default_city
        )

    logger.debug("[server-side cache] had to fetch in health endpoint")

    # if we get no data at all when fetching with from == now, we're in deep trouble
    if last_updated is None:
        return Health(status="NOK", age=9999999)

    # if we got the latest data, update the cache for the next health check (or forecast request)
    default_latest_cache.update(last_updated, latest_df)

    # in the good case, new data was fetched, is now ready/fresh in the cache and everything is ok.
    # in the bad case, the same data was fetched that is already in the stale cache.
    # in the very bad case, this stale data is so old that it triggers an 'unhealthy' response.
    # if it's in between, the service cannot make use of the cache because it's considered stale, but
    # it's not yet old enough for the system to be considered unhealthy and it might recover.
    # in theory, it's also possible to fetch data that's newer than the one in the stale cache, but still so old that
    # the unhealthy state triggers. But I think this can only happen if the health endpoint is called very rarely
    # or after a restart (then cache is always stale), and it's also not really a problem, just wanted to mention it.
    # As an additional sidenote, even if the system is in an 'unhealthy' state, the forecast endpoint will continue
    # to return the latest data until the configured maximum forecast age is reached, then it will return empty.

    age_sec = int(default_latest_cache.age.total_seconds())
    if age_sec < settings.unhealthy_age_sec:
        return Health(status="OK", age=age_sec)

    return Health(status="NOK", age=age_sec)


if __name__ == "__main__":
    # mostly for debugging, run via uvicorn cli in production
    uvicorn.run(app, host="0.0.0.0", port=8080)
