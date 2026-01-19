import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, UTC
from enum import StrEnum
from typing import Annotated

import pytz
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Header, Query, Request, Response, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from psycopg import AsyncConnection

from aare_logging.logging import setup_logging
from lib.access_log_filter import AccessLogFilter
from lib.client_caching import get_last_modified, set_client_caching, response_still_fresh
from lib.dba import fetch_forecast, get_model_info, init_db_pool
from lib.dto import (
    ForecastPayload,
    ForecastMetadata,
    Config,
    ForecastDataFormat,
    ForecastColumnData,
    ForecastRowData,
    Health,
    ModelInfo,
)
from lib.latest_cache import LatestCache
from lib.oraku_settings import OrakuSettings, settings, CityEnum
from lib.routers.dependencies import get_settings, open_db

logger = logging.getLogger(__name__)

router = APIRouter()

valid_variables = ("temperature", "flow")

# TODO will need to be in the global app state as well if also used in health.
#  extract init like this into own module and also make it X loc since we're gonna need that in the future anyway.
#  Health endpoint needs some streamlined method to fetch all of them
#  in parallel, actually maybe it can/should use a custom method to just fetch the max run_ts once per var (table).
#  Redundant but you could also check that the last forecast run was in recent enough and successful, but meh.
# Also, this would directly support setting different caching policies for different variable and cities if needed :)
latest_caches = {
    var: LatestCache(timedelta(seconds=settings.expected_interval_sec), timedelta(seconds=settings.cache_tolerance_sec))
    for var in valid_variables
}


@router.get("/forecast/{variable}", response_model=ForecastPayload)
async def get_forecasts(
    conn: Annotated[AsyncConnection, Depends(open_db)],
    settings: Annotated[OrakuSettings, Depends(get_settings)],
    request: Request,
    response: Response,
    variable: str,
    from_: Annotated[datetime | None, Query(alias="from", description=FROM_API_DESC)] = None,
    horizon: Annotated[
        int, Query(gt=0, le=settings.maximum_horizon, description=HORIZON_API_DESC)
    ] = settings.default_horizon,
    city: CityEnum = CityEnum(settings.default_city),  # cannot disable jetbrains warning here, but it works :)
    model_info: Annotated[bool, Query(description=MODEL_INFO_API_DESC)] = False,
    format: ForecastDataFormat = ForecastDataFormat.COLUMN,
    if_modified_since: Annotated[str | None, Header()] = None,
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
        from_ = from_.astimezone(settings.tz)

    if from_ > now:
        raise HTTPException(
            400,
            "Cannot request forecasts made in the future. "
            + "If you want the latest forecasts (furthest into the future), omit the 'from' parameter.",
        )

    default_latest_cache = latest_caches[variable]

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
        run_ts, df = await fetch_forecast(
            conn, variable, from_, horizon, city, settings.maximum_forecast_age, settings.tz
        )

        logger.debug("[server-side cache] had to fetch in forecast endpoint")

        if ss_cacheable and run_ts is not None:
            logger.debug("[server-side cache] updated cache in forecast endpoint")
            default_latest_cache.update(run_ts, df)

    if run_ts is None:
        # instead of an error, return an empty response.
        return ForecastPayload(
            data=ForecastColumnData() if format == ForecastDataFormat.COLUMN else ForecastRowData.empty(),
            metadata=ForecastMetadata(
                variable=variable,
                city=city,
                format=format,
                last_updated=None,
                model=None,
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

    model = None
    if model_info:
        if variable == "temperature":
            model = await get_model_info(conn, run_ts)
        else:
            assert variable == "flow", "variable neither temperature nor flow"
            # fake model version to show it's an external model, versioned via when it was run I guess...
            model = ModelInfo(name="BAFU-Hochwasser", version=run_ts.date().isoformat())

    if fetching_latest:
        # if the client didn't set a 'from' param, we can use client-side caching. see comments in function.
        # note: our expected_interval is of course only for our own runs, so temperature forecasts, so flow forecasts
        # will be cached less aggressive than we could. To avoid complexity and because the interval is dynamic, KISS.
        set_client_caching(
            response, now, run_ts, default_latest_cache.expected_interval, default_latest_cache.tolerance
        )

    return ForecastPayload(
        # TODO cannot use temp and flow, need to sync dto or have a more generic one e.g. with value and q25, q75, etc.
        data=ForecastColumnData(time=df["time"].to_list(), temp=df["temp"].to_list())
        if format == ForecastDataFormat.COLUMN
        else ForecastRowData.model_validate(df[["time", "temp"]].to_dict(orient="records")),
        metadata=ForecastMetadata(
            variable=variable,
            last_updated=run_ts,
            model=model,
            city=city,
            format=format,
        ),
    )
