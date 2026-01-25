import logging
from datetime import datetime, timedelta, UTC
from typing import Annotated

from fastapi import HTTPException, Depends, Header, Query, Response, APIRouter
from psycopg import AsyncConnection

from lib.client_caching import get_last_modified, set_client_caching, response_still_fresh
from lib.dba import fetch_forecast, fetch_model_info
from lib.dto import (
    ForecastPayload,
    ForecastMetadata,
    ForecastDataFormat,
    ForecastColumnData,
    ForecastRowData,
    ModelInfo,
)
from lib.latest_cache import LatestCache
from lib.oraku_settings import OrakuSettings, settings, CityEnum
from lib.routers.dependencies import get_settings, open_db

logger = logging.getLogger(__name__)

router = APIRouter()

# Mapping from internal variable names to dataframe column names
VARIABLE_COLUMN = {"temperature": "temp", "flow": "flow"}

# Mapping from internal variable names to display names in API response
VARIABLE_DISPLAY = {"temperature": "temp", "flow": "flow"}

# Valid variables for caching (using internal names)
CACHEABLE_VARIABLES = ("temperature", "flow")

# the server side cache is only for the default request, so default city, default horizon and no or very recent 'from'
# TODO will need to be in the global app state as well if also used in health.
#  extract init like this into own module and also make it X loc since we're gonna need that in the future anyway.
#  Health endpoint needs some streamlined method to fetch all of them
#  in parallel, actually maybe it can/should use a custom method to just fetch the max run_ts once per var (table).
#  Redundant but you could also check that the last forecast run was in recent enough and successful, but meh.
# Also, this would directly support setting different caching policies for different variable and cities if needed :)
latest_caches = {
    var: LatestCache(timedelta(seconds=settings.expected_interval_sec), timedelta(seconds=settings.cache_tolerance_sec))
    for var in CACHEABLE_VARIABLES
}

API_DESC = (
    "Get the latest forecasts made before the specified time, or the most recent forecasts if not specified. "
    "If the timestamp is specified without a timezone, it is interpreted as the timezone specified in /config "
    f"(currently '{settings.timezone}'). Unless you truly need it, do not set 'from', it allows for better caching!\n\n"
    "You may optionally specify a horizon in hours if you want determinism or do not want the default. "
    "'last_updated' is the exact timestamp when the returned forecast was made. It must be between the specified "
    f"time ('from') and {settings.maximum_forecast_age} before that. If no forecast was made in that timeframe, "
    f"an empty response is returned where 'last_updated' is null.\n\nFor statistical purposes, "
    "please add &app={your app name} and optionally add &version={your app version} to all of your requests."
)
FROM_API_DESC = (
    "Only set if you want a forecast made before a specific time! "
    "Timestamp in the format YYYY-MM-DDThh:mm:ssZ. "
    "Use 'Z' for UTC or url-encode the timestamp to use a plus. "
    f"If no timezone is specified, it is interpreted as {settings.timezone}!"
)
HORIZON_API_DESC = "Number of steps (hours) the forecast should contain (24 = one day forecast)"
MODEL_INFO_API_DESC = "Set to true if you want information on the model that was used to make the returned forecast"


async def _get_forecast(
    conn: AsyncConnection,
    settings: OrakuSettings,
    response: Response,
    variable_internal: str,
    variable_display: str,
    from_: datetime | None,
    horizon: int,
    city: CityEnum,
    model_info: bool,
    format: ForecastDataFormat,
    if_modified_since: str | None,
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

    default_latest_cache = latest_caches[variable_internal]

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
            conn, variable_internal, from_, horizon, city, settings.maximum_forecast_age, settings.tz
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
                variable=variable_display,
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
        if variable_internal == "temperature":
            model = await fetch_model_info(conn, run_ts)
        else:
            assert variable_internal == "flow", "variable neither temperature nor flow"
            # fake model version to show it's an external model, versioned via when it was run I guess...
            model = ModelInfo(name="BAFU-Hochwasser", version=run_ts.date().isoformat())

    if fetching_latest:
        # if the client didn't set a 'from' param, we can use client-side caching. see comments in function.
        # note: our expected_interval is of course only for our own runs (temperature forecasts), so flow forecasts
        # will be cached less aggressive than we could. To avoid complexity and because the interval is dynamic, KISS.
        set_client_caching(
            response, now, run_ts, default_latest_cache.expected_interval, default_latest_cache.tolerance
        )

    # Get the correct column name for the variable
    column_name = VARIABLE_COLUMN[variable_internal]

    return ForecastPayload(
        data=ForecastColumnData(time=df["time"].to_list(), value=df[column_name].to_list())
        if format == ForecastDataFormat.COLUMN
        else ForecastRowData.model_validate(
            [
                {"time": row["time"], "value": row[column_name]}
                for row in df[["time", column_name]].to_dict(orient="records")
            ]
        ),
        metadata=ForecastMetadata(
            variable=variable_display,
            last_updated=run_ts,
            model=model,
            city=city,
            format=format,
        ),
    )


# Temperature forecast endpoints - all three return variable: "temp"
@router.get("/forecast", description=API_DESC, response_model=ForecastPayload)
@router.get("/forecast/temp", description=API_DESC, response_model=ForecastPayload)
@router.get("/forecast/temperature", description=API_DESC, response_model=ForecastPayload)
async def get_temp_forecasts(
    conn: Annotated[AsyncConnection, Depends(open_db)],
    settings: Annotated[OrakuSettings, Depends(get_settings)],
    response: Response,
    from_: Annotated[datetime | None, Query(alias="from", description=FROM_API_DESC)] = None,
    horizon: Annotated[
        int, Query(gt=0, le=settings.maximum_horizon, description=HORIZON_API_DESC)
    ] = settings.default_horizon,
    city: CityEnum = CityEnum(settings.default_city),
    model_info: Annotated[bool, Query(description=MODEL_INFO_API_DESC)] = False,
    format: ForecastDataFormat = ForecastDataFormat.COLUMN,
    if_modified_since: Annotated[str | None, Header()] = None,
) -> ForecastPayload | Response:
    return await _get_forecast(
        conn=conn,
        settings=settings,
        response=response,
        variable_internal="temperature",
        variable_display="temp",
        from_=from_,
        horizon=horizon,
        city=city,
        model_info=model_info,
        format=format,
        if_modified_since=if_modified_since,
    )


# Flow forecast endpoint
@router.get("/forecast/flow", description=API_DESC, response_model=ForecastPayload)
async def get_flow_forecasts(
    conn: Annotated[AsyncConnection, Depends(open_db)],
    settings: Annotated[OrakuSettings, Depends(get_settings)],
    response: Response,
    from_: Annotated[datetime | None, Query(alias="from", description=FROM_API_DESC)] = None,
    horizon: Annotated[
        int, Query(gt=0, le=settings.maximum_horizon, description=HORIZON_API_DESC)
    ] = settings.default_horizon,
    city: CityEnum = CityEnum(settings.default_city),
    model_info: Annotated[bool, Query(description=MODEL_INFO_API_DESC)] = False,
    format: ForecastDataFormat = ForecastDataFormat.COLUMN,
    if_modified_since: Annotated[str | None, Header()] = None,
) -> ForecastPayload | Response:
    return await _get_forecast(
        conn=conn,
        settings=settings,
        response=response,
        variable_internal="flow",
        variable_display="flow",
        from_=from_,
        horizon=horizon,
        city=city,
        model_info=model_info,
        format=format,
        if_modified_since=if_modified_since,
    )
