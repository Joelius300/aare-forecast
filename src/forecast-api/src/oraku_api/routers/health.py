import logging
from contextlib import asynccontextmanager
from datetime import datetime, UTC
from typing import Annotated

from fastapi import APIRouter, Depends, Request, Response

from oraku_api.dba import fetch_forecast
from oraku_api.dto import Health
from oraku_api.oraku_settings import settings
from oraku_api.routers.dependencies import open_db, get_caches
from oraku_api.server_caching import CachesType

logger = logging.getLogger(__name__)

router = APIRouter()


@router.head("/health")  # uptimerobot sends head by default
@router.get("/health")
async def get_health(
    request: Request, response: Response, latest_caches: Annotated[CachesType, Depends(get_caches)]
) -> Health:
    BAD_STATUS = 500
    # use the temperature cache for health checks since we want to ensure service health. flow has lower priority.
    variable = "temp"
    default_latest_cache = latest_caches[variable]

    # in the best case, the cache is still fresh, and we're sure (enough) that we're up to date.
    # this needs to be revisited once more than one location is supported.
    if default_latest_cache.fresh:
        logger.debug("[server-side cache] hit cache in health endpoint")
        return Health(status="OK", age=int(default_latest_cache.age.total_seconds()))

    # if the cache is stale, we need to fetch from the database.
    # in case the database has issues, this will cause an exception and the health endpoint will return a 500 status.
    # manually invoking open_db (wrapped in asynccontextmanager for convenience) avoids cost in fast past (cache hit).
    async with asynccontextmanager(open_db)(request) as conn:
        last_updated, latest_df = await fetch_forecast(
            conn,
            variable,
            datetime.now(UTC),
            settings.default_horizon,
            settings.default_city,
            settings.maximum_forecast_age,
            settings.tz,
        )

    logger.debug("[server-side cache] had to fetch in health endpoint")

    # if we get no data at all when fetching with from == now, we're in deep trouble
    if last_updated is None:
        response.status_code = BAD_STATUS
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

    # redundant ig, but you could also check that the last forecast run was recent enough and successful.
    response.status_code = BAD_STATUS
    return Health(status="NOK", age=age_sec)
