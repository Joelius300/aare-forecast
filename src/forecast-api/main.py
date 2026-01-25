import logging
from contextlib import asynccontextmanager
from datetime import datetime, UTC

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from aare_logging.logging import setup_logging
from lib.access_log_filter import AccessLogFilter
from lib.dba import fetch_forecast, init_db_pool
from lib.dto import Config, Health
from lib.oraku_settings import settings
from lib.routers.dependencies import open_db
from lib.routers.forecast import latest_caches, router as forecast_router

setup_logging(
    settings.logging_level,
    settings.loki_url,
    settings.loki_password,
    "aare-oraku-api",
    ["uvicorn.access", "uvicorn.error"],
)

# ignore /health endpoint in access logs
logging.getLogger("uvicorn.access").addFilter(AccessLogFilter())

logger = logging.getLogger(__name__)


# setup lifespan to initialize and cleanup the psycopg connection pool
@asynccontextmanager
async def lifespan(app: FastAPI):
    # setup, store and open db pool
    db_pool = init_db_pool(settings.connection_string)

    await db_pool.open()

    yield {
        "db_pool": db_pool,
        "settings": settings,
    }

    await db_pool.close()


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=86400,  # 24h
)

# Mount the forecast router
app.include_router(forecast_router)


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


@app.head("/health")  # uptimerobot sends head by default
@app.get("/health")
async def get_health(request: Request, response: Response) -> Health:
    BAD_STATUS = 500
    # Use the temperature cache for health checks (as before)
    default_latest_cache = latest_caches["temperature"]

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
            "temperature",
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

    response.status_code = BAD_STATUS
    return Health(status="NOK", age=age_sec)


if __name__ == "__main__":
    # mostly for debugging, run via uvicorn cli in production
    uvicorn.run(app, host="0.0.0.0", port=8080)
