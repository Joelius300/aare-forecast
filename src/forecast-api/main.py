import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from aare_logging.logging import setup_logging
from lib.access_log_filter import AccessLogFilter
from lib.dba import init_db_pool
from lib.dto import Config
from lib.oraku_settings import settings
from lib.routers.forecast import router as forecast_router
from lib.routers.health import router as health_router

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

app.include_router(forecast_router)
app.include_router(health_router)


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


if __name__ == "__main__":
    # mostly for debugging, run via uvicorn cli in production
    uvicorn.run(app, host="0.0.0.0", port=8080)
