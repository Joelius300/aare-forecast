import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from aare_logging.logging import setup_logging
from oraku_api.access_log_filter import AccessLogFilter
from oraku_api.dba import init_db_pool
from oraku_api.oraku_settings import settings
from oraku_api.routers.boilerplate import router as boilerplate_router
from oraku_api.routers.forecast import router as forecast_router
from oraku_api.routers.health import router as health_router
from oraku_api.server_caching import init_latest_caches
from oraku_api.version import __version__

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


# setup lifespan to initialize and cleanup the psycopg connection pool. plus expose deps/appstate.
@asynccontextmanager
async def lifespan(_app: FastAPI):
    db_pool = init_db_pool(settings.connection_string)
    latest_caches = init_latest_caches(settings)

    await db_pool.open()

    yield {
        "db_pool": db_pool,
        "settings": settings,
        "latest_caches": latest_caches,
    }

    await db_pool.close()


app = FastAPI(
    lifespan=lifespan,
    title="Aare Oraku",
    description="""
API to get forecasts for the Swiss river Aare, intended for use in [aare.guru](https://aare.guru).
Still in early stages, so don't expect perfect accuracy.
The endpoints may also change at any time, there is no guarantee on backwards compatibility.

If you call the API from your own app/site, please add &app={your app name} 
and optionally &version={your app version} to all of your requests. Note the usage/licensing restriction below.

This API may only be used for personal and educational projects. Neither the API endpoints nor 
the data provided by the API may be used commercially. Please contact us for further information.

[Source code](https://github.com/Joelius300/aare-forecast) (License: AGPLv3)
""",
    version=__version__,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=86400,  # 24h
)

app.include_router(boilerplate_router)
app.include_router(forecast_router)
app.include_router(health_router)


if __name__ == "__main__":
    # mostly for debugging, run via uvicorn cli in production
    uvicorn.run(app, host="0.0.0.0", port=8080)
