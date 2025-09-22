from contextlib import asynccontextmanager
from typing import Optional

import psycopg_pool
from fastapi import FastAPI, HTTPException, Depends
from datetime import datetime, UTC

from lib.settings import Settings

# noinspection PyArgumentList
settings = Settings()  # pyright: ignore[reportCallIssue]
if settings.default_horizon > settings.maximum_horizon:
    raise ValueError("default_horizon cannot be larger than maximum_horizon")

pool = psycopg_pool.AsyncConnectionPool(settings.connection_string, open=False, min_size=1, max_size=4, num_workers=1)

@asynccontextmanager
async def lifespan(_app: FastAPI):
    await pool.open()
    yield
    await pool.close()

app = FastAPI(lifespan=lifespan)

async def get_conn():
    async with pool.connection() as conn:
        yield conn

@app.get("/predictions")
async def get_predictions(at: Optional[datetime] = None, horizon: Optional[int] = None, conn = Depends(get_conn)):
    if horizon is None:
        horizon = settings.default_horizon

    if horizon > settings.maximum_horizon:
        raise HTTPException(400, f"Cannot request a horizon larger than the maximum horizon of {settings.maximum_horizon}")

    if at is None:
        at = datetime.now(UTC)

    return {
        "time": [at, at, at],
        "temp_bern": [1, 2, 3],
        "metadata": {
            "connection_string": settings.connection_string,
        }
    }
