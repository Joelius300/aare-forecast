from contextlib import asynccontextmanager
from typing import Optional, Annotated

import psycopg_pool
from fastapi import FastAPI, HTTPException, Depends, Query
from datetime import datetime, UTC

from lib.settings import Settings
from lib.postgres import select_predictions

# noinspection PyArgumentList
settings = Settings()  # pyright: ignore[reportCallIssue]
if settings.default_horizon > settings.maximum_horizon:
    raise ValueError("default_horizon cannot be larger than maximum_horizon")

PREPARE_THRESHOLD = 0  # prepare every query the first time it's executed -> not sure if this works correctly with copy
pool = psycopg_pool.AsyncConnectionPool(
    settings.connection_string,
    open=False,
    min_size=1,
    max_size=4,
    num_workers=1,
    kwargs=dict(prepare_threshold=PREPARE_THRESHOLD),
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


@app.get("/settings")
def get_settings():
    return {}


@app.get("/predictions")
async def get_predictions(
    from_: Annotated[Optional[datetime], Query(alias="from")] = None,
    horizon: Annotated[Optional[int], Query(gt=0, le=settings.maximum_horizon)] = settings.default_horizon,
    conn=Depends(get_conn),
):
    """
    Get the latest predictions made before the specified time, or the most recent predictions if not specified.
    You may optionally specify a horizon if you want determinism or do not want the default.
    """
    if horizon is None:
        horizon = settings.default_horizon

    if horizon > settings.maximum_horizon:
        raise HTTPException(
            400, f"Cannot request a horizon larger than the maximum horizon of {settings.maximum_horizon}"
        )

    now = datetime.now(UTC)
    if from_ is None:
        from_ = now
    elif from_ > now:
        raise HTTPException(
            400,
            "Cannot request predictions made in the future. "
            "If you want predictions for sometime in the future, omit the at ",
        )

    df = await select_predictions(conn, from_, settings.maximum_prediction_age, horizon)
    if df.empty:
        raise HTTPException(
            # todo better status code?
            400,
            f"No predictions available from {from_} (considered ).",
        )

    print(df)
    run_ts_unique = df["run_ts"].unique()
    print(run_ts_unique)
    run_ts = run_ts_unique.item()

    return {
        "time": df["time"],
        "temp_bern": df["temp_bern"],
        "metadata": {
            "run_ts": run_ts,
        },
    }
