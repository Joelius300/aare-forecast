import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, UTC
from enum import StrEnum
from typing import Annotated

import psycopg_pool
import pytz
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Header, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from psycopg import AsyncConnection

from aare_logging.logging import setup_logging
from lib.access_log_filter import AccessLogFilter
from lib.client_caching import get_last_modified, set_client_caching, response_still_fresh
from lib.dba import fetch_forecast, get_model_info
from lib.dto import (
    ForecastPayload,
    ForecastMetadata,
    Config,
    ForecastDataFormat,
    ForecastColumnData,
    ForecastRowData,
    Health,
)
from lib.latest_cache import LatestCache
from lib.oraku_settings import OrakuSettings
from fastapi import APIRouter

router = APIRouter()

@router.get("/forecast/{variable}", response_model=ForecastPayload)
async def get_forecasts(
        conn: Annotated[AsyncConnection, Depends(db_opener())],
        response: Response,
        if_modified_since: Annotated[str | None, Header()] = None,
        from_: Annotated[datetime | None, Query(alias="from", description=FROM_API_DESC)] = None,
        horizon: Annotated[
            int, Query(gt=0, le=settings.maximum_horizon, description=HORIZON_API_DESC)
        ] = settings.default_horizon,
        city: CityEnum = CityEnum(settings.default_city),  # cannot disable jetbrains warning here, but it works :)
        model_info: Annotated[bool, Query(description=MODEL_INFO_API_DESC)] = False,
        format: ForecastDataFormat = ForecastDataFormat.COLUMN,
):
    pass


