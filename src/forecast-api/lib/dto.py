from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class ModelInfo(BaseModel):
    name: str
    version: str


class ForecastMetadata(BaseModel):
    last_updated: Optional[datetime]
    """Exact time the forecast was made (=last updated) or null if no forecast was returned."""
    model: Optional[ModelInfo]
    city: str


class ForecastPayload(BaseModel):
    time: list[datetime]
    temp: list[float]
    metadata: ForecastMetadata


class Config(BaseModel):
    timezone: str
    maximum_forecast_age: str
    """Amount of time to look back when searching for a forecast at a given time = maximum age"""
    default_horizon: int
    maximum_horizon: int
    available_cities: list[str]
