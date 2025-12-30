from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, RootModel


class ModelInfo(BaseModel):
    name: str
    version: str


class ForecastDataFormat(StrEnum):
    ROW = "row"
    COLUMN = "column"


class ForecastMetadata(BaseModel):
    last_updated: datetime | None
    """Exact time the forecast was made (=last updated) or null if no forecast was returned."""
    model: ModelInfo | None
    city: str
    format: ForecastDataFormat


class ForecastSingleRow(BaseModel):
    time: datetime
    temp: float


class ForecastColumnData(BaseModel):
    time: list[datetime] = []
    temp: list[float] = []


class ForecastRowData(RootModel[list[ForecastSingleRow]]):
    @staticmethod
    def empty():
        # noinspection PyArgumentList
        return ForecastRowData([])


class ForecastPayload(BaseModel):
    data: ForecastColumnData | ForecastRowData
    metadata: ForecastMetadata


class Config(BaseModel):
    timezone: str
    maximum_forecast_age: str
    """Amount of time to look back when searching for a forecast at a given time = maximum age"""
    default_horizon: int
    maximum_horizon: int
    available_cities: list[str]


class Health(BaseModel):
    status: str
    age: int
