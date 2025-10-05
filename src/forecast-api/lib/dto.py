from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class ModelInfo(BaseModel):
    name: str
    version: str


class PredictionMetadata(BaseModel):
    last_updated: Optional[datetime]
    """Exact time the prediction was made (=last updated) or null if no prediction was returned."""
    model: Optional[ModelInfo]


class PredictionPayload(BaseModel):
    time: list[datetime]
    temp_bern: list[float]
    metadata: PredictionMetadata


class Config(BaseModel):
    timezone: str
    maximum_prediction_age: str
    """Amount of time to look back when searching for a prediction at a given time = maximum age"""
    default_horizon: int
    maximum_horizon: int
