from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class ModelInfo(BaseModel):
    name: str
    version: str


class PredictionMetadata(BaseModel):
    run_ts: Optional[datetime]
    """Exact time the prediction was made or null if not prediction was returned."""
    model: Optional[ModelInfo]


class PredictionPayload(BaseModel):
    time: list[datetime]
    temp_bern: list[float]
    metadata: PredictionMetadata
