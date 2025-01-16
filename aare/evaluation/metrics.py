from dataclasses import dataclass, asdict
from typing import cast

from darts import TimeSeries
from darts.metrics import mae, rmse


@dataclass
class Metrics:
    """Collection of applicable metrics for this on the temperature."""

    METRICS = dict(mae=mae, rmse=rmse)

    mae: float
    rmse: float

    @classmethod
    def from_series(cls, actual: TimeSeries, prediction: TimeSeries):
        """Returns a fully calculated set of metrics for a ground truth and forecast."""
        metrics = {key: cast(float, metric(actual, prediction)) for (key, metric) in cls.METRICS.items()}

        return Metrics(**metrics)

    def to_dict(self) -> dict[str, float]:
        return asdict(self)
