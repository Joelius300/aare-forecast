from dataclasses import dataclass, asdict
from typing import cast, Optional

import numpy as np
from darts import TimeSeries
from darts.metrics import mae, rmse


@dataclass
class Metrics:
    """Collection of applicable metrics for point forecasts on the temperature. MAE has prio."""

    METRICS = dict(mae=mae, rmse=rmse)

    mae: float
    """
    The mean absolute error of all the points on the forecast.
    
    If these are the aggregate metrics for many forecast, it's the MEDIAN of all those MAE.
    """
    rmse: float
    """
    The root mean squared error of all the points on the forecast.

    If these are the aggregate metrics for many forecast, it's the MEDIAN of all those RMSE.
    """

    # Currently only std of many different forecasts, NOT std for the different point-errors within a forecast
    mae_std: Optional[float] = None
    """STD of MAE, if this is an aggregated metric over many forecasts."""
    rmse_std: Optional[float] = None
    """STD of RMSE, if this is an aggregated metric over many forecasts."""

    def __repr__(self):
        return (
            f"MAE: {self.mae:.3f}{f' (STD: {self.mae_std:.2f})' if self.mae_std is not None else ''}"
            " / "
            f"RMSE: {self.rmse:.3f}{f' (STD: {self.rmse_std:.2f})' if self.rmse_std is not None else ''}"
        )

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_series(cls, actual: TimeSeries, forecast: TimeSeries):
        """Returns a fully calculated set of metrics for a ground truth and forecast."""
        metrics = {key: cast(float, metric(actual, forecast)) for (key, metric) in cls.METRICS.items()}

        return Metrics(**metrics)

    @classmethod
    def from_ndarray(cls, values: np.ndarray):
        # noinspection PyTypeChecker
        return Metrics(mae=values[0], rmse=values[1])

    @classmethod
    def from_dict(cls, value: dict[str, float]):
        return Metrics(**value)
