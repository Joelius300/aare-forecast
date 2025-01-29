from dataclasses import dataclass

from aare.evaluation.forecast import Forecast


@dataclass
class ForecastSamples:
    last_forecast: Forecast
    """
    The last forecast of a set of forecasts (usually all of validation).

    Most recent but otherwise as unbiased as we can. Taking the last is more reproducible than
    picking any random forecast from anywhere.
    """

    best_forecast: Forecast
    """Forecast with the minimum MAE of a set of forecasts (usually all of validation)."""
    worst_forecast: Forecast
    """Forecast with the maximum MAE of a set of forecasts (usually all of validation)."""
