from dataclasses import dataclass

from matplotlib import pyplot as plt

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

    def plot(self, title: str, with_covariates=False):
        fig, axes = plt.subplot_mosaic("AA;BC")

        self.last_forecast.plot(title + " (last)", ax=axes["A"], with_covariates=with_covariates)
        self.best_forecast.plot(title + " (best)", ax=axes["B"], with_covariates=with_covariates)
        self.worst_forecast.plot(title + " (worst)", ax=axes["C"], with_covariates=with_covariates)

        return fig
