from dataclasses import dataclass

from matplotlib import pyplot as plt

from aare.evaluation.forecast import Forecast


@dataclass
class ForecastSamples:
    most_avg_forecast: Forecast
    """
    The most average forecast of a set of forecasts (usually all of validation).

    The forecast with the closest metric as the mean metric of all the forecasts (= hopefully representable).
    """

    best_forecast: Forecast
    """Forecast with the minimum MAE of a set of forecasts (usually all of validation)."""
    worst_forecast: Forecast
    """Forecast with the maximum MAE of a set of forecasts (usually all of validation)."""

    def plot(self, title: str, with_covariates: bool | list[str] = False):
        fig, axes = plt.subplot_mosaic("AA;BC")

        self.most_avg_forecast.plot(title + " (avg)", ax=axes["A"], with_covariates=with_covariates)
        self.best_forecast.plot(title + " (best)", ax=axes["B"], with_covariates=with_covariates)
        self.worst_forecast.plot(title + " (worst)", ax=axes["C"], with_covariates=with_covariates)

        return fig
