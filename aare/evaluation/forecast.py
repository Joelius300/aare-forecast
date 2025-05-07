from dataclasses import dataclass
from typing import cast, Optional, Sequence

import matplotlib.axes
from darts import TimeSeries
from pandas import Timedelta, Timestamp

from aare.evaluation.metrics import Metrics


def _find_section(
    ts: TimeSeries | Sequence[TimeSeries] | None, start: Timestamp, end: Timestamp, none_ok=True
) -> Optional[TimeSeries]:
    if ts is None:
        if none_ok:
            return None

        raise ValueError("ts must not be None if none_ok is False")

    if not isinstance(ts, TimeSeries):
        assert isinstance(ts, Sequence), "ts is something other than None, TimeSeries or Sequence"
        ts = next(s for s in ts if start in s and end in s)

    return ts[start:end]


@dataclass
class Forecast:
    actual: TimeSeries
    prediction: TimeSeries
    lookback: Timedelta
    metrics: Optional[Metrics]
    future_cov: TimeSeries | None = None

    def __init__(
        self,
        actual_full: TimeSeries | Sequence[TimeSeries],
        prediction: TimeSeries,
        lookback_hours: int,
        metrics: Optional[Metrics] = None,
        add_metrics=True,
        future_cov: TimeSeries | Sequence[TimeSeries] | None = None,
    ):
        """
        Pulls the actual data out of all_data according to the period of the forecast plus some lookback period.
        Optionally calculates metrics for this specific forecast period.
        """
        assert lookback_hours >= 0, "lookback_hours must be positive"
        self.lookback = cast(Timedelta, Timedelta(hours=lookback_hours))  # cannot be NaT
        pred_start = cast(Timestamp, prediction.start_time())
        start, end = pred_start - self.lookback, prediction.end_time()
        assert isinstance(end, Timestamp), "Passed prediction TimeSeries with range index?!"

        self.actual = cast(TimeSeries, _find_section(actual_full, start, end, none_ok=False))
        self.future_cov = _find_section(future_cov, start, end)

        self.prediction = prediction
        self.metrics = metrics
        if add_metrics and metrics is None:
            self.calc_metrics()

    def calc_metrics(self):
        """Calculate and store the metrics for this instance. Also returns them for convenience."""
        self.metrics = Metrics.from_series(self.actual, self.prediction)
        return self.metrics

    def plot(self, title: str, ax: Optional[matplotlib.axes.Axes] = None, with_covariates: bool | list[str] = False):
        ax = self.actual.plot(label="actual", ax=ax)
        self.prediction.plot(label="prediction", ax=ax)
        ax.set_xlabel("Time")
        ax.set_ylabel("Temperature [°C]")

        if with_covariates and self.future_cov is not None:
            fc = self.future_cov
            if isinstance(with_covariates, list):
                fc = fc[with_covariates]
            fc.plot(label="fc", ax=ax)

        if self.metrics is not None:
            title += f" [{self.metrics}]"

        ax.set_title(title)

        return ax
