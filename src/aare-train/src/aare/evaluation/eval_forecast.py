from dataclasses import dataclass
from typing import cast, Optional, Sequence

import matplotlib.axes
from darts import TimeSeries
from pandas import Timedelta, Timestamp

from aare.evaluation.eval_metric import EvalMetric


def _find_section(
    ts: TimeSeries | Sequence[TimeSeries] | None, start: Timestamp, end: Timestamp, none_ok=True
) -> Optional[TimeSeries]:
    if ts is None:
        if none_ok:
            return None

        raise ValueError("ts must not be None if none_ok is False")

    if not isinstance(ts, TimeSeries):
        assert isinstance(ts, Sequence), "ts is something other than None, TimeSeries or Sequence"
        # if ts is a sequence, none of them will overlap, so only one can contain the end time.
        # what can happen is that the min_lookback is greater than the context length of the model
        # which means the section we're trying to find here might be larger than the slices can offer.
        # in those cases, we need to truncate the start. if the lookback was never larger than the context length,
        # both start and end times should be inside of exactly one slice given the constraints of darts.

        slice = next((s for s in ts if end in s), None)
        if not slice:
            raise ValueError(
                f"Unable to find a slice for ({start}, {end}) in ["
                + ",".join(f"({s.start_time()}, {s.end_time()})" for s in ts)
                + "]"
            )

        ts = slice

    # truncate if requested start is earlier than start time of this subseries
    slice_start = cast(Timestamp, ts.start_time())
    new_start = max(start, slice_start)
    assert new_start < end, (
        f"Start got moved from '{start}' to '{new_start}' because of slice period ({ts.start_time()}, {ts.end_time()}) and is now after '{end}'!"
    )

    return ts[start:end]


@dataclass
class EvalForecast:
    actual: TimeSeries
    forecast: TimeSeries
    lookback: Timedelta
    metrics: Optional[EvalMetric]
    future_cov: TimeSeries | None = None
    """A forecast that was created as part of a model evaluation with actual/known data and resulting metrics."""

    def __init__(
        self,
        actual_full: TimeSeries | Sequence[TimeSeries],
        forecast: TimeSeries,
        lookback_hours: int,
        metrics: Optional[EvalMetric] = None,
        add_metrics=True,
        future_cov: TimeSeries | Sequence[TimeSeries] | None = None,
    ):
        """
        Pulls the actual data out of all_data according to the period of the forecast plus some lookback period.
        Optionally calculates metrics for this specific forecast period.
        """
        assert lookback_hours >= 0, "lookback_hours must be positive"
        self.lookback = cast(Timedelta, Timedelta(hours=lookback_hours))  # cannot be NaT
        pred_start = cast(Timestamp, forecast.start_time())
        start, end = pred_start - self.lookback, forecast.end_time()
        assert isinstance(end, Timestamp), "Passed forecast TimeSeries with range index?!"

        self.actual = cast(TimeSeries, _find_section(actual_full, start, end, none_ok=False))
        self.future_cov = _find_section(future_cov, start, end)

        self.forecast = forecast
        self.metrics = metrics
        if add_metrics and metrics is None:
            self.calc_metrics()

    def calc_metrics(self):
        """Calculate and store the metrics for this instance. Also returns them for convenience."""
        self.metrics = EvalMetric.from_series(self.actual, self.forecast)
        return self.metrics

    def plot(self, title: str, ax: Optional[matplotlib.axes.Axes] = None, with_covariates: bool | list[str] = False):
        ax = self.actual.plot(label="actual", ax=ax)
        self.forecast.plot(label="forecast", ax=ax)
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
