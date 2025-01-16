from dataclasses import dataclass
from typing import cast, Optional

from darts import TimeSeries
from pandas import Timedelta, Timestamp

from aare.evaluation.metrics import Metrics


@dataclass
class Forecast:
    actual: TimeSeries
    prediction: TimeSeries
    lookback: Timedelta
    metrics: Optional[Metrics]

    def __init__(self, all_data: TimeSeries, prediction: TimeSeries, lookback_hours: int, add_metrics=True):
        """
        Pulls the actual data out of all_data according to the period of the forecast plus some lookback period.
        Optionally calculates metrics for this specific forecast period.
        """
        assert lookback_hours >= 0, "lookback_hours must be positive"
        self.lookback = cast(Timedelta, Timedelta(hours=lookback_hours))  # cannot be NaT
        pred_start = cast(Timestamp, prediction.start_time())
        self.actual = all_data[pred_start - self.lookback : prediction.end_time()]
        self.prediction = prediction
        if add_metrics:
            self.calc_metrics()

    def calc_metrics(self):
        """Calculate and store the metrics for this instance. Also returns them for convenience."""
        self.metrics = Metrics.from_series(self.actual, self.prediction)
        return self.metrics
