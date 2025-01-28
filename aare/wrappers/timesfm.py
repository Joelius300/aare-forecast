from typing import Optional, Union, Sequence

import pandas as pd
import timesfm
from darts import TimeSeries
from darts.models.forecasting.forecasting_model import GlobalForecastingModel

from aare.constants import TIME
from aare.utils import to_ts


class TimesFmDarts(GlobalForecastingModel):
    def __init__(
        self,
        tfm: timesfm.TimesFm,  # TODO instantiate here
        # input_chunk_length: int,
    ):
        super().__init__()
        self.tfm = tfm
        # self.input_chunk_length = input_chunk_length

        # no need to call fit or to store any info on the dimensions etc.
        self._fit_called = True

    def fit(
        self,
        series: Union[TimeSeries, Sequence[TimeSeries]],
        past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
    ):
        if past_covariates is not None or future_covariates is not None:
            raise ValueError("Covariates are not supported atm")

        # for this model, this just does some checks, stores the training series
        # and sets _fit_called. For other models, like TorchForecastingModel, it
        # does some more important things like storing the dimensions/components.
        super().fit(series, past_covariates, future_covariates)

        return self

    def predict(
        self,
        n: int,
        series: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        num_samples: int = 1,
        verbose: bool = False,
        predict_likelihood_parameters: bool = False,
        show_warnings: bool = True,
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:
        if past_covariates is not None or future_covariates is not None:
            raise ValueError("Covariates are not supported atm")

        super().predict(
            n,
            series,
            past_covariates,
            future_covariates,
            num_samples,
            verbose,
            predict_likelihood_parameters,
            show_warnings,
        )

        if series is None or not isinstance(series, TimeSeries):
            raise ValueError("must provide a single series to predict on")

        df = pd.DataFrame(index=series.time_index, data=series.values(), columns=series.columns)
        df = df.melt(ignore_index=False, var_name="unique_id", value_name="values").reset_index(names="ds")

        # hard-code hourly frequency here, which is mapped to the same high-frequency settings
        # as seconds[?], minutes, days, business days and microseconds are (everything up to daily).
        forecast = self.tfm.forecast_on_df(df, freq="H")
        forecast = forecast[["ds", "unique_id", "timesfm"]]  # drop quantiles
        forecast = forecast.pivot(index="ds", columns="unique_id", values="timesfm")
        forecast = forecast.reset_index(names=TIME)  # rename ds (index) to _time (column)

        return to_ts(forecast)

    @property
    def supports_multivariate(self) -> bool:
        return True

    @property
    def uses_past_covariates(self) -> bool:
        return False

    @property
    def uses_future_covariates(self) -> bool:
        return False

    @property
    def uses_static_covariates(self) -> bool:
        return False

    @property
    def extreme_lags(
        self,
    ) -> tuple[
        Optional[int],
        Optional[int],
        Optional[int],
        Optional[int],
        Optional[int],
        Optional[int],
        int,
        Optional[int],
    ]:
        raise ValueError("MUST BOTHER WITH EXTREME LAGS :(")
        # return (-context_len, ?, None, None, None, None, 0, None)

    @property
    def _model_encoder_settings(
        self,
    ) -> tuple[
        Optional[int],
        Optional[int],
        bool,
        bool,
        Optional[list[int]],
        Optional[list[int]],
    ]:
        raise ValueError("MUST BOTHER WITH encoder settings :(")
