from typing import Optional, Union, Sequence

import pandas as pd
import timesfm
import torch.cuda
from darts import TimeSeries
from darts.models.forecasting.forecasting_model import GlobalForecastingModel

from aare.constants import TIME
from aare.utils import to_ts


class TimesFmDarts(GlobalForecastingModel):
    def __init__(
        self,
        # context_length: int,
        forecast_horizon: int,
    ):
        super().__init__()

        # todo make readonly properties
        self.context_length = 512  # default it was trained on. equiv to 21.3 days.
        self.input_chunk_length = 32  # cannot be changed for pre-trained
        self.forecast_horizon = forecast_horizon
        self._output_chunk_length = 128  # cannot be changed for pre-trained
        # self.window_size  <- hparam
        # self.version = 200m or 500m -> has implications i.e. for context length etc.

        # TODO experiment with 200m and 500m model, and with window_size, but I don't think that does any good.
        self.tfm = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend="gpu" if torch.cuda.is_available() else "cpu",
                per_core_batch_size=32,
                horizon_len=forecast_horizon,
                output_patch_len=self._output_chunk_length,
                input_patch_len=self.input_chunk_length,
                context_len=self.context_length,
                # even though we don't want quantiles, the model weights
                # contain quantiles heads and must be loaded if we want
                # to use the pre-trained one, apparently.
            ),
            checkpoint=timesfm.TimesFmCheckpoint(huggingface_repo_id="google/timesfm-1.0-200m-pytorch"),
        )

        # no need to call fit or to store any info on the dimensions etc.
        self._fit_called = True

    @property
    def output_chunk_length(self) -> Optional[int]:
        return self._output_chunk_length

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
        n: Optional[int] = None,
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

        if n is not None and n != self.forecast_horizon:
            raise ValueError(
                f"The forecast horizon must always be the same (specified {self.forecast_horizon} at init)"
            )

        super().predict(
            self.forecast_horizon,
            series,
            past_covariates,
            future_covariates,
            num_samples,
            verbose,
            predict_likelihood_parameters,
            show_warnings,
        )

        if series is None or not isinstance(series, TimeSeries):
            # historical_forecast only every provides a single one, probably no need to do multiple (but possible)
            raise ValueError("must provide a single series to predict on")

        df = pd.DataFrame(index=series.time_index, data=series.values(), columns=series.columns)
        df = df.melt(ignore_index=False, var_name="unique_id", value_name="values").reset_index(names="ds")

        # hard-code hourly frequency here, which is mapped to the same high-frequency settings
        # as seconds[?], minutes, days, business days and microseconds are (everything up to daily).
        # we won't ever forecast anything below daily frequency anyway, so this should be fine.
        # ps. there might be a bug with ms, would need to use L. see freq_map.
        forecast = self.tfm.forecast_on_df(df, freq="h", verbose=verbose)
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
        return -self.context_length, self._output_chunk_length - 1, None, None, None, None, 0, None

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
