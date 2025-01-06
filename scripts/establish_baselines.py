import json
import logging
import pickle
from typing import cast, TypedDict

import numpy as np
import torch
from darts import TimeSeries
from darts.metrics import mae, rmse
from darts.models import GlobalNaiveSeasonal, GlobalNaiveAggregate
from darts.models.forecasting.forecasting_model import GlobalForecastingModel
from darts.utils.missing_values import extract_subseries
from pandas import Timedelta, Timestamp

from aare.AareDataset import AareDataset
from aare.params import read_params
from aare.preparation import (
    resample,
    remove_faulty_periods,
    remove_outliers,
    interpolate,
)
from aare.utils import to_ts, DATA_FOLDER

logger = logging.getLogger(__name__)

METRICS_FOLDER = DATA_FOLDER / "metrics"
LAST_PREDICTIONS_FOLDER = DATA_FOLDER / "last_predictions"


def prepare_data(dataset: AareDataset) -> TimeSeries:
    val = dataset.get_val()

    val = resample(val)
    val = remove_faulty_periods(val)
    val = remove_outliers(val)
    val = interpolate(val, drop_filled=True)

    ts = to_ts(val)

    return ts


# TODO Refactor metrics and evaluation into library


class LastPrediction(TypedDict):
    actual: TimeSeries
    prediction: TimeSeries
    lookback: Timedelta


class Metrics(TypedDict):
    mae: float
    rmse: float


def evaluate_model(
    model: GlobalForecastingModel, val: list[TimeSeries], stride: int, horizon: int
) -> tuple[Metrics, TimeSeries]:
    # only takes the components etc. global naive don't care about the values
    model.fit(val[0])

    # produces multiple predictions (TimeSeries) with the specified stride FOR EACH SUBSERIES
    historical_forecasts = cast(
        list[list[TimeSeries]],
        model.historical_forecasts(
            val,
            stride=stride,
            forecast_horizon=horizon,
            last_points_only=False,
            retrain=False,
        ),
    )

    backtest = model.backtest(
        val, historical_forecasts=historical_forecasts, metric=[mae, rmse]
    )

    metrics = np.mean(backtest, axis=0)
    metrics = Metrics(mae=float(metrics[0]), rmse=float(metrics[1]))

    last_forecast = historical_forecasts[-1][-1]

    return metrics, last_forecast


def get_last_prediction(
    all_data: TimeSeries, last_forecast: TimeSeries, lookback_hours: int
):
    assert lookback_hours >= 0, "lookback_hours must be positive"
    lookback = cast(Timedelta, Timedelta(hours=lookback_hours))  # cannot be NaT
    pred_start = cast(Timestamp, last_forecast.start_time())
    actual = all_data[pred_start - lookback : last_forecast.end_time()]

    return LastPrediction(actual=actual, prediction=last_forecast, lookback=lookback)


def main():
    dataset = AareDataset.from_conf()
    params = read_params()
    horizon = params["general"]["forecast_horizon"]
    stride = params["validation"]["stride"]
    lookback_hours = params["validation"]["lookback_hours"]
    val = prepare_data(dataset)
    val_subs = extract_subseries(val)

    # mostly to suppress the torch notice, darts has bad support for this
    torch.set_float32_matmul_precision("medium")
    models = {
        "LOCF": GlobalNaiveSeasonal(input_chunk_length=1, output_chunk_length=1),
        # daily seasonality
        "SNAIVE": GlobalNaiveSeasonal(input_chunk_length=24, output_chunk_length=1),
        # weekly mean
        "MEAN": GlobalNaiveAggregate(
            input_chunk_length=7 * 24, output_chunk_length=horizon
        ),
    }

    METRICS_FOLDER.mkdir(exist_ok=True)
    LAST_PREDICTIONS_FOLDER.mkdir(exist_ok=True)

    for name, model in models.items():
        metrics, last_forecast = evaluate_model(model, val_subs, stride, horizon)
        last_prediction = get_last_prediction(val, last_forecast, lookback_hours)

        with open(METRICS_FOLDER / f"{name}.json", "wt") as metrics_file:
            json.dump(metrics, metrics_file)

        with open(
            LAST_PREDICTIONS_FOLDER / f"{name}.pkl", "wb"
        ) as past_prediction_file:
            pickle.dump(last_prediction, past_prediction_file)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
