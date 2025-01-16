import json
import logging
import pickle

import torch
from darts import TimeSeries
from darts.models import GlobalNaiveSeasonal, GlobalNaiveAggregate
from darts.utils.missing_values import extract_subseries

from aare.AareDataset import AareDataset
from aare.evaluation import evaluate_model, Forecast
from aare.params import read_params
from aare.preparation import (
    resample,
    remove_faulty_periods,
    remove_outliers,
    interpolate,
)
from aare.utils import to_ts, METRICS_FOLDER, LAST_FORECASTS_FOLDER

logger = logging.getLogger(__name__)


def prepare_data(dataset: AareDataset) -> TimeSeries:
    val = dataset.get_val()

    val = resample(val)
    val = remove_faulty_periods(val)
    val = remove_outliers(val)
    val = interpolate(val, drop_filled=True)

    ts = to_ts(val)

    return ts


def main():
    dataset = AareDataset.from_conf()
    params = read_params()
    horizon = params["general"]["forecast_horizon"]
    stride = params["validation"]["stride"]
    min_lookback_hours = params["validation"]["min_lookback_hours"]
    val = prepare_data(dataset)
    val_subs = extract_subseries(val)

    # mostly to suppress the torch notice, darts has bad support for this
    torch.set_float32_matmul_precision("medium")
    models = {
        "LOCF": GlobalNaiveSeasonal(input_chunk_length=1, output_chunk_length=1),
        # daily seasonality
        "SNAIVE": GlobalNaiveSeasonal(input_chunk_length=24, output_chunk_length=1),
        # weekly mean
        "MEAN": GlobalNaiveAggregate(input_chunk_length=7 * 24, output_chunk_length=horizon),
    }

    METRICS_FOLDER.mkdir(exist_ok=True)
    LAST_FORECASTS_FOLDER.mkdir(exist_ok=True)

    for name, model in models.items():
        metrics, last_prediction = evaluate_model(model, val_subs, stride, horizon)
        lookback_hours = max(model.input_chunk_length, min_lookback_hours)
        last_forecast = Forecast(val, last_prediction, lookback_hours)

        with open(METRICS_FOLDER / f"{name}.json", "wt") as metrics_file:
            json.dump(metrics.to_dict(), metrics_file)

        with open(LAST_FORECASTS_FOLDER / f"{name}.pkl", "wb") as last_forecast_file:
            pickle.dump(last_forecast, last_forecast_file)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
