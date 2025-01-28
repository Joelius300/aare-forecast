# /// script
# requires-python = ">=3.10,<3.12"
# dependencies = [
#     "darts",
#     "dvc",
#     "influxdb-client",
#     "pandas",
#     "timesfm[torch]",
# ]
# ///
import json
import logging
import pickle

from darts import TimeSeries
from darts.utils.missing_values import extract_subseries

from aare.AareDataset import AareDataset
from aare.evaluation import evaluate_model, Forecast
from aare.params import GeneralParams
from aare.params import read_params
from aare.preparation import (
    resample,
    remove_faulty_periods,
    remove_outliers,
    interpolate,
)
from aare.utils import to_ts, METRICS_FOLDER, LAST_FORECASTS_FOLDER
from aare.wrappers.timesfm import TimesFmDarts


def load_model(params: GeneralParams):
    return


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

    model = TimesFmDarts(horizon)

    metrics, last_prediction = evaluate_model(model, val_subs, stride, horizon)
    lookback_hours = max(model.context_length, min_lookback_hours)
    last_forecast = Forecast(val, last_prediction, lookback_hours)

    name = "TIMESFM"
    with open(METRICS_FOLDER / f"{name}.json", "wt") as metrics_file:
        json.dump(metrics.to_dict(), metrics_file)

    with open(LAST_FORECASTS_FOLDER / f"{name}.pkl", "wb") as last_forecast_file:
        pickle.dump(last_forecast, last_forecast_file)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
