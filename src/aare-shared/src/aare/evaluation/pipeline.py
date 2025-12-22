import json
import pickle
from datetime import tzinfo
from typing import Mapping

import torch
from darts.models.forecasting.forecasting_model import ForecastingModel
from darts.utils.missing_values import extract_subseries

from aare.AareDataset import AareDataset
from aare.evaluation.evaluation import evaluate_model
from aare.params import ValidationParams
from aare.preparation import prepare_ts_aare_temp
from aare.utils import METRICS_FOLDER, FORECAST_SAMPLES_FOLDER


def evaluation_pipeline_uni(
    models: Mapping[str, ForecastingModel],
    forecast_horizon: int,
    validation_params: ValidationParams,
    tz: str | tzinfo,
) -> None:
    """Evaluate all specified models on the validation data and write the results to the pre-defined folders."""
    dataset = AareDataset.from_conf()
    stride = validation_params["stride"]
    min_lookback_hours = validation_params["min_lookback_hours"]
    val = prepare_ts_aare_temp(dataset.get_val())
    val_subs = extract_subseries(val)

    # mostly to suppress the torch notice, darts has bad support for this
    torch.set_float32_matmul_precision("medium")

    METRICS_FOLDER.mkdir(exist_ok=True)
    FORECAST_SAMPLES_FOLDER.mkdir(exist_ok=True)

    for name, model in models.items():
        metrics, sample = evaluate_model(model, val_subs, forecast_horizon, stride, min_lookback_hours, tz=tz)

        with open(METRICS_FOLDER / f"{name}.json", "wt") as metrics_file:
            json.dump(metrics.to_dict(), metrics_file)

        with open(FORECAST_SAMPLES_FOLDER / f"{name}.pkl", "wb") as forecast_sample_file:
            pickle.dump(sample, forecast_sample_file)
