from typing import Mapping

import torch
from darts.models.forecasting.forecasting_model import ForecastingModel

from aare_train.evaluation.evaluation import evaluate_model, store_results
from aare_train.features.registry import FEATURES
from aare_train.fetching.feature_set import FeatureSet
from aare_train.params import Params
from aare_train.paths import METRICS_FOLDER, FORECAST_SAMPLES_FOLDER


def evaluation_pipeline_uni(
    models: Mapping[str, ForecastingModel],
    params: Params,
    use_test: bool = False,
) -> None:
    """Evaluate all specified models on the validation (or test) data and write the results to the pre-defined folders."""
    general_params = params["general"]
    forecast_horizon = general_params["forecast_horizon"]
    stride = params["validation"]["stride"]
    min_lookback_hours = params["validation"]["min_lookback_hours"]
    season = general_params["season_start"], general_params["season_end"]
    tz = general_params["timezone"]

    fs = FeatureSet(FEATURES[general_params["target"]], split_params=params["split"])
    target_data, _, _ = fs.get_test() if use_test else fs.get_val()

    # mostly to suppress the torch notice, darts has bad support for this
    torch.set_float32_matmul_precision("medium")

    METRICS_FOLDER.mkdir(exist_ok=True)
    FORECAST_SAMPLES_FOLDER.mkdir(exist_ok=True)

    for name, model in models.items():
        metrics, samples, raw_metrics = evaluate_model(
            model, target_data, forecast_horizon, stride, min_lookback_hours, tz=tz, month_filter=season, get_raw=True
        )

        if use_test:
            name += "-test"

        store_results(name, metrics, raw_metrics, samples, override=True)
