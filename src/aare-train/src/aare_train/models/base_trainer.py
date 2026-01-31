from abc import ABC, abstractmethod
from typing import Any, cast
from collections.abc import Mapping

import mlflow
from darts.models import RNNModel
from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel
from darts.models.forecasting.sklearn_model import SKLearnModel
from mlflow import ActiveRun
from matplotlib import pyplot as plt

from aare_train.evaluation.evaluation import evaluate_model
from aare_train.evaluation.eval_metric import EvalMetric
from aare_train.fetching.feature_identifiers import FeatureIdentifiers
from aare_train.fetching.feature_set import FeatureSet
from aare_train.features.registry import FEATURES
from aare_train.normalization import get_scalers
from aare_train.params import read_params, Params
from aare_train.darts_utils import get_data_stats

ModelType = TorchForecastingModel | SKLearnModel


class BaseTrainer(ABC):
    """Base class for model trainers that abstracts common training and evaluation logic."""

    def __init__(self, params: Params, features: FeatureIdentifiers):
        if features.get("past") is not None:
            raise NotImplementedError("Past covariates are not yet supported")

        ds = FeatureSet(
            targets=FEATURES.get_many(features["targets"]),
            future=FEATURES.get_many(features.get("future")),
            past=FEATURES.get_many(features.get("past")),
            split_params=params["split"],
        )

        self.train = ds.get_train()
        self.val = ds.get_val()

        self.tz = params["general"]["timezone"]
        self.train_target_subs, self.train_fc_subs = self.train[0], self.train[2]
        self.val_target_subs, self.val_fc_subs = self.val[0], self.val[2]
        self.scalers = get_scalers(self.train_target_subs, train_fc_subs=self.train_fc_subs)
        self.horizon = params["general"]["forecast_horizon"]
        self.stride = params["validation"]["stride"]
        self.min_lookback_hours = params["validation"]["min_lookback_hours"]
        self.season = params["general"]["season_start"], params["general"]["season_end"]

        self.hparams_general = {
            "horizon": self.horizon,
            "val_stride": self.stride,
            "split_train": params["split"]["train_split"],
            "split_val": params["split"]["val_split"],
            "split_test": params["split"]["test_split"],
            "features_targets": features["targets"],
            "features_future": features.get("future", []),
            "features_past": features.get("past", []),
        }

    @staticmethod
    def _prefix_dict(vals: Mapping[str, Any], prefix: str):
        prefix = prefix.removesuffix("_")
        return {prefix + "_" + key: value for key, value in vals.items()}

    def log_params_prefix(self, params: Mapping[str, Any], prefix: str):
        """Log all params in a dict with an added prefix"""
        mlflow.log_params(self._prefix_dict(params, prefix))

    def log_metrics_prefix(self, metrics: Mapping[str, Any], prefix: str):
        """Log all metrics in a dict with an added prefix"""
        mlflow.log_metrics(self._prefix_dict(metrics, prefix))

    @abstractmethod
    def build_model(self) -> ModelType:
        """Build and return the model with configured hyperparameters."""
        pass

    def fit(self, model: ModelType):
        """Fit the model on training data with validation."""
        assert "series" in self.scalers
        assert "future_covariates" in self.scalers
        assert self.train_fc_subs is not None
        assert self.val_fc_subs is not None

        orig_series = len(self.train_target_subs)
        orig_series_len = sum((len(x) for x in self.train_target_subs))

        if isinstance(model, RNNModel):
            # some training series might be too short for the model if it needs 4 days (horizon) for val and 1 day lookback (input)
            # I believe this only applies to RNNModels because it's handled automatically for non-AR models via the extreme_lags
            # which are in turn just derived from output_chunk_length, etc.
            self.train_target_subs = [x for x in self.train_target_subs if len(x) >= model.training_length]
            self.train_fc_subs = [x for x in self.train_fc_subs if len(x) >= model.training_length]
            self.val_target_subs = [x for x in self.val_target_subs if len(x) >= model.training_length]
            self.val_fc_subs = [x for x in self.val_fc_subs if len(x) >= model.training_length]

        data_stats = get_data_stats(self.train_target_subs, self.val_target_subs)
        data_stats["sub_series_dropped"] = orig_series - data_stats["train_n_subs"]
        data_stats["sub_series_dropped_len"] = orig_series_len - data_stats["train_len_total"]

        self.log_params_prefix(data_stats, "data")

        scaler_target = self.scalers["series"]
        scaler_fc = self.scalers["future_covariates"]

        fit_args = dict(
            series=scaler_target.transform(self.train_target_subs),
            future_covariates=scaler_fc.transform(self.train_fc_subs),
        )

        if isinstance(model, TorchForecastingModel):
            # torch models do validation during training, sklearn models don't
            fit_args |= dict(
                val_series=scaler_target.transform(self.val_target_subs),
                val_future_covariates=scaler_fc.transform(self.val_fc_subs),
            )

        return model.fit(**fit_args)  # pyright: ignore[reportArgumentType]

    def evaluate(self, model: ModelType, run: ActiveRun) -> EvalMetric:
        """Evaluate the model on validation data."""
        metrics, samples = evaluate_model(
            model,
            self.val_target_subs,
            self.horizon,
            self.stride,
            self.min_lookback_hours,
            future_cov=self.val_fc_subs,
            data_transformers=self.scalers,
            tz=self.tz,
            month_filter=self.season,
        )

        metrics_dict = metrics.to_dict()
        self.log_metrics_prefix(metrics_dict, "eval")
        sample_fig = samples.plot(str(run.info.run_name))
        mlflow.log_figure(sample_fig, artifact_file="samples.png")
        plt.close(sample_fig)  # otherwise it's kept in memory, well done matplotlib

        return metrics

    def train(self, run_name: str, experiment_name: str, evaluate: bool = True) -> tuple[ModelType, EvalMetric | None]:
        """Train the model and optionally evaluate it."""
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=run_name, log_system_metrics=True) as run:
            mlflow.log_dict(cast(dict, read_params(ensure_dvc=True)), "params.yaml")
            mlflow.log_params(self.hparams_general)

            model = self.build_model()
            self.fit(model)

            metrics = None
            if evaluate:
                metrics = self.evaluate(model, run)

            return model, metrics
