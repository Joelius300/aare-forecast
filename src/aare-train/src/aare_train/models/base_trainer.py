from abc import ABC, abstractmethod
from typing import Any, cast
from collections.abc import Mapping
import os

from darts import TimeSeries
import mlflow
from darts.models import RNNModel
from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel
from darts.models.forecasting.sklearn_model import SKLearnModel
from mlflow import ActiveRun
from matplotlib import pyplot as plt
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger

from aare_train.evaluation.evaluation import evaluate_model
from aare_train.evaluation.eval_metric import EvalMetric
from aare_train.fetching.feature_identifiers import FeatureIdentifiers
from aare_train.fetching.feature_set import FeatureSet
from aare_train.features.registry import FEATURES
from aare_train.normalization import get_scalers
from aare_train.params import Params
from aare_train.darts_utils import get_data_stats, exclude_short_series, model_needs_fc
from aare_train.compat.types import DataTransformers

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

        self._train = ds.get_train()
        self._val = ds.get_val()

        self._scalers: DataTransformers | None = None
        self._val_subs: tuple[list[TimeSeries], list[TimeSeries] | None] | None = None

        self.tz = params["general"]["timezone"]
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

    @staticmethod
    def get_add_encoders(add_day_enc: bool, add_year_enc: bool) -> dict[str, dict[str, list[str]]] | None:
        """Get the add_encoders configuration for cyclic temporal features."""
        if not add_day_enc and not add_year_enc:
            return None

        enc = []
        if add_day_enc:
            enc.append("hour")
        if add_year_enc:
            enc.append("day_of_year")

        return {"cyclic": {"future": enc}}

    def get_trainer_params(
        self,
        model_name: str,
        early_stopping_var: str | None = "val_loss",
        early_stopping_patience: int = 5,
        log_every_n_steps: int = 50,
    ):
        """Get pytorch lightning trainer parameters for torch models."""
        run = mlflow.active_run()
        assert run is not None, "no active mlflow run"
        mlflow_logger = MLFlowLogger(
            model_name,
            # they fucked up their default which is evaluated at the import of the module, so set_tracking_uri is ignored
            # https://github.com/Lightning-AI/pytorch-lightning/discussions/11197#discussioncomment-9164713
            tracking_uri=os.getenv("MLFLOW_TRACKING_URI"),
            run_id=run.info.run_id,  # pyright: ignore[reportUnknownArgumentType]
        )

        callbacks: list[Callback] = []
        if early_stopping_var:
            callbacks.append(EarlyStopping(early_stopping_var, patience=early_stopping_patience))

        return {
            "logger": mlflow_logger,
            "callbacks": callbacks,
            "log_every_n_steps": log_every_n_steps,
        }

    def log_params_prefix(self, params: Mapping[str, Any], prefix: str):
        """Log all params in a dict with an added prefix"""
        mlflow.log_params(self._prefix_dict(params, prefix))

    def log_metrics_prefix(self, metrics: Mapping[str, Any], prefix: str):
        """Log all metrics in a dict with an added prefix"""
        mlflow.log_metrics(self._prefix_dict(metrics, prefix))

    @abstractmethod
    def build_model(self, **hparams: Any) -> ModelType:
        """Build and return the model with given hyperparameters."""
        pass

    def fit(self, model: ModelType) -> ModelType:
        """Fit the model on training data with validation."""
        mlflow.log_params(self.hparams_general)

        self._scalers = None
        self._val_subs = None

        train_target, _, train_fc = self._train
        val_target, _, val_fc = self._val
        scalers = get_scalers(train_target, train_fc_subs=train_fc)

        assert (train_fc is not None) == (val_fc is not None), "train has fc but val does not, or vice versa"
        assert model_needs_fc(model) == (train_fc is not None), "future covariates not in line with model support"

        orig_series_n_subs = len(train_target)
        orig_series_len = sum(len(x) for x in train_target)
        min_len = model.training_length if isinstance(model, RNNModel) else abs(model.extreme_lags[0] or 0)
        if min_len < 1:
            raise ValueError(f"Invalid min_len for training in {min_len}")

        # Some training series might be too short for the model if it needs 4 days (horizon) for val and 1 day lookback (input)
        # Previously, I believed this only applied to RNNModels because it's handled automatically for non-AR models via the extreme_lags
        # which are in turn just derived from output_chunk_length, etc. but that does not seem to be the case.
        train_target = exclude_short_series(train_target, min_len)
        train_fc = exclude_short_series(train_fc, min_len)
        val_target = exclude_short_series(val_target, min_len)
        val_fc = exclude_short_series(val_fc, min_len)

        data_stats = get_data_stats(train_target, val_target)
        data_stats["sub_series_dropped"] = orig_series_n_subs - cast(int, data_stats["train_n_subs"])
        data_stats["sub_series_dropped_len"] = orig_series_len - cast(int, data_stats["train_len_total"])

        self.log_params_prefix(data_stats, "data")

        scaler_target = scalers.get("series")
        assert scaler_target is not None, "target scaler is none"
        scaler_fc = scalers.get("future_covariates")
        assert (train_fc is None) == (scaler_fc is None)

        fit_args = dict(
            series=scaler_target.transform(train_target),
            future_covariates=scaler_fc.transform(train_fc) if scaler_fc and train_fc else None,
        )

        if isinstance(model, TorchForecastingModel):
            # torch models do validation during training, sklearn models don't
            fit_args |= dict(
                val_series=scaler_target.transform(val_target),
                val_future_covariates=scaler_fc.transform(val_fc) if scaler_fc and val_fc else None,
            )

        model.fit(**fit_args)  # pyright: ignore[reportArgumentType, reportUnknownMemberType]

        self._scalers = scalers
        self._val_subs = val_target, val_fc

        return model

    def evaluate(self, model: ModelType, run: ActiveRun | None) -> EvalMetric:
        """Evaluate the model on validation data and optionally log metrics and the sample figure to mlflow."""
        if self._val_subs is None:
            raise ValueError("Must first successfully train a model to then evaluate with appropriate scaling.")

        val_target, val_fc = self._val_subs

        metrics, samples = evaluate_model(
            model,
            val_target,
            self.horizon,
            self.stride,
            self.min_lookback_hours,
            future_cov=val_fc,
            data_transformers=self._scalers,
            tz=self.tz,
            month_filter=self.season,
            # TODO only evaluate on the data that makes sense if season is set (performance optimization)
            # TODO parallelization
            # TODO allow storing artifacts like the raw metrics
        )

        metrics_dict = metrics.to_dict()
        if run is not None:
            self.log_metrics_prefix(metrics_dict, "eval")
            sample_fig = samples.plot(str(run.info.run_name))  # pyright: ignore[reportUnknownMemberType]
            mlflow.log_figure(sample_fig, artifact_file="samples.png")
            plt.close(sample_fig)  # otherwise it's kept in memory, well done matplotlib

        return metrics
