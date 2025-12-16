from abc import ABC, abstractmethod
import os
from typing import Any, Optional, cast
from collections.abc import Iterator

import mlflow
from darts.models import RNNModel
from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel
from darts.models.forecasting.sklearn_model import SKLearnModel
from mlflow import ActiveRun
from optuna import Trial, TrialPruned
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger
from matplotlib import pyplot as plt

from aare.compat.optuna_lightning_integration import PyTorchLightningPruningCallback
from aare.evaluation.evaluation import evaluate_model
from aare.evaluation.eval_metric import EvalMetric
from aare.feature_identifiers import FeatureIdentifiers
from aare.feature_set import FeatureSet
from aare.features.registry import FEATURES
from aare.normalization import get_scalers
from aare.params import read_params, Params
from aare.utils import get_data_stats

ModelType = TorchForecastingModel | SKLearnModel


class BaseTuner(ABC):
    def __init__(self, model_name: str, params: Params, features: FeatureIdentifiers):
        self.model_name = model_name
        self.current_run: Optional[ActiveRun] = None

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

    def get_trainer_params(
        self,
        trial: Trial,
        early_stopping_var: str | None = "val_loss",
        early_stopping_patience=5,
        pruning_var: str | None = "val_mae",
        log_every_n_steps=50,
    ):
        """Get the common trainer parameters for pl_trainer_kwargs"""
        assert self.current_run is not None
        # they fucked up their default which is evaluated at the import of the module, so set_tracking_uri is ignored
        # https://github.com/Lightning-AI/pytorch-lightning/discussions/11197#discussioncomment-9164713
        mlflow_logger = MLFlowLogger(
            self.model_name, tracking_uri=os.getenv("MLFLOW_TRACKING_URI"), run_id=self.current_run.info.run_id
        )

        callbacks = []
        if early_stopping_var:
            callbacks.append(EarlyStopping(early_stopping_var, patience=early_stopping_patience))

        if pruning_var:
            callbacks.append(PyTorchLightningPruningCallback(trial, monitor=pruning_var))

        # TODO add callback to do a full evaluation with our evaluate_model every 10-20 epochs maybe?

        return {
            "logger": mlflow_logger,
            "callbacks": callbacks,
            "log_every_n_steps": log_every_n_steps,
        }

    @staticmethod
    def suggest_add_encoders(trial: Trial) -> Optional[dict]:
        """
        Suggest values for 'add_encoders' with cyclic daily and yearly encoding.

        The optuna keys are:

        - add_day_enc
        - add_year_enc
        """
        add_day_enc = trial.suggest_categorical("add_day_enc", [True, False])
        add_year_enc = trial.suggest_categorical("add_year_enc", [True, False])

        if not add_day_enc and not add_year_enc:
            return None

        enc = []
        if add_day_enc:
            enc.append("hour")
        if add_year_enc:
            enc.append("day_of_year")

        return {"cyclic": {"future": enc}}

    @staticmethod
    def _prefix_dict(vals: dict, prefix: str):
        prefix = prefix.removesuffix("_")
        return {prefix + "_" + key: value for key, value in vals.items()}

    def log_params_prefix(self, params: dict, prefix: str):
        """Log all params in a dict with an added prefix"""
        mlflow.log_params(self._prefix_dict(params, prefix))

    def log_metrics_prefix(self, metrics: dict, prefix: str):
        """Log all metrics in a dict with an added prefix"""
        mlflow.log_metrics(self._prefix_dict(metrics, prefix))

    @staticmethod
    def prune_if_requested(trial: Trial):
        if trial.should_prune():
            raise TrialPruned()

    @abstractmethod
    def get_model(self, trial: Trial) -> ModelType:
        """
        Instantiate a compatible model with parameters suggested from the optuna trial.
        Log all the selected hyperparameters with the 'model' prefix!
        """
        pass

    def get_optim_vars(self, metrics: EvalMetric, model: ModelType):
        """Get the metric(s) that optuna should minimize."""
        return metrics.mae

    def fit(self, model: ModelType):
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

        # I'm impressed how much it can statically derive, but somehow it still thinks this is wrong
        return model.fit(**fit_args)  # pyright: ignore[reportArgumentType]

    def evaluate(self, model: ModelType, run: ActiveRun) -> EvalMetric:
        metrics, samples = evaluate_model(
            model,
            self.val_target_subs,
            self.horizon,
            self.stride,
            self.min_lookback_hours,
            future_cov=self.val_fc_subs,
            data_transformers=self.scalers,
            tz=self.tz,
        )

        metrics_dict = metrics.to_dict()
        self.log_metrics_prefix(metrics_dict, "eval")
        sample_fig = samples.plot(str(run.info.run_name))
        mlflow.log_figure(sample_fig, artifact_file="samples.png")
        plt.close(sample_fig)  # otherwise it's kept in memory, well done matplotlib

        return metrics

    def __call__(self, trial: Trial):
        with mlflow.start_run(nested=True, log_system_metrics=True) as run:
            self.current_run = run
            trial.set_user_attr("mlflow_run_id", run.info.run_id)
            mlflow.set_tag("optuna_study", trial.study.study_name)
            mlflow.set_tag("optuna_trial", trial.number)

            mlflow.log_dict(cast(dict, read_params(ensure_dvc=True)), "params.yaml")
            mlflow.log_params(self.hparams_general)

            model = self.get_model(trial)
            self.prune_if_requested(trial)  # check if we should even start training

            self.fit(model)

            metrics = self.evaluate(model, run)

            # this is what optuna optimizes
            return self.get_optim_vars(metrics, model)

    def initial_trials(self) -> Iterator[dict[str, Any]]:
        """A list of sensible optuna parameters to kickstart the study and speed up pruning."""
        return
        yield  # no-op yield to ensure function is turned into an empty generator
