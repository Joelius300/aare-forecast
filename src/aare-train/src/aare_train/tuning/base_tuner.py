from abc import ABC, abstractmethod
import os
from typing import Any, cast
from collections.abc import Iterator, Mapping

import mlflow
from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel
from darts.models.forecasting.sklearn_model import SKLearnModel
from mlflow import ActiveRun
from optuna import Trial, TrialPruned
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger

from aare_train.compat.optuna_lightning_integration import PyTorchLightningPruningCallback
from aare_train.evaluation.eval_metric import EvalMetric
from aare_train.fetching.feature_identifiers import FeatureIdentifiers
from aare_train.params import read_params, Params

ModelType = TorchForecastingModel | SKLearnModel


class BaseTuner(ABC):
    def __init__(self, model_name: str, params: Params, features: FeatureIdentifiers):
        self.model_name = model_name
        self.params = params
        self.features = features
        self.current_run: ActiveRun | None = None
        self._trainer = None  # lazily created

        # compute hparams_general from params (no data fetching)
        self.horizon = params["general"]["forecast_horizon"]
        self.hparams_general = {
            "horizon": self.horizon,
            "val_stride": params["validation"]["stride"],
            "split_train": params["split"]["train_split"],
            "split_val": params["split"]["val_split"],
            "split_test": params["split"]["test_split"],
            "features_targets": features["targets"],
            "features_future": features.get("future", []),
            "features_past": features.get("past", []),
        }

    @property
    def trainer(self):
        """Lazily create trainer once."""
        if self._trainer is None:
            self._trainer = self.create_trainer()
        return self._trainer

    @abstractmethod
    def create_trainer(self):
        """Create the trainer instance for this tuner."""
        pass

    def get_trainer_params_with_pruning(
        self,
        trial: Trial,
        model_name: str,
        early_stopping_var: str | None = "val_loss",
        early_stopping_patience: int = 5,
        pruning_var: str | None = "val_mae",
        log_every_n_steps: int = 50,
    ):
        """Get pytorch lightning trainer parameters with optuna pruning callback."""
        assert self.current_run is not None
        # they fucked up their default which is evaluated at the import of the module, so set_tracking_uri is ignored
        # https://github.com/Lightning-AI/pytorch-lightning/discussions/11197#discussioncomment-9164713
        mlflow_logger = MLFlowLogger(
            model_name, tracking_uri=os.getenv("MLFLOW_TRACKING_URI"), run_id=self.current_run.info.run_id
        )

        callbacks = []
        if early_stopping_var:
            callbacks.append(EarlyStopping(early_stopping_var, patience=early_stopping_patience))

        if pruning_var:
            callbacks.append(PyTorchLightningPruningCallback(trial, monitor=pruning_var))

        return {
            "logger": mlflow_logger,
            "callbacks": callbacks,
            "log_every_n_steps": log_every_n_steps,
        }

    @staticmethod
    def _prefix_dict(vals: Mapping[str, Any], prefix: str):
        prefix = prefix.removesuffix("_")
        return {prefix + "_" + key: value for key, value in vals.items()}

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

            # use trainer methods
            self.trainer.fit(model)
            metrics = self.trainer.evaluate(model, run)

            # this is what optuna optimizes
            return self.get_optim_vars(metrics, model)

    def initial_trials(self) -> Iterator[dict[str, Any]]:
        """A list of sensible optuna parameters to kickstart the study and speed up pruning."""
        return
        yield  # no-op yield to ensure function is turned into an empty generator
