from abc import ABC, abstractmethod
from typing import Any
from collections.abc import Iterator

from aare_train.models import BaseTrainer
import mlflow
from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel
from darts.models.forecasting.sklearn_model import SKLearnModel
from mlflow import ActiveRun
from optuna import Trial, TrialPruned

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
        self._trainer: BaseTrainer | None = None  # lazily created

    @property
    def trainer(self) -> BaseTrainer:
        """Get lazily created trainer instance."""
        if self._trainer is None:
            self._trainer = self.create_trainer()

        assert self._trainer is not None, "create_trainer returned None!"

        return self._trainer

    @abstractmethod
    def create_trainer(self) -> BaseTrainer:
        """Create the trainer instance for this tuner."""
        pass

    def get_trainer_params(
        self,
        trial: Trial,
        model_name: str,
        early_stopping_var: str | None = "val_loss",
        early_stopping_patience: int = 5,
        pruning_var: str | None = "val_mae",
        log_every_n_steps: int = 50,
    ):
        """Get pytorch lightning trainer parameters with optuna pruning callback."""
        params = self.trainer.get_trainer_params(
            model_name, early_stopping_var, early_stopping_patience, log_every_n_steps
        )

        if pruning_var:
            callbacks = params["callbacks"]
            assert isinstance(callbacks, list)
            callbacks.append(PyTorchLightningPruningCallback(trial, monitor=pruning_var))

        return params

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

            mlflow.log_dict(read_params(ensure_dvc=True).__dict__, "params.yaml")

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
