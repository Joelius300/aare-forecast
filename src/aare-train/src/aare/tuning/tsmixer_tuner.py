from typing import override

from aare.evaluation.eval_metric import EvalMetric
from aare.fetching.feature_identifiers import FeatureIdentifiers
from aare.params.params_types import Params
from aare.tuning.base_tuner import BaseTuner, ModelType
from optuna import Trial
from darts.models import TSMixerModel
import torch
import torch.nn as nn
from torchmetrics import MetricCollection
import torchmetrics


# TODO: see todos in gru_tuner
class TSMixerTuner(BaseTuner):
    def __init__(
        self, model_name: str, params: Params, features: FeatureIdentifiers, batch_size: int, max_n_epochs: int
    ):
        super().__init__(model_name, params, features)
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_n_epochs = max_n_epochs

    @override
    def get_model(self, trial: Trial) -> ModelType:
        # how many hours lookback when predicting
        input_chunk_length = trial.suggest_int("input_chunk_length", 1, 48, step=6)
        output_chunk_length = trial.suggest_int("output_chunk_length", 1, 48, step=6)

        hparams_model = {
            "input_chunk_length": input_chunk_length,
            "output_chunk_length": output_chunk_length,
            "hidden_size": trial.suggest_int("hidden_size", 1, 256, log=True),
            "ff_size": trial.suggest_int("ff_size", 1, 256, log=True),
            "num_blocks": trial.suggest_int("num_blocks", 1, 10),
            "activation": "ReLU",
            "dropout": trial.suggest_float("dropout", 0.0, 0.4),
            "norm_type": "LayerNorm",
            "normalize_before": False,
            "use_static_covariates": False,  # currently no static covariates
            "add_encoders": self.suggest_add_encoders(trial),
            "batch_size": self.batch_size,
            "n_epochs": self.max_n_epochs,
            # TODO try different optimizers
            "optimizer_cls": torch.optim.Adam,
            "optimizer_kwargs": dict(
                lr=trial.suggest_float("lr", 1e-7, 1e-1, log=True),
            ),
            # lr_scheduler_cls=
            # lr_scheduler_kwargs=
            # either loss or likelihood
            # TODO do more trials with likelihood regression
            "loss_fn": nn.MSELoss(),
            # likelihood=QuantileRegression([0.25, 0.5, 0.75]),
            "torch_metrics": MetricCollection(
                {
                    "mae": torchmetrics.MeanAbsoluteError(),
                    "rmse": torchmetrics.MeanSquaredError(squared=False),
                }
            ),
            "save_checkpoints": False,
            "pl_trainer_kwargs": self.get_trainer_params(trial),
        }

        self.log_params_prefix(hparams_model, "model")

        # even with kwargs-only it can't figure out that this is correct..
        return TSMixerModel(**hparams_model)  # pyright: ignore [reportArgumentType]

    @override
    def get_optim_vars(self, metrics: EvalMetric, model: ModelType):
        # this is what will be minimized by optuna
        return metrics.mae

    @override
    def initial_trials(self):
        small = {
            "input_chunk_length": 24,
            "output_chunk_length": 24,
            "hidden_size": 16,
            "ff_size": 16,
            "num_blocks": 1,
            "dropout": 0,
            "lr": 1e-4,
            "add_day_enc": False,
            "add_year_enc": False,
        }
        medium = {
            "input_chunk_length": 24,
            "output_chunk_length": 24,
            "hidden_size": 64,
            "ff_size": 64,
            "num_blocks": 2,
            "dropout": 0.1,
            "lr": 1e-4,
            "add_day_enc": False,
            "add_year_enc": False,
        }

        yield small
        yield medium
