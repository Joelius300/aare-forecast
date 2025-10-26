from typing import override

from aare.constants import RANDOM_SEED
from aare.evaluation.metrics import Metrics
from aare.feature_identifiers import FeatureIdentifiers
from aare.params.params_types import Params
from aare.tuning.base_tuner import BaseTuner, ModelType
from optuna import Trial
from darts.models import RNNModel
import torch
import torch.nn as nn
from torchmetrics import MetricCollection
import torchmetrics


class GRUTuner(BaseTuner):
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
        input_chunk_length = trial.suggest_int("input_chunk_length", 1, 48)

        hparams_model = dict(
            model="GRU",
            input_chunk_length=input_chunk_length,
            # log=True just means lower values are sampled more often
            hidden_dim=trial.suggest_int("hidden_dim", 1, 256, log=True),
            n_rnn_layers=trial.suggest_int("n_rnn_layers", 1, 10),
            dropout=trial.suggest_float("dropout", 0.0, 0.4),
            # output_chunk_length is always 1 for RNNs, but at inference time,
            # we want to predict more than 1 data point at once. To better model
            # that behaviour, it takes a training_length and will do
            # training_length - input_chunk_length (= forecast_horizon) steps
            # and combine the loss of all of them before optimizing.
            # https://github.com/unit8co/darts/issues/1397#issuecomment-1331936411
            training_length=self.horizon + input_chunk_length,
            # training kwargs
            model_name=self.model_name,
            random_state=RANDOM_SEED,
            batch_size=self.batch_size,
            n_epochs=self.max_n_epochs,
            # TODO try different optimizers
            optimizer_cls=torch.optim.Adam,
            optimizer_kwargs=dict(
                lr=trial.suggest_float("lr", 1e-7, 1e-1, log=True),
            ),
            # lr_scheduler_cls=
            # lr_scheduler_kwargs=
            # either loss or likelihood
            # TODO do more trials with likelihood regression
            loss_fn=nn.MSELoss(),
            # likelihood=QuantileRegression([0.25, 0.5, 0.75]),
            torch_metrics=MetricCollection(
                {
                    "mae": torchmetrics.MeanAbsoluteError(),
                    "rmse": torchmetrics.MeanSquaredError(squared=False),
                }
            ),
            save_checkpoints=False,
            pl_trainer_kwargs=self.get_trainer_params(trial),
            add_encoders=self.suggest_add_encoders(trial),
        )

        self.log_params_prefix(hparams_model, "model")

        # even with kwargs-only it can't figure out that this is correct..
        return RNNModel(**hparams_model)  # pyright: ignore [reportArgumentType]

    @override
    def get_optim_vars(self, metrics: Metrics, model: ModelType):
        # this is what will be minimized by optuna
        # TODO also minimize nr of params? will need a optim_direction property in base_tuner when doing multi-obj
        # TODO also minimize other metrics like mean abs peak diff and time-weighted (day&year) MAE/RMSE
        #  An interesting combo might be RMSE (penalizes larger errors more than smaller ones, can lead to flatter forecasts if timing is hard)
        #  together with MADPD (very forgiving on timing but magnitude is important).
        return metrics.mae

    @override
    def initial_trials(self):
        small = {
            "input_chunk_length": 1,
            "hidden_dim": 16,
            "rnn_layers": 1,
            "dropout": 0,
            "lr": 1e-4,
            "add_day_enc": False,
            "add_year_enc": False,
        }
        medium = {
            "input_chunk_length": 24,
            "hidden_dim": 32,
            "rnn_layers": 2,
            "dropout": 0.05,
            "lr": 1e-4,
            "add_day_enc": False,
            "add_year_enc": False,
        }

        yield small
        yield medium
        yield medium | {"lr": 1e-3}
        yield medium | {"lr": 1e-5}
