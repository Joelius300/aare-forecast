import logging
import os

import matplotlib.pyplot as plt
import mlflow
import optuna
import torch
import torchmetrics
from darts.models import RNNModel
from lightning_fabric import seed_everything
from mlflow import ActiveRun
from optuna import Trial
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger
from torch import nn
from torchmetrics import MetricCollection

from aare.compat.optuna_lightning_integration import PyTorchLightningPruningCallback
from aare.evaluation.evaluation import evaluate_model
from aare.evaluation.metrics import Metrics
from aare.feature_identifiers import FeatureIdentifiers
from aare.feature_set import FeatureSet
from aare.features.registry import FEATURES
from aare.normalization import get_scalers
from aare.params import read_params, Params
from aare.utils import OPTUNA_STORE_URI, get_data_stats

MODEL_NAME = "GRU"
RANDOM_SEED = 42


# TODO Extract reusable parts from above and below
class GRUTuning:
    def __init__(self, params: Params, features: FeatureIdentifiers, batch_size: int, max_n_epochs: int):
        ds = FeatureSet(
            targets=[FEATURES[f] for f in features["targets"]],
            future=[FEATURES[f] for f in features["future"]] if "future" in features else None,
            past=[FEATURES[f] for f in features["past"]] if "past" in features else None,
            split_params=params["split"],
        )
        self.train = ds.get_train()
        self.val = ds.get_val()

        self.train_target_subs, self.train_fc_subs = self.train[0], self.train[2]
        self.val_target_subs, self.val_fc_subs = self.val[0], self.val[2]
        self.scalers = get_scalers(self.train_target_subs, train_fc_subs=self.train_fc_subs)
        self.horizon = params["general"]["forecast_horizon"]
        self.stride = params["validation"]["stride"]
        self.min_lookback_hours = params["validation"]["min_lookback_hours"]
        self.batch_size = batch_size
        self.max_n_epochs = max_n_epochs

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

    def get_model(self, trial: Trial, run: mlflow.ActiveRun):
        # seems like the default doesn't work?!? it should take that without setting it explicitly..
        # https://github.com/Lightning-AI/pytorch-lightning/discussions/11197#discussioncomment-9164713
        mlflow_logger = MLFlowLogger(MODEL_NAME, tracking_uri=os.getenv("MLFLOW_TRACKING_URI"), run_id=run.info.run_id)
        early_stopping = EarlyStopping("val_loss", patience=5)
        pruning = PyTorchLightningPruningCallback(trial, monitor="val_mae")

        # how many hours lookback when predicting
        input_chunk_length = trial.suggest_int("input_chunk_length", 1, 24)

        add_encoders = None
        add_hour_enc = trial.suggest_categorical("add_hour_enc", [True, False])
        add_year_enc = trial.suggest_categorical("add_year_enc", [True, False])
        if add_hour_enc or add_year_enc:
            enc = []
            if add_hour_enc:
                enc.append("hour")
            if add_year_enc:
                enc.append("day_of_year")
            add_encoders = {"cyclic": {"future": enc}}
        hparams_model = dict(
            model=MODEL_NAME,
            input_chunk_length=input_chunk_length,
            hidden_dim=trial.suggest_int("hidden_dim", 1, 256),
            dropout=trial.suggest_float("dropout", 0.0, 0.5),
            n_rnn_layers=trial.suggest_int("n_rnn_layers", 1, 10),
            # output_chunk_length is always 1 for RNNs, but at inference time,
            # we want to predict more than 1 data point at once. To better model
            # that behaviour, it takes a training_length and will do
            # training_length - input_chunk_length (= forecast_horizon) steps
            # and combine the loss of all of them before optimizing.
            # https://github.com/unit8co/darts/issues/1397#issuecomment-1331936411
            training_length=self.horizon + input_chunk_length,
            # training kwargs
            model_name=MODEL_NAME,
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
            pl_trainer_kwargs={
                "logger": mlflow_logger,
                "callbacks": [early_stopping, pruning],
                "log_every_n_steps": 50,
            },
            add_encoders=add_encoders,
        )

        mlflow.log_params({"model_" + key: value for key, value in hparams_model.items()})

        return RNNModel(**hparams_model)  # pyright: ignore [reportArgumentType]

    def fit(self, model: RNNModel):
        assert "series" in self.scalers
        assert "future_covariates" in self.scalers
        assert self.train_fc_subs is not None
        assert self.val_fc_subs is not None

        orig_series = len(self.train_target_subs)
        orig_series_len = sum((len(x) for x in self.train_target_subs))

        # some training series might be too short for the model if it needs 4 days for val and 1 day lookback e.g.
        self.train_target_subs = [x for x in self.train_target_subs if len(x) >= model.training_length]
        self.train_fc_subs = [x for x in self.train_fc_subs if len(x) >= model.training_length]
        # val should not have this issue, but technically, we should do it there too
        data_stats = get_data_stats(self.train_target_subs, self.val_target_subs)
        data_stats["sub_series_dropped"] = orig_series - data_stats["train_n_subs"]
        data_stats["sub_series_dropped_len"] = orig_series_len - data_stats["train_len_total"]

        mlflow.log_params({"data_" + key: value for key, value in data_stats.items()})

        scaler_target = self.scalers["series"]
        scaler_fc = self.scalers["future_covariates"]

        return model.fit(
            series=scaler_target.transform(self.train_target_subs),
            future_covariates=scaler_fc.transform(self.train_fc_subs),
            val_series=scaler_target.transform(self.val_target_subs),
            val_future_covariates=scaler_fc.transform(self.val_fc_subs),
        )

    def evaluate(self, model: RNNModel, run: ActiveRun) -> Metrics:
        metrics, samples = evaluate_model(
            model,
            self.val_target_subs,
            self.horizon,
            self.stride,
            self.min_lookback_hours,
            future_cov=self.val_fc_subs,
            data_transformers=self.scalers,
        )

        mlflow.log_metrics({f"eval_{k}": v for k, v in metrics.to_dict().items()})
        mlflow.log_figure(samples.plot(str(run.info.run_name)), artifact_file="samples.png")

        return metrics

    def __call__(self, trial: Trial):
        with mlflow.start_run(nested=True, log_system_metrics=True) as run:
            trial.set_user_attr("mlflow_run_id", run.info.run_id)
            mlflow.set_tag("optuna_study", trial.study.study_name)
            mlflow.set_tag("optuna_trial", trial.number)
            mlflow.log_params(self.hparams_general)
            model = self.get_model(trial, run)
            self.fit(model)
            metrics = self.evaluate(model, run)

            # this is what will be minimized by optuna
            # TODO also minimize nr of params?
            # TODO also minimize other metrics like (TODO) mean abs peak diff and time-weighted (day&year) MAE/RMSE
            return metrics.mae


def main():
    params = read_params()

    features: FeatureIdentifiers = {
        "targets": ["temp_bern"],
        "future": [
            "tt_bern",
            "flow_bern",
            "ss_bern",
        ],
    }

    batch_size = 1024
    max_n_epochs = 200
    name = "tune-gru"

    study = optuna.create_study(
        study_name=name,
        direction="minimize",
        sampler=TPESampler(),
        pruner=HyperbandPruner(max_resource=max_n_epochs),
        storage=OPTUNA_STORE_URI,
        load_if_exists=True,  # resume
    )

    mlflow.set_experiment(MODEL_NAME)
    with mlflow.start_run(run_name=name, description="Tune hparams of GRU without changing features") as parent_run:
        mlflow.set_tag("optuna_study", study.study_name)
        study.set_user_attr("mlflow_exp_id", parent_run.info.experiment_id)
        study.set_user_attr("mlflow_parent_run_id", parent_run.info.run_id)

        # Stop fake "running" trial and re-queue them (they are left when cancelling with ctrl+c).
        # Of course this won't work in a distributed setting where multiple runs could actually be running etc.
        # TODO Guard with a semaphore and add logs. For _big_ distributed settings, you wouldn't use sqlite.
        # TODO add initial trials to speed up search, esp. pruning, makes a big difference.
        # TODO Set it up so that different models can be tried by optuna and enqueue at least one trial per model.
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.RUNNING:
                logging.info(f"Failing and re-queuing previously running trial {trial.number}")
                study.enqueue_trial(trial.params)
                study.tell(trial.number, state=optuna.trial.TrialState.FAIL)

        # TODO add callback to store new best model (here's probably best place, but idk) with
        study.optimize(GRUTuning(params, features, batch_size, max_n_epochs))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    torch.set_float32_matmul_precision("medium")
    seed_everything(RANDOM_SEED)
    plt.rcParams["figure.figsize"] = (16, 9)
    mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")

    main()
