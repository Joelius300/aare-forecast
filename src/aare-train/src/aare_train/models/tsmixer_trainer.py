from typing import override
import os

from aare_train.models.base_trainer import BaseTrainer, ModelType
from aare_train.fetching.feature_identifiers import FeatureIdentifiers
from aare_train.params.params_types import Params
from darts.models import TSMixerModel
import torch
import torch.nn as nn
from torchmetrics import MetricCollection
import torchmetrics
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger
import mlflow


class TSMixerTrainer(BaseTrainer):
    """Trainer for TSMixer models with configurable hyperparameters."""

    def __init__(
        self,
        params: Params,
        features: FeatureIdentifiers,
        *,
        input_chunk_length: int = 24,
        output_chunk_length: int = 24,
        hidden_size: int = 64,
        ff_size: int = 64,
        num_blocks: int = 2,
        dropout: float = 0.1,
        lr: float = 1e-4,
        batch_size: int = 1024,
        max_n_epochs: int = 200,
        add_day_enc: bool = False,
        add_year_enc: bool = False,
        early_stopping_patience: int = 5,
        model_name: str = "TSMixer",
    ):
        super().__init__(params, features)
        self.model_name = model_name
        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length
        self.hidden_size = hidden_size
        self.ff_size = ff_size
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.max_n_epochs = max_n_epochs
        self.add_day_enc = add_day_enc
        self.add_year_enc = add_year_enc
        self.early_stopping_patience = early_stopping_patience

    def _get_add_encoders(self) -> dict[str, dict[str, list[str]]] | None:
        """Get the add_encoders configuration."""
        if not self.add_day_enc and not self.add_year_enc:
            return None

        enc = []
        if self.add_day_enc:
            enc.append("hour")
        if self.add_year_enc:
            enc.append("day_of_year")

        return {"cyclic": {"future": enc}}

    def _get_trainer_params(self):
        """Get the pytorch lightning trainer parameters."""
        mlflow_logger = MLFlowLogger(
            self.model_name, tracking_uri=os.getenv("MLFLOW_TRACKING_URI"), run_id=mlflow.active_run().info.run_id
        )

        callbacks = [EarlyStopping("val_loss", patience=self.early_stopping_patience)]

        return {
            "logger": mlflow_logger,
            "callbacks": callbacks,
            "log_every_n_steps": 50,
        }

    @override
    def build_model(self) -> ModelType:
        hparams_model = {
            "input_chunk_length": self.input_chunk_length,
            "output_chunk_length": self.output_chunk_length,
            "hidden_size": self.hidden_size,
            "ff_size": self.ff_size,
            "num_blocks": self.num_blocks,
            "activation": "ReLU",
            "dropout": self.dropout,
            "norm_type": "LayerNorm",
            "normalize_before": False,
            "use_static_covariates": False,
            "add_encoders": self._get_add_encoders(),
            "batch_size": self.batch_size,
            "n_epochs": self.max_n_epochs,
            "optimizer_cls": torch.optim.Adam,
            "optimizer_kwargs": dict(lr=self.lr),
            "loss_fn": nn.MSELoss(),
            "torch_metrics": MetricCollection(
                {
                    "mae": torchmetrics.MeanAbsoluteError(),
                    "rmse": torchmetrics.MeanSquaredError(squared=False),
                }
            ),
            "save_checkpoints": False,
            "pl_trainer_kwargs": self._get_trainer_params(),
        }

        self.log_params_prefix(hparams_model, "model")

        return TSMixerModel(**hparams_model)  # pyright: ignore[reportArgumentType]
