from typing_extensions import override, Any

from aare_train.models.base_trainer import BaseTrainer, ModelType
from aare_train.fetching.feature_identifiers import FeatureIdentifiers
from aare_train.params.params_types import Params
from darts.models import TSMixerModel
import torch
import torch.nn as nn
from torchmetrics import MetricCollection
import torchmetrics


class TSMixerTrainer(BaseTrainer):
    """Trainer for TSMixer models with configurable hyperparameters."""

    def __init__(self, params: Params, features: FeatureIdentifiers):
        super().__init__(params, features)

    @override
    def build_model(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
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
        pl_trainer_kwargs: dict[str, Any] | None = None,
    ) -> ModelType:
        hparams_model = {
            "input_chunk_length": input_chunk_length,
            "output_chunk_length": output_chunk_length,
            "hidden_size": hidden_size,
            "ff_size": ff_size,
            "num_blocks": num_blocks,
            "activation": "ReLU",
            "dropout": dropout,
            "norm_type": "LayerNorm",
            "normalize_before": False,
            "use_static_covariates": False,
            "add_encoders": self.get_add_encoders(add_day_enc, add_year_enc),
            "batch_size": batch_size,
            "n_epochs": max_n_epochs,
            "optimizer_cls": torch.optim.Adam,
            "optimizer_kwargs": dict(lr=lr),
            "loss_fn": nn.MSELoss(),
            "torch_metrics": MetricCollection(
                {
                    "mae": torchmetrics.MeanAbsoluteError(),
                    "rmse": torchmetrics.MeanSquaredError(squared=False),
                }
            ),
            "save_checkpoints": False,
            "pl_trainer_kwargs": pl_trainer_kwargs
            or self.get_trainer_params(model_name, early_stopping_patience=early_stopping_patience),
        }

        self.log_params_prefix(hparams_model, "model")

        return TSMixerModel(**hparams_model)  # pyright: ignore[reportArgumentType]
