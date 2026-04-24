from typing_extensions import override, Any

from aare.constants import RANDOM_SEED
from aare_train.models.base_trainer import BaseTrainer, ModelType
from aare_train.fetching.feature_identifiers import FeatureIdentifiers
from aare_train.params.params_types import Params
from darts.models import RNNModel
import torch
import torch.nn as nn
from torchmetrics import MetricCollection
import torchmetrics


class GRUTrainer(BaseTrainer):
    """Trainer for GRU models with configurable hyperparameters."""

    def __init__(self, params: Params, features: FeatureIdentifiers):
        super().__init__(params, features)

    # could use TypedDict and Unpack but I don't think there's any benefit here
    @override
    def build_model(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        *,
        input_chunk_length: int = 24,
        hidden_dim: int = 32,
        n_rnn_layers: int = 2,
        dropout: float = 0.0,
        lr: float = 1e-4,
        batch_size: int = 1024,
        max_n_epochs: int = 200,
        add_day_enc: bool = False,
        add_year_enc: bool = False,
        early_stopping_patience: int = 5,
        model_name: str = "GRU",
        pl_trainer_kwargs: dict[str, Any] | None = None,
    ) -> ModelType:
        hparams_model = dict(
            model="GRU",
            input_chunk_length=input_chunk_length,
            hidden_dim=hidden_dim,
            n_rnn_layers=n_rnn_layers,
            dropout=dropout,
            # output_chunk_length is always 1 for RNNs, but at inference time,
            # we want to predict more than 1 data point at once. To better model
            # that behaviour, it takes a training_length and will do
            # training_length - input_chunk_length (= forecast_horizon) steps
            # and combine the loss of all of them before optimizing.
            # https://github.com/unit8co/darts/issues/1397#issuecomment-1331936411
            training_length=self.horizon + input_chunk_length,
            # training kwargs
            model_name=model_name,
            random_state=RANDOM_SEED,
            batch_size=batch_size,
            n_epochs=max_n_epochs,
            optimizer_cls=torch.optim.Adam,
            optimizer_kwargs=dict(lr=lr),
            loss_fn=nn.MSELoss(),
            torch_metrics=MetricCollection(
                {
                    "mae": torchmetrics.MeanAbsoluteError(),
                    "rmse": torchmetrics.MeanSquaredError(squared=False),
                }
            ),
            save_checkpoints=False,
            pl_trainer_kwargs=pl_trainer_kwargs
            or self.get_trainer_params(model_name, early_stopping_patience=early_stopping_patience),
            add_encoders=self.get_add_encoders(add_day_enc, add_year_enc),
        )

        self.log_params_prefix(hparams_model, "model")

        return RNNModel(**hparams_model)  # pyright: ignore[reportArgumentType]
