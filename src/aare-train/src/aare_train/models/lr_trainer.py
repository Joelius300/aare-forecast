from typing import Literal, override

from aare.constants import RANDOM_SEED
from aare_train.models.base_trainer import BaseTrainer, ModelType
from aare_train.fetching.feature_identifiers import FeatureIdentifiers
from aare_train.params.params_types import Params
from darts.models import SKLearnModel
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge


RegularizationType = Literal["none", "lasso", "ridge", "elastic"]
regularization_model_classes = {"none": LinearRegression, "lasso": Lasso, "ridge": Ridge, "elastic": ElasticNet}


class LRTrainer(BaseTrainer):
    """Trainer for linear regression models with configurable hyperparameters."""

    def __init__(
        self,
        params: Params,
        features: FeatureIdentifiers,
        *,
        lag_max: int = 24,
        lag_step: int = 6,
        output_chunk_length: int = 1,
        regularization: RegularizationType = "none",
        alpha: float | None = None,
        l1_ratio: float | None = None,
        add_day_enc: bool = False,
        add_year_enc: bool = False,
        model_name: str = "LR",
    ):
        super().__init__(params, features)
        self.model_name = model_name
        self.lag_max = lag_max
        self.lag_step = lag_step
        self.output_chunk_length = output_chunk_length
        self.regularization = regularization
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.add_day_enc = add_day_enc
        self.add_year_enc = add_year_enc

        # validate hyperparameters
        if regularization in ["lasso", "ridge", "elastic"] and alpha is None:
            raise ValueError(f"alpha must be specified for regularization={regularization}")
        if regularization == "elastic" and l1_ratio is None:
            raise ValueError("l1_ratio must be specified for elastic regularization")

    @override
    def build_model(self) -> ModelType:
        lags_raw = list(range(-self.lag_max, -1, self.lag_step))

        hparams_sk_model = {}
        if self.regularization != "none":
            assert self.alpha is not None
            hparams_sk_model["alpha"] = self.alpha
        else:
            hparams_sk_model["n_jobs"] = -1

        if self.regularization == "elastic":
            assert self.l1_ratio is not None
            hparams_sk_model["l1_ratio"] = self.l1_ratio

        hparams_darts_model = {
            "lags": [-1, *lags_raw],
            "lags_future_covariates": [0, -1, *lags_raw],
            "random_state": RANDOM_SEED,
            "output_chunk_length": self.output_chunk_length,
            "use_static_covariates": False,
            "multi_models": True,
            "add_encoders": self.get_add_encoders(self.add_day_enc, self.add_year_enc),
        }

        self.log_params_prefix(
            hparams_sk_model | hparams_darts_model | dict(regularization=self.regularization), "model"
        )

        sk_model = regularization_model_classes[self.regularization](**hparams_sk_model)
        return SKLearnModel(model=sk_model, **hparams_darts_model)
