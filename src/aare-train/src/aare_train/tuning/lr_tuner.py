from collections.abc import Sequence
from typing_extensions import override

from aare_train.evaluation.eval_metric import EvalMetric
from aare_train.fetching.feature_identifiers import FeatureIdentifiers
from aare_train.params.params_types import Params
from aare_train.tuning.base_tuner import BaseTuner, ModelType
from aare_train.models.lr_trainer import LRTrainer, RegularizationType
from optuna import Study, Trial
from optuna.trial import FrozenTrial
from optuna.trial._state import TrialState
from optuna.pruners import BasePruner


class DuplicateLagsPruner(BasePruner):
    def _get_transformed_params(self, trial: Trial | FrozenTrial):
        params = {**trial.params}

        del params["lag_max"]
        del params["lag_step"]
        params["lags_raw"] = trial.user_attrs.get("lags_raw")

        return params

    def _params_equal(self, trial_a: Trial | FrozenTrial, trial_b: Trial | FrozenTrial):
        return self._get_transformed_params(trial_a) == self._get_transformed_params(trial_b)

    def prune(self, study: Study, trial: FrozenTrial) -> bool:
        completed_trials = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
        for comp_trial in completed_trials:
            if self._params_equal(comp_trial, trial):
                return True

        return False


class LRTuner(BaseTuner):
    def __init__(
        self,
        model_name: str,
        params: Params,
        features: FeatureIdentifiers,
        enabled_regularizations: Sequence[RegularizationType] | RegularizationType = "none",
    ):
        super().__init__(model_name, params, features)
        self.enabled_regularizations = enabled_regularizations

    @override
    def create_trainer(self):
        return LRTrainer(self.params, self.features)

    @override
    def get_model(self, trial: Trial) -> ModelType:
        # suggest hyperparameters
        lag_max = trial.suggest_int("lag_max", 22, 24)
        lag_step = trial.suggest_int("lag_step", 1, 3)
        output_chunk_length = trial.suggest_int("output_chunk_length", 1, 1, log=True)
        add_day_enc = trial.suggest_categorical("add_day_enc", [True, False])
        add_year_enc = trial.suggest_categorical("add_year_enc", [True, False])

        # store lags for duplicate pruner
        lags_raw = list(range(-lag_max, -1, lag_step))
        trial.set_user_attr("lags_raw", lags_raw)

        # determine regularization
        if isinstance(self.enabled_regularizations, str):
            regularization = self.enabled_regularizations
        else:
            assert len(self.enabled_regularizations) > 1, (
                "Passed a list of regularizations to try, but only one element!"
            )
            regularization = trial.suggest_categorical("regularization", self.enabled_regularizations)

        # suggest alpha and l1_ratio based on regularization type
        alpha = None
        l1_ratio = None
        if regularization != "none":
            alpha = trial.suggest_float("alpha", 0.01, 2, log=True)
        if regularization == "elastic":
            l1_ratio = trial.suggest_float("l1_ratio", 0.05, 1, step=0.05)

        # build model via trainer
        return self.trainer.build_model(
            lag_max=lag_max,
            lag_step=lag_step,
            output_chunk_length=output_chunk_length,
            regularization=regularization,
            alpha=alpha,
            l1_ratio=l1_ratio,
            add_day_enc=add_day_enc,
            add_year_enc=add_year_enc,
            model_name=self.model_name,
        )

    @override
    def get_optim_vars(self, metrics: EvalMetric, model: ModelType):
        # see comments in GRUTuner
        return metrics.mae

    @override
    def initial_trials(self):
        yield {
            "lag_max": 1,
            "lag_step": 1,
            "add_day_enc": False,
            "add_year_enc": False,
            "output_chunk_length": 1,
            "regularization": "none",
        }

        yield {
            "lag_max": 24,
            "lag_step": 6,
            "add_day_enc": False,
            "add_year_enc": False,
            "output_chunk_length": 1,
            "regularization": "none",
        }

        yield {
            "lag_max": 4,
            "lag_step": 3,
            "add_day_enc": False,
            "add_year_enc": False,
            "output_chunk_length": 1,
            "regularization": "lasso",
            "alpha": 0.1,
        }

        yield {
            "lag_max": 4,
            "lag_step": 3,
            "add_day_enc": False,
            "add_year_enc": False,
            "output_chunk_length": 1,
            "regularization": "elastic",
            "alpha": 0.01,
            "l1_ratio": 0.5,
        }
