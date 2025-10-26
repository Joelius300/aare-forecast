from typing import override

from aare.constants import RANDOM_SEED
from aare.evaluation.metrics import Metrics
from aare.tuning.base_tuner import BaseTuner, ModelType
from optuna import Study, Trial
from optuna.trial import FrozenTrial
from optuna.trial._state import TrialState
from optuna.pruners import BasePruner
from darts.models import LinearRegressionModel


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


# TODO transform into sklearn regression tuner and inlcude ridge, lasso, etc. (NNG, elastic net)
# https://scikit-learn.org/stable/api/sklearn.linear_model.html
class LRTuner(BaseTuner):
    @override
    def get_model(self, trial: Trial) -> ModelType:
        lag_max = trial.suggest_int("lag_max", 1, 24)
        lag_step = trial.suggest_int("lag_step", 1, 12)
        output_chunk_length = trial.suggest_int("output_chunk_length", 1, 24, log=True)

        lags_raw = list(range(-lag_max, -1, lag_step))
        trial.set_user_attr("lags_raw", lags_raw)

        hparams_model = {
            "lags": [-1, *lags_raw],
            "lags_future_covariates": [0, -1, *lags_raw],
            "likelihood": None,  # "quantile"
            "quantiles": [0.25, 0.5, 0.75],
            "random_state": RANDOM_SEED,
            "output_chunk_length": output_chunk_length,
            "use_static_covariates": False,  # currently no static covariates
            "multi_models": True,  # 1 model for each output step (doesn't matter if out_chunk_length is 1 anyway)
            "add_encoders": self.suggest_add_encoders(trial),
        }

        self.log_params_prefix(hparams_model, "model")

        # even with kwargs-only it can't figure out that this is correct..
        return LinearRegressionModel(**hparams_model)  # pyright: ignore [reportArgumentType]

    @override
    def get_optim_vars(self, metrics: Metrics, model: ModelType):
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
        }

        yield {
            "lag_max": 24,
            "lag_step": 6,
            "add_day_enc": False,
            "add_year_enc": False,
            "output_chunk_length": 1,
        }
