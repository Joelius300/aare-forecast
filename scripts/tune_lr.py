import itertools
import logging

from aare.constants import RANDOM_SEED
from aare.tuning.lr_tuner import DuplicateLagsPruner, LRTuner
import matplotlib.pyplot as plt
import mlflow
import optuna
import torch
from lightning_fabric import seed_everything
from optuna.samplers import TPESampler

from aare.feature_identifiers import FeatureIdentifiers
from aare.params import read_params
from aare.paths import OPTUNA_STORE_URI

logger = logging.getLogger(__name__)


def main():
    params = read_params()

    features: FeatureIdentifiers = {
        "targets": ["temp_bern"],
        "future": [
            f"{feat}_bern{suffix}"
            for feat, suffix in itertools.product(
                ["tt", "ss", "rr", "rh", "wind"],
                ["", "_log", "_sq", "_cube", "_sqrt", "_ma3", "_ma6", "_ma12", "_diff", "_diff_sq"],
            )
        ],
    }

    model_name = "LR"
    run_name = "tune-LR-ridge"

    tuner = LRTuner(model_name, params, features, enabled_regularizations="ridge")

    study = optuna.create_study(
        study_name=run_name,
        direction="minimize",
        # https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html#which-sampler-and-pruner-should-be-used
        sampler=TPESampler(),
        pruner=DuplicateLagsPruner(),
        storage=OPTUNA_STORE_URI,
        load_if_exists=True,  # resume
    )

    mlflow.set_experiment(tuner.model_name)
    with mlflow.start_run(
        run_name=run_name, description=f"Tune hparams of {tuner.model_name} without changing features"
    ) as parent_run:
        mlflow.set_tag("optuna_study", study.study_name)
        study.set_user_attr("mlflow_exp_id", parent_run.info.experiment_id)
        study.set_user_attr("mlflow_parent_run_id", parent_run.info.run_id)

        trials = study.trials
        for trial in trials:
            if trial.state == optuna.trial.TrialState.RUNNING:
                logging.info(f"Failing and re-queuing previously running trial {trial.number}")
                study.enqueue_trial(trial.params, user_attrs={"restart_of": trial.number})
                study.tell(trial.number, state=optuna.trial.TrialState.FAIL)

        if not trials:
            logger.info("Fresh study; enqueuing initial trials")
            for params in tuner.initial_trials():
                study.enqueue_trial(params, user_attrs={"initial_trial_of": tuner.model_name})

        study.optimize(tuner)


# TODO see todos and improvement ideas in tune_gru
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    torch.set_float32_matmul_precision("medium")
    seed_everything(RANDOM_SEED)
    plt.rcParams["figure.figsize"] = (16, 9)
    mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")

    main()
