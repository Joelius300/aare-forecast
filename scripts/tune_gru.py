import logging

from aare.constants import RANDOM_SEED
from aare.tuning.gru_tuner import GRUTuner
import matplotlib.pyplot as plt
import mlflow
import optuna
import torch
from lightning_fabric import seed_everything
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler

from aare.fetching.feature_identifiers import FeatureIdentifiers
from aare.params import read_params
from aare.paths import OPTUNA_STORE_URI

logger = logging.getLogger(__name__)


def main():
    params = read_params()

    features: FeatureIdentifiers = {
        "targets": ["temp_bern"],
        "future": [
            "tt_bern",
            "ss_bern",
            "rr_bern",
            "rh_bern",
            "wind_bern",
        ],
    }

    # adjust to fit in memory, but remember that often too large batch sizes degrade performance (unintuitively)
    batch_size = 1024
    max_n_epochs = 200

    model_name = "GRU"
    run_name = "tune-gru"

    tuner = GRUTuner(model_name, params, features, batch_size, max_n_epochs)

    study = optuna.create_study(
        study_name=run_name,
        direction="minimize",
        # https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html#which-sampler-and-pruner-should-be-used
        sampler=TPESampler(),
        pruner=HyperbandPruner(max_resource=max_n_epochs),
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

        # Stop fake "running" trial and re-queue them (they are left when cancelling with ctrl+c).
        # Of course this won't work in a distributed setting where multiple runs could actually be running etc.
        # TODO Guard with a semaphore and add logs. For _big_ distributed settings, you wouldn't use sqlite (nor TPE).
        # TODO add initial trials to speed up search, esp. pruning, makes a big difference.
        # TODO Set it up so that different models can be tried by optuna and enqueue at least one trial per model.
        trials = study.trials
        for trial in trials:
            if trial.state == optuna.trial.TrialState.RUNNING:
                logging.info(f"Failing and re-queuing previously running trial {trial.number}")
                study.enqueue_trial(trial.params, user_attrs={"restart_of": trial.number})
                study.tell(trial.number, state=optuna.trial.TrialState.FAIL)

        # TODO enqueue all trials that have not been tried yet instead of guarding for a fresh study
        if not trials:
            logger.info("Fresh study; enqueuing initial trials")
            for params in tuner.initial_trials():
                study.enqueue_trial(params, user_attrs={"initial_trial_of": tuner.model_name})

        # TODO add callback to store new best model (here's probably best place, but idk)
        #  alternatively, store all models and metrics from finished models (unpromising should be pruned anyway).
        study.optimize(tuner)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    torch.set_float32_matmul_precision("medium")
    seed_everything(RANDOM_SEED)
    plt.rcParams["figure.figsize"] = (16, 9)
    mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")

    # TODO add some "force initial" to always suggest initial trials maybe?
    # TODO add simple way to train/repro a single model with specific parameters (and store all artifacts) (no pruning).
    main()
