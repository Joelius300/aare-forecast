import argparse
import logging

from aare.constants import RANDOM_SEED
from aare_train.fetching.feature_identifiers import FeatureIdentifiers
from aare_train.tuning.gru_tuner import GRUTuner
from aare_train.tuning.lr_tuner import LRTuner, DuplicateLagsPruner
from aare_train.tuning.tsmixer_tuner import TSMixerTuner
from aare_train.params import read_params
from aare_train.paths import OPTUNA_STORE_URI
import matplotlib.pyplot as plt
import mlflow
import optuna
import torch
from lightning_fabric import seed_everything
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler

logger = logging.getLogger(__name__)


def parse_features(targets: list[str], future: list[str] | None) -> FeatureIdentifiers:
    """Parse feature lists into FeatureIdentifiers."""
    return {
        "targets": targets,
        "future": future or [],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run hyperparameter tuning for a model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Tune GRU model
  python scripts/tune.py gru --targets temp_bern --future tt_bern ss_bern rr_bern

  # Tune LR model with custom batch size
  python scripts/tune.py lr --targets temp_bern --future tt_bern ss_bern --batch-size 2048

  # Tune TSMixer model with custom study name
  python scripts/tune.py tsmixer --targets temp_bern --future tt_bern --study-name my-tsmixer-study
        """,
    )

    # TODO consolidate common args with tune.py into some importable helper function and type them
    parser.add_argument(
        "model_type",
        choices=["gru", "lr", "tsmixer"],
        help="Type of model to tune",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        required=True,
        help="Target features (e.g., temp_bern)",
    )
    parser.add_argument(
        "--future",
        nargs="+",
        help="Future covariate features (e.g., tt_bern ss_bern)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Batch size for training (default: 1024, only for GRU/TSMixer)",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=200,
        help="Maximum number of epochs (default: 200, only for GRU/TSMixer)",
    )
    parser.add_argument(
        "--study-name",
        help="Custom Optuna study name (default: tune-{model_type})",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        help="Number of trials to run (default: unlimited, stop with Ctrl+C)",
    )
    # todo think about what you want to do with this
    parser.add_argument(
        "--lr-regularization",
        choices=["none", "lasso", "ridge", "elastic"],
        default="ridge",
        help="Regularization type for LR (default: ridge, only for LR)",
    )

    args = parser.parse_args()

    # parse features
    features = parse_features(args.targets, args.future)
    params = read_params()

    # set default study name
    study_name = args.study_name or f"tune-{args.model_type}"
    model_name = args.model_type.upper()

    # create tuner based on model type
    logger.info(f"Setting up tuner for {model_name}")

    if args.model_type == "gru":
        tuner = GRUTuner(model_name, params, features, args.batch_size, args.max_epochs)
        pruner = HyperbandPruner(max_resource=args.max_epochs)
    elif args.model_type == "lr":
        tuner = LRTuner(model_name, params, features, enabled_regularizations=args.lr_regularization)
        pruner = DuplicateLagsPruner()
    elif args.model_type == "tsmixer":
        tuner = TSMixerTuner(model_name, params, features, args.batch_size, args.max_epochs)
        pruner = HyperbandPruner(max_resource=args.max_epochs)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    # create optuna study
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        sampler=TPESampler(),
        pruner=pruner,
        storage=OPTUNA_STORE_URI,
        load_if_exists=True,
    )

    mlflow.set_experiment(tuner.model_name)
    with mlflow.start_run(
        run_name=study_name, description=f"Tune hparams of {tuner.model_name} without changing features"
    ) as parent_run:
        # TODO wtf is this not duplicated?!?!
        mlflow.set_tag("optuna_study", study.study_name)
        study.set_user_attr("mlflow_exp_id", parent_run.info.experiment_id)
        study.set_user_attr("mlflow_parent_run_id", parent_run.info.run_id)

        # stop fake "running" trials and re-queue them (they are left when cancelling with ctrl+c)
        trials = study.trials
        for trial in trials:
            if trial.state == optuna.trial.TrialState.RUNNING:
                logger.info(f"Failing and re-queuing previously running trial {trial.number}")
                study.enqueue_trial(trial.params, user_attrs={"restart_of": trial.number})
                study.tell(trial.number, state=optuna.trial.TrialState.FAIL)

        # enqueue initial trials for fresh studies
        if not trials:
            logger.info("Fresh study; enqueuing initial trials")
            for trial_params in tuner.initial_trials():
                study.enqueue_trial(trial_params, user_attrs={"initial_trial_of": tuner.model_name})

        # run optimization
        logger.info(f"Starting optimization (n_trials={args.n_trials or 'unlimited'})")
        study.optimize(tuner, n_trials=args.n_trials)

        logger.info(f"Best trial: {study.best_trial.number} with MAE={study.best_value:.3f}")
        logger.info(f"Best hyperparameters: {study.best_params}")


if __name__ == "__main__":
    raise NotImplementedError("you need to fix some things here, at least the todos")
    logging.basicConfig(level=logging.INFO)
    torch.set_float32_matmul_precision("medium")
    seed_everything(RANDOM_SEED)
    plt.rcParams["figure.figsize"] = (16, 9)
    mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")

    main()
