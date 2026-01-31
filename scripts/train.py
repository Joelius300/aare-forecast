import argparse
from collections.abc import Callable
import logging
from typing import Any

from aare.constants import RANDOM_SEED
from aare_train.fetching.feature_identifiers import FeatureIdentifiers
from aare_train.models import GRUTrainer, LRTrainer, TSMixerTrainer, BaseTrainer
from aare_train.params import Params, read_params
import matplotlib.pyplot as plt
import mlflow
import torch
from lightning_fabric import seed_everything

logger = logging.getLogger(__name__)

MODEL_TRAINERS: dict[str, Callable[[Params, FeatureIdentifiers], BaseTrainer]] = {
    "GRU": GRUTrainer,
    "LR": LRTrainer,
    "TSMIXER": TSMixerTrainer,
}
MODEL_TYPES = tuple(k.lower() for k in MODEL_TRAINERS.keys())


def parse_features(targets: list[str], future: list[str] | None) -> FeatureIdentifiers:
    """Parse feature lists into FeatureIdentifiers."""
    return {
        "targets": targets,
        "future": future or [],
    }


def parse_hyperparameters(args: list[str]) -> dict[str, Any]:
    """Parse hyperparameters from command line arguments in key=value format."""
    hparams: dict[str, Any] = {}
    for arg in args:
        if "=" not in arg:
            raise ValueError(f"Invalid hyperparameter format: {arg}. Expected key=value")

        key, value = arg.split("=", 1)

        # try to parse as int, float, bool, or keep as string
        try:
            if value.lower() in ["true", "false"]:
                hparams[key] = value.lower() == "true"
            elif "." in value:
                hparams[key] = float(value)
            else:
                hparams[key] = int(value)
        except ValueError:
            hparams[key] = value

    return hparams


def main():
    parser = argparse.ArgumentParser(
        description="Train a single model with specified hyperparameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train GRU with custom hyperparameters
  python scripts/train.py gru --targets temp_bern --future tt_bern ss_bern \\
    --hparams input_chunk_length=24 hidden_dim=64 lr=0.001

  # Train LR with ridge regularization
  python scripts/train.py lr --targets temp_bern --future tt_bern ss_bern \\
    --hparams regularization=ridge alpha=0.1 lag_max=24

  # Train TSMixer without evaluation
  python scripts/train.py tsmixer --targets temp_bern --future tt_bern \\
    --hparams hidden_size=128 --no-evaluate
        """,
    )

    # TODO consolidate common args with tune.py into some importable helper function and type them
    parser.add_argument(
        "model_type",
        choices=MODEL_TYPES,
        help="Type of model to train",
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
        "--hparams",
        nargs="*",
        default=[],
        help="Hyperparameters in key=value format (e.g., lr=0.001 hidden_dim=64)",
    )
    parser.add_argument(
        "--evaluate",
        dest="evaluate",
        action="store_true",
        default=True,
        help="Evaluate the model after training (default: True)",
    )
    parser.add_argument(
        "--no-evaluate",
        dest="evaluate",
        action="store_false",
        help="Skip evaluation after training",
    )
    parser.add_argument(
        "--run-name",
        help="Custom MLflow run name (default: train-{model_type})",
    )
    parser.add_argument(
        "--experiment-name",
        help="Custom MLflow experiment name (default: model_type)",
    )

    args = parser.parse_args()

    # parse features and hyperparameters
    features = parse_features(args.targets, args.future)
    hparams = parse_hyperparameters(args.hparams)
    params = read_params()

    # set default names
    run_name = args.run_name or f"train-{args.model_type}"
    experiment_name = args.experiment_name or args.model_type.upper()

    # create trainer based on model type
    logger.info(f"Training {args.model_type.upper()} with hyperparameters: {hparams}")

    trainer_cls = MODEL_TRAINERS.get(args.model_type.upper())
    if trainer_cls is None:
        raise ValueError(f"Unknown model type: {args.model_type}")

    trainer = trainer_cls(params, features)

    # train and optionally evaluate
    mlflow.set_experiment(experiment_name)
    metrics = None
    with mlflow.start_run(run_name=run_name, log_system_metrics=True) as run:
        mlflow.log_dict(read_params(ensure_dvc=True).__dict__, "params.yaml")
        mlflow.log_params(trainer.hparams_general)

        model = trainer.build_model(**hparams)
        trainer.fit(model)

        if args.evaluate:
            metrics = trainer.evaluate(model, run)

    if metrics:
        logger.info(f"Evaluation metrics: MAE={metrics.mae:.3f}, RMSE={metrics.rmse:.3f}")
    else:
        logger.info("Training completed without evaluation")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    torch.set_float32_matmul_precision("medium")
    seed_everything(RANDOM_SEED)
    plt.rcParams["figure.figsize"] = (16, 9)
    mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")

    main()
