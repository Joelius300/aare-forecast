import argparse
from collections.abc import Sequence
from dataclasses import dataclass
import logging

from aare.constants import RANDOM_SEED
from aare_train.cli.parsing import parse_features, parse_hyperparameters
from aare_train.models.registry import MODEL_TYPES, MODEL_TRAINERS
from aare_train.params import read_params
import matplotlib.pyplot as plt
import mlflow
import torch
from lightning_fabric import seed_everything

from aare_train.storage.model import model_exists, save_model

logger = logging.getLogger(__name__)


@dataclass
class TrainArgs:
    model_type: str
    targets: Sequence[str]
    future: Sequence[str]
    hparams: Sequence[str]
    evaluate: bool
    experiment_name: str | None
    version: str


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a single model with specified hyperparameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
      # Train GRU with custom hyperparameters
      uv run scripts/train.py gru --targets temp_bern --future tt_bern ss_bern \\
        --hparams input_chunk_length=24 hidden_dim=64 lr=0.001

      # Train LR with ridge regularization
      uv run scripts/train.py lr --targets temp_bern --future tt_bern ss_bern \\
        --hparams regularization=ridge alpha=0.1 lag_max=24

      # Train TSMixer without evaluation
      uv run scripts/train.py tsmixer --targets temp_bern --future tt_bern \\
        --hparams hidden_size=128 --no-evaluate
            """,
    )

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
        "--experiment-name",
        help="Custom MLflow experiment name (default: model_type)",
    )
    parser.add_argument(
        "--version",
        help="Model version to store (default: dev)",
        default="dev",
    )

    args = parser.parse_args()

    return TrainArgs(**args.__dict__)


def main():
    args = parse_args()
    # parse features and hyperparameters
    features = parse_features(args.targets, args.future)
    hparams = parse_hyperparameters(args.hparams)
    params = read_params()

    # set default names
    model_name: str = args.model_type.upper()
    experiment_name: str = args.experiment_name or model_name
    model_version: str = args.version

    if model_version != "dev" and model_exists(model_name, model_version):
        raise ValueError(f"Model {model_name}-{model_version} already exists.")

    # create trainer based on model type
    logger.info(f"Training {args.model_type.upper()} with hyperparameters: {hparams}")

    trainer_cls = MODEL_TRAINERS.get(args.model_type.upper())
    if trainer_cls is None:
        raise ValueError(f"Unknown model type: {args.model_type}")

    # train and optionally evaluate
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(log_system_metrics=True) as run:
        mlflow.log_dict(read_params(ensure_dvc=True), "params.yaml")  # pyright: ignore[reportArgumentType]

        trainer = trainer_cls(params, features)
        mlflow.log_params(trainer.hparams_general)

        model = trainer.build_model(**hparams)
        scalers, metrics = trainer.fit(model, args.evaluate)

        if metrics:
            logger.info(f"Evaluation metrics: {metrics}")
        else:
            logger.info("Training completed without evaluation")

        save_model(model_name, model_version, model, features, scalers, run.info, hparams)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    torch.set_float32_matmul_precision("medium")
    seed_everything(RANDOM_SEED)
    plt.rcParams["figure.figsize"] = (16, 9)
    mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")

    main()
