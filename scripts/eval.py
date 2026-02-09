import argparse
import logging
from dataclasses import dataclass

from aare_train.storage.model import load_model

logger = logging.getLogger(__name__)

MODEL_NAME_SEP = "-"


@dataclass
class EvalArgs:
    model_name: str
    model_version: str
    stride: int
    store: bool
    # todo start and end times, can/should be used to see how much worse the model is when using weather forecasts
    #  instead of true measurement data.


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
      uv run scripts/eval.py lr dev

      uv run scripts/eval.py gru-1.3 --stride 1
            """,
    )

    parser.add_argument(
        "model_name",
        required=True,
        help="Name of model",
    )
    parser.add_argument(
        "model_version",
        help="Model version",
        default=None,
    )
    parser.add_argument(
        "--stride",
        "-s",
        help="Evaluation stride, will take dvc param if not specified",
        default=None,
    )
    parser.add_argument(
        "--no-store",
        dest="store",
        default=True,
        action="store_false",
        help="Don't store results to disk",
    )

    args = parser.parse_args()
    args = EvalArgs(**args.__dict__)

    if not args.model_version:
        if MODEL_NAME_SEP not in args.model_name:
            raise ValueError(
                f"Specified model as a combined value ('{args.model_name}'), "
                + "but could be split into name and version"
            )

        name, version = args.model_name.split(MODEL_NAME_SEP, maxsplit=2)
        args.model_name = name
        args.model_version = version

    return args


def main():
    args = parse_args()
    meta, model, scalers = load_model(name=args.model_name, version=args.model_version)
    # todo :)


if __name__ == "__main__":
    main()
