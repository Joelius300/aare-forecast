import argparse
import logging
from dataclasses import dataclass
from datetime import datetime


from aare_train.evaluation.evaluation import evaluate_model, store_results
from aare_train.features.registry import FEATURES
from aare_train.fetching.feature_set import FeatureSet
from aare_train.params import read_params
from aare_train.storage.model import load_model

logger = logging.getLogger(__name__)

MODEL_NAME_SEP = "-"
NOW_SUFFIX = "__NOW__"

params = read_params(ensure_dvc=True)


@dataclass
class EvalArgs:
    model_name: str
    model_version: str
    stride: int
    horizon: int
    store: bool
    override: bool
    suffix: str
    use_test: bool
    # todo start and end times, can/should be used to see how much worse the model is when using weather forecasts
    #  instead of true measurement data.
    min_lookback_hours: int
    season: tuple[int, int]
    tz: str


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
      uv run scripts/eval.py nowcasting_temp-dev

      uv run scripts/eval.py nowcasting_temp-1.3 --stride 1 --test
""",
    )

    parser.add_argument(
        "model",
        help="Model including name and version, e.g. nowcasting_temp-dev",
    )
    parser.add_argument(
        "--stride",
        help="Evaluation stride, will take dvc param if not specified",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--horizon",
        help="Forecast horizon, will take dvc param if not specified",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--no-store",
        dest="store",
        default=True,
        action="store_false",
        help="Don't store results to disk",
    )
    parser.add_argument(
        "--override",
        dest="override",
        default=False,
        action="store_true",
        help="Override files even if version is not dev",
    )
    parser.add_argument(
        "--suffix",
        nargs="?",
        const=NOW_SUFFIX,  # value when flag is present but no argument
        default=None,  # when flag is not present at all
        help="Optional suffix for stored results. If flag is provided without value, the current time is used.",
    )
    parser.add_argument(
        "--test",
        dest="use_test",
        default=False,
        action="store_true",
        help="Use test data as specified in dvc params. Implies --suffix test.",
    )

    args = parser.parse_args()
    name, version = args.model.split(MODEL_NAME_SEP, maxsplit=2)

    if args.stride is None:
        args.stride = params["validation"]["stride"]

    if args.horizon is None:
        args.horizon = params["general"]["forecast_horizon"]

    min_lookback_hours = params["validation"]["min_lookback_hours"]
    season = params["general"]["season_start"], params["general"]["season_end"]
    tz = params["general"]["timezone"]
    # todo: add from/to resp. period as well as --val (default) and --test.

    if args.use_test:
        if args.suffix is None or args.suffix == NOW_SUFFIX:
            args.suffix = "test"
        else:
            raise ValueError("You may not provide a suffix when doing test eval.")

    suffix = args.suffix if args.suffix != NOW_SUFFIX else datetime.now().strftime("%Y%m%dT%H%M%S")

    return EvalArgs(
        name,
        version,
        args.stride,
        args.horizon,
        args.store,
        args.override,
        suffix,
        args.use_test,
        min_lookback_hours,
        season,
        tz,
    )


def main():
    args = parse_args()
    meta, model, scalers = load_model(name=args.model_name, version=args.model_version)
    fs = FeatureSet(
        FEATURES.get_many(meta["features"]["targets"]),
        past=FEATURES.get_many(meta["features"].get("past")),
        future=FEATURES.get_many(meta["features"].get("future")),
        split_params=params["split"],  # todo remove, see below
    )

    # todo: always use from/to, potentially populated from the split_params, but always use get()
    if args.use_test:
        targets, _, fc = fs.get_test()
    else:
        targets, _, fc = fs.get_val()

    metrics, samples, raw_metrics = evaluate_model(
        model,
        targets,
        args.horizon,
        args.stride,
        args.min_lookback_hours,
        future_cov=fc,
        tz=args.tz,
        month_filter=args.season,
        data_transformers=scalers,
        get_raw=True,
    )

    test_data_disclaimer = " (ON TEST DATA)" if args.use_test else ""
    print(f"Evaluation of {args.model_name}{MODEL_NAME_SEP}{args.model_version}{test_data_disclaimer}:")
    print(metrics)  # could also use fancy tools to make a table etc. but eh

    if not args.store:
        return

    name = args.model_name + MODEL_NAME_SEP + args.model_version
    if args.suffix:
        name += f"-{args.suffix}"

    store_results(name, metrics, raw_metrics, samples, override=args.override or args.model_version == "dev")


if __name__ == "__main__":
    main()
