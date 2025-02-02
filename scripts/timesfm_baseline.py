# /// script
# requires-python = ">=3.10,<3.12"
# dependencies = [
#     "darts",
#     "dvc",
#     "influxdb-client",
#     "pandas",
#     "timesfm[torch]",
# ]
# ///
import logging

from aare.evaluation.evaluation import evaluation_pipeline
from aare.params import read_params
from aare.wrappers.timesfm import TimesFmDarts


def main():
    params = read_params()
    horizon = params["general"]["forecast_horizon"]
    model_version = params["timesfm"]["version"]
    models = {
        "TIMESFM": TimesFmDarts(horizon, TimesFmDarts.Version(model_version)),
    }

    evaluation_pipeline(models, horizon, params["validation"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
