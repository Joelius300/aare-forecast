# /// script
# requires-python = ">=3.10,<3.12"
# dependencies = [
#     "aare-shared",
#     "darts==0.35.0",
#     "dvc",
#     "influxdb-client",
#     "pandas",
#     "timesfm[torch]",
# ]
#
# [tool.uv.sources]
# aare-shared = { path = "../src/aare-shared" }
# ///
import logging

from aare.evaluation.evaluation import evaluation_pipeline_uni
from aare.params import read_params
from aare.wrappers.timesfm import TimesFmDarts


def main():
    params = read_params()
    horizon = params["general"]["forecast_horizon"]
    model_version = params["timesfm"]["version"]
    models = {
        "TIMESFM": TimesFmDarts(horizon, TimesFmDarts.Version(model_version)),
    }

    evaluation_pipeline_uni(models, horizon, params["validation"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
