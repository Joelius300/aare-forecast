# /// script
# requires-python = ">=3.10,<3.12"
# dependencies = [
#     "aare-train[gpu-auto]",
#     "dvc",
#     "influxdb-client",
#     "pandas<3.0.0",
#     "numpy>=2.0.0",
#     "timesfm[torch]<2.0.0",
#     "pyarrow",
# ]
#
# [tool.uv.sources]
# aare-train = { path = "../src/aare-train", editable = true }
# ///
import logging

from aare_train.evaluation.pipeline import evaluation_pipeline_uni
from aare_train.params import read_params
from aare_train.wrappers.timesfm import TimesFmDarts


# todo replace with darts integration of timesfm
def main():
    params = read_params()
    horizon = params["general"]["forecast_horizon"]
    model_version = params["timesfm"]["version"]
    models = {
        "TIMESFM": TimesFmDarts(horizon, TimesFmDarts.Version(model_version)),
    }

    evaluation_pipeline_uni(models, params)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
