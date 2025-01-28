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

import timesfm
from darts import TimeSeries
from darts.utils.missing_values import extract_subseries

from aare.AareDataset import AareDataset
from aare.params import GeneralParams
from aare.params import read_params
from aare.preparation import (
    resample,
    remove_faulty_periods,
    remove_outliers,
    interpolate,
)
from aare.utils import to_ts
from aare.wrappers.timesfm import TimesFmDarts


def load_model(params: GeneralParams):
    return timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(
            backend="gpu",
            per_core_batch_size=32,
            horizon_len=params["forecast_horizon"],
            input_patch_len=32,  # cannot be changed
            context_len=6 * 32,
            # even though we don't want quantiles, the model weights
            # contain quantiles heads and must be loaded if we want
            # to use the pre-trained one, apparently.
        ),
        checkpoint=timesfm.TimesFmCheckpoint(huggingface_repo_id="google/timesfm-1.0-200m-pytorch"),
    )


def prepare_data(dataset: AareDataset) -> TimeSeries:
    val = dataset.get_val()

    val = resample(val)
    val = remove_faulty_periods(val)
    val = remove_outliers(val)
    val = interpolate(val, drop_filled=True)

    ts = to_ts(val)

    return ts


def main():
    dataset = AareDataset.from_conf()
    params = read_params()

    val = prepare_data(dataset)
    ts = extract_subseries(val)[-1]

    tfm = load_model(params["general"])
    d_tfm = TimesFmDarts(tfm)
    pred = d_tfm.predict(tfm.horizon_len, ts)

    print(pred)
    print(type(pred))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
