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
    return


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

    print(ts)
    print(type(ts))

    tfm = TimesFmDarts(params["general"]["forecast_horizon"])
    pred = tfm.predict(series=ts)

    print(pred)
    print(type(pred))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
