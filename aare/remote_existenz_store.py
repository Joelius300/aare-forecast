import logging
from typing import cast

import pandas as pd
from influxdb_client import InfluxDBClient  # pyright: ignore [reportPrivateImportUsage]

logger = logging.getLogger(__name__)


def _chain_equality(
    column: str, *values: str | int | list, separator="or", wrap_in_quotes=True
):
    """Returns a predicate function where a column is tested against one or more values with equality (==)."""
    # using contains(value: r["loc"], set: ["2135", "2030"]) has muuuuch worse performance
    if values is None or len(values) == 0:
        raise ValueError("cannot equality-chain 0 values")

    if len(values) == 1 and (isinstance(values[0], list)):
        # unpack list so you don't have to on the caller's side
        values: list = cast(list, values[0])

    q = '"' if wrap_in_quotes else ""
    return "(r) => " + (
        f" {separator} ".join([f'r["{column}"] == {q}{value}{q}' for value in values])
    )


class RemoteExistenzStore:
    def __init__(self, timeout=60_000, debug=False):
        self.client = InfluxDBClient(
            url="https://influx.konzept.space/",
            # this is a public readonly token, so while not best practice, there's no danger in hard-coding it here :)
            token="0yLbh-D7RMe1sX1iIudFel8CcqCI8sVfuRTaliUp56MgE6kub8-nSd05_EJ4zTTKt0lUzw8zcO73zL9QhC3jtA==",
            org="api.existenz.ch",
            debug=debug,
            timeout=timeout,
        )

    def query_hydro(
        self,
        period: str | tuple[str, str],
        locations: str | int | list[str | int],
        fields: str | list[str] = "temperature",
        agg_freq="1h",  # could also read from params
        agg_func="mean",
        agg_create_empty=False,
        keep_loc=False,
    ):
        """Queries hydrology data from the remote influx store by existenz.ch"""
        # can later be split and extended for non-hydro data
        start = period if isinstance(period, str) else period[0]
        stop = "now()" if isinstance(period, str) else period[1]

        # NOTE: aggregateWindow 1h on 12:00 will take the values from
        #   11:00 until 12:00 and combine them into a single value
        #   with timestamp ** 12:00 **. This means that filtering from
        #   00:00 to xxxx will result in the first entry being
        #   01:00 (aggregate of the values between 00:00 and 01:00).
        #   This might be important because the first entry the model sees
        #   will be 01:00, but it contains values since 00:00 so it's correct.
        #   Can get confusing, esp. if you filter up to but excluding 2024, you
        #   will still get a timestamp on 2024-01-01T00:00:00 containing the last
        #   hour of 2023. IT DOES NOT CONTAIN VALUES FROM 2024!
        # PS. Maybe training data should also be aligned to first of January for consistency.
        #   But it should be enough to align to the stride used in training and validation (prob daily, so 00:00).
        query = (
            f'from(bucket: "existenzApi")\n'
            f'  |> range(start: {start}, stop: {stop})\n'
            f'  |> filter(fn: {_chain_equality("_measurement", "hydro")})\n'
            f'  |> filter(fn: {_chain_equality("_field", fields)})\n'
            f'  |> filter(fn: {_chain_equality("loc", locations)})\n'
            f'  |> aggregateWindow(every: {agg_freq}, fn: {agg_func}, createEmpty: {str(agg_create_empty).lower()})\n'
            f'  |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")\n'
            f'  |> drop(columns: ["_start", "result", "_stop", "table", "_measurement"])'
        )

        logger.debug("Executing Flux Query:\n{%s}", query)

        df = cast(pd.DataFrame, self.client.query_api().query_data_frame(query))

        unnecessary_cols = ["result", "table"]
        if not keep_loc and (
            isinstance(locations, int)
            or isinstance(locations, str)
            or len(locations) == 1
        ):
            unnecessary_cols.append("loc")

        return df.drop(unnecessary_cols, axis=1)
