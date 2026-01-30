import asyncio
from datetime import timedelta
import logging
from collections.abc import Sequence
from typing import final, cast, override

from aare.constants import TIME
from aare.locations import LOC_HYDRO_ALIAS
from aare_train.utils import join_many, trav
import httpx
import pandas as pd

from oraku_forecast.external_sources.external_source import ExternalSource

logger = logging.getLogger(__name__)


@final
class BafuFlowSource(ExternalSource):
    """
    Fetch median, min, max, q25 and q75 flow forecast from BAFU via their plots API (probably not intended lol).
    Median forecasts are required (will raise otherwise), all the other columns may be missing in the returned dataframe
    if they couldn't be fetched from the plot response and warnings are emitted.
    """

    # the data is computed via many (~21) black- and white-box models and aggregated into min/max, q25/q75 and median.
    # in data[0] and data[1] is max and min forecast.
    # in data[2] are 25% and 75% quantiles ASC, then DESC, so twice as many data points.
    # in data[3] is the median, so should have the same n as data[0] and data[1]. most important for us.
    # in data[4] are the true measurement values up to the point the forecast was made.
    # it seems the prediction was made at the first data point or shortly before that, found no exact time.
    # maybe send an email to BAFU to see where best to consume this data, given that we're pulling a plot and parsing
    # the data out of that, but I don't think they have an API for consumers for stuff like this.

    def __init__(self, url_template: str, locations: Sequence[int], last_updated_delta: timedelta):
        self._url_template = url_template
        self._locations = locations
        self._last_updated_delta = last_updated_delta

    @override
    async def fetch(self) -> pd.DataFrame:
        async with httpx.AsyncClient() as client:
            tasks = [self._fetch_loc(client, loc, self._last_updated_delta) for loc in self._locations]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        dfs: list[pd.DataFrame] = []
        for loc, result in zip(self._locations, results):
            if isinstance(result, BaseException):
                logger.error(f"Could not fetch hydro flow forecast for loc '{loc}': {result}", exc_info=result)
                continue

            dfs.append(result)

        if not dfs:
            raise ValueError("Could not fetch any locations for hydro flow forecast; source is unusable")

        return pd.concat(dfs)

    async def _fetch_loc(self, client: httpx.AsyncClient, loc: int, last_updated_delta: timedelta) -> pd.DataFrame:
        r = await client.get(self._url_template.format(loc=loc))
        r.raise_for_status()
        json = r.json()
        data = cast(list[dict[str, str | float]], trav(json, "plot", "data"))
        base_col_name = "flow"

        parts: list[pd.DataFrame] = []
        try:
            median = self._trace_to_df(data[3], base_col_name, assert_name="median")
            parts.append(median)
        except Exception as e:
            raise ValueError("Could not extract median flow forecast from response; unrecoverable.") from e

        try:
            min = self._trace_to_df(data[1], f"{base_col_name}_min", assert_name="min")
            max = self._trace_to_df(data[0], f"{base_col_name}_max", assert_name="max")
            parts.extend([min, max])
        except Exception as e:
            logger.warning("Could not extract min and max flow forecasts from response; continuing still.", exc_info=e)

        try:
            quantiles = self._quantile_trace_to_df(data[2], base_col_name)
            parts.append(quantiles)
        except Exception as e:
            logger.warning(
                "Could not extract 25 and 75 percentile flow forecasts from response; continuing still.", exc_info=e
            )

        df = join_many(*parts, on="time")
        df["location"] = loc

        # since we want to display these flow forecasts in the app too and not just use them for our models,
        # and because the bafu flow forecast is only made once per day (unless there's a flood, then it's more),
        # we also estimate when the forecast was last updated by simply taking the first timestamp - a bit.
        # last_updated is therefore like run_ts, but not referencing the aare-oraku run, but the upstream forecast run.
        # for our own forecasts it's run_ts, for meteotest we don't care, but here we want it.
        # the same flow forecast may be stored 24*4 times in the worst case, but I don't think we need to care lol.
        df["last_updated"] = df["time"].min() - last_updated_delta

        return df

    def _trace_to_df(self, trace: dict[str, str | float], col_name: str, assert_name: str | None = None):
        if assert_name and assert_name not in (trace_name := cast(str, trav(trace, "name")).lower()):
            logger.warning(f"Trace name expected to contain '{assert_name}', but is '{trace_name}'")

        x, y = self._get_x_y(trace)

        return pd.DataFrame({"time": pd.to_datetime(x), col_name: y})

    def _quantile_trace_to_df(
        self, trace: dict[str, str | float], base_col_name: str, q25_suffix: str = "_q25", q75_suffix: str = "_q75"
    ):
        trace_name = cast(str, trav(trace, "name")).lower()
        if "25" not in trace_name or "75" not in trace_name:
            logger.warning(f"Trace name for quantile trace expected to contain '25' and '75', but is '{trace_name}'")

        x, y = self._get_x_y(trace)
        half, rest = divmod(len(x), 2)

        # for some weird reason, it seems that the last datapoint is (always?) duplicated, so we subtract rest (0 or 1).
        # note that the second half is the reverse of the first half, in line with plotly's fill='tozerox'.
        x1, x2 = x[:half], x[half : len(x) - rest][::-1]
        y1, y2 = y[:half], y[half : len(y) - rest][::-1]

        times = pd.to_datetime(x1)
        if not (times == pd.to_datetime(x2)).all():
            raise ValueError("Times in first and second half aren't aligned")

        return pd.DataFrame({"time": times, base_col_name + q25_suffix: y1, base_col_name + q75_suffix: y2})

    @staticmethod
    def _get_x_y(trace: dict[str, str | float]) -> tuple[list[str], list[float]]:
        x = trav(trace, "x")
        y = trav(trace, "y")
        assert isinstance(x, list), "x is not a list"
        assert isinstance(y, list), "y is not a list"
        x = cast(list[str], x)
        y = cast(list[float], y)

        return x, y

    @override
    def prepare(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        # could assert that there is only one run_ts and that all times are unique
        df = raw_data.drop(["run_ts", "last_updated"], axis=1)

        # setting index to (time, location) then unstacking [location] is the same as
        # pivoting with index="time", columns="location" and values = {all other columns}
        df = cast(pd.DataFrame, df.set_index(["time", "location"]).unstack())
        # after unstacking (or pivoting, doesn't matter) the columns will be a MultiIndex with
        # the first level the original name of the col (e.g. 'flow_min') and the second the location (e.g. 2135)
        # so they have to be combined into a single combined name. But need to make sure that the feature base name does
        # not contain underscore, since those are used to divide between base feature, location and transformations.
        # ('flow_min', 2135) will become 'flow-min_bern'
        df.columns = df.columns.map(lambda x: f"{x[0].replace('_', '-')}_{LOC_HYDRO_ALIAS[x[1]].lower()}")
        # currently, the index is the time col and named 'time', but it should be an extra col '_time'
        df = df.reset_index(names=TIME)

        return df
