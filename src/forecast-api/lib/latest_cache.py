from datetime import datetime, timedelta, UTC

import pandas as pd


class LatestCache:
    def __init__(self, expected_interval: timedelta, tolerance: timedelta):
        self._expected_interval: timedelta = expected_interval
        self._tolerance: timedelta = tolerance

        self._data: pd.DataFrame | None = None
        self._last_updated: datetime | None = None

    @property
    def expected_interval(self):
        return self._expected_interval

    @property
    def tolerance(self) -> timedelta:
        """
        How much time to subtract from the age to make sure it's not declared 'fresh' when new data is available.
        Wrongfully stale is better than wrongfully fresh, esp. with heuristic invalidation!
        """
        return self.tolerance

    def update(self, last_updated: datetime, df: pd.DataFrame):
        """Update the cache with new data. DATA IS ONLY CACHED IF THE LAST_UPDATED KEY IS NEWER!"""
        if self._last_updated is not None and self._last_updated > last_updated:
            raise ValueError("Cache update attempted with data older than already stored in cache!")

        self._last_updated = last_updated
        self._data = df

    @property
    def age(self) -> timedelta:
        """Exact age of the cached data."""
        if self._last_updated is None:
            return timedelta.max

        now = datetime.now(UTC)
        return now - self._last_updated

    @property
    def age_plus_tolerance(self) -> timedelta:
        """Age plus the configured tolerance."""
        return self.age - self._tolerance

    @property
    def fresh(self) -> bool:
        return self.age_plus_tolerance < self._expected_interval

    @property
    def last_updated(self):
        return self._last_updated

    @property
    def data(self):
        return self._data
