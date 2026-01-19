from enum import StrEnum
from functools import cache
from typing import TypeAlias

import pytz
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


@cache
def _transform_tz(tz_raw: str):
    return pytz.timezone(tz_raw)


class OrakuSettings(BaseSettings):
    connection_string: str
    logging_level: str = "WARNING"
    default_horizon: int = 36
    maximum_horizon: int = 48
    maximum_forecast_age: str = "1d"
    timezone: str = "Europe/Zurich"
    available_cities: list[str] = ["bern"]
    default_city: str = "bern"
    loki_url: str = ""
    loki_password: str = ""
    expected_interval_sec: int = 15 * 60
    cache_tolerance_sec: int = 60
    unhealthy_age_sec: int = 35 * 60

    model_config = SettingsConfigDict(env_prefix="oraku_", env_file=".env")

    @model_validator(mode="after")
    def validate_settings(self):
        if self.default_horizon > self.maximum_horizon:
            raise ValueError("default_horizon cannot be larger than maximum_horizon")

        if self.cache_tolerance_sec >= self.expected_interval_sec:
            raise ValueError("Cache tolerance must be considerable smaller than the expected update interval")

        if self.expected_interval_sec > self.unhealthy_age_sec:
            raise ValueError("The unhealthy age threshold must be at least as large as the expected update interval.")

        return self

    @property
    def tz(self):
        return _transform_tz(self.timezone)

    # pydantic(-settings) doesn't work well with static type checkers. there's a plugin for mypy but not pyright.
    # noinspection PyArgumentList
    settings = OrakuSettings()  # pyright: ignore[reportCallIssue]


"""Singleton instance of the settings loaded in from env etc."""

CityEnum = StrEnum("CityEnum", settings.available_cities)
"""Dynamically enum from specified supported cities for automatic input validation and nicer swagger docs."""
