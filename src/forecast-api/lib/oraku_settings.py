from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class OrakuSettings(BaseSettings):
    connection_string: str
    logging_level: str = "WARNING"
    default_horizon: int = 24
    maximum_horizon: int = 24
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
