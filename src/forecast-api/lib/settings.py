from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    connection_string: str
    logging_level: str = "WARNING"
    default_horizon: int = 24
    maximum_horizon: int = 24
    maximum_forecast_age: str = "1h"
    timezone: str = "Europe/Zurich"

    model_config = SettingsConfigDict(env_prefix="oraku_", env_file=".env")
