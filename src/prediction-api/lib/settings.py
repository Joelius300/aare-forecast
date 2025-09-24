from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    connection_string: str
    logging_level: str = "DEBUG"
    default_horizon: int = 24
    maximum_horizon: int = 24
    maximum_prediction_age: str = "1h"
    time_zone: str = "Europe/Zurich"

    # TODO read from .env during development (similar to dev_config.yaml)
    model_config = SettingsConfigDict(env_prefix="oraku_")
