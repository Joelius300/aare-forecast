from pathlib import Path
from fastapi import APIRouter
from fastapi.responses import FileResponse
from oraku_api.dto import Config
from oraku_api.oraku_settings import settings

router = APIRouter()

# this is fancy, but completely overkill for our use case
# static_dir = importlib.resources.files("oraku_api") / "static"
# with importlib.resources.as_file(static_dir / "index.html") as index_path:
#     return FileResponse(index_path)
static_dir = Path(__file__).parent.parent / "static"


# yes, using async for non-async methods is better in FastAPI (except if there is blocking IO in the function)
@router.get("/config")
async def get_config() -> Config:
    """Get the config the API is running with. Things like maximum_forecast_age, timezone, etc."""
    return Config(
        timezone=settings.timezone,
        maximum_forecast_age=settings.maximum_forecast_age,
        default_horizon=settings.default_horizon,
        maximum_horizon=settings.maximum_horizon,
        available_cities=settings.available_cities,
    )


@router.get("/", include_in_schema=False)
async def get_index():
    return FileResponse(static_dir / "index.html")
