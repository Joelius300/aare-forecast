from typing import cast

from psycopg_pool import AsyncConnectionPool
from fastapi import Request

from oraku_api.latest_cache import LatestCache
from oraku_api.oraku_settings import OrakuSettings


# unfortunately, fastapi DI cannot handle if the function is already an async context manager,
# even though it's converted to that internally before use...
async def open_db(request: Request):
    """
    Generator function that opens, yields and closes a connection of the provided pool.
    For use with FastAPI DI (Depends).
    """
    pool = request.state.db_pool  # pyright: ignore[reportAny]
    assert isinstance(pool, AsyncConnectionPool), "Could not retrieve db pool with correct type from app state."
    pool = cast(AsyncConnectionPool, pool)  # needed for default generics to kick in below
    async with pool.connection() as conn:
        yield conn


def get_settings(request: Request):
    settings = request.state.settings  # pyright: ignore[reportAny]
    assert isinstance(settings, OrakuSettings), "Could not retrieve settings with correct type from app state."

    return settings


def get_caches(request: Request):
    caches = request.state.latest_caches  # pyright: ignore[reportAny]
    assert isinstance(caches, dict), "Could not retrieve caches with correct type from app state."

    return cast(dict[str, LatestCache], caches)
