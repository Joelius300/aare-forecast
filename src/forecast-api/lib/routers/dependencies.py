from typing import cast

from psycopg_pool import AsyncConnectionPool
from fastapi import Request

# unfortunately, fastapi DI cannot handle if the function is already an async context manager,
# even though it's converted to that internally before use...
async def open_db(request: Request):
    """
    Generator function that opens, yields and closes a connection of the provided pool.
    For use with FastAPI DI (Depends).
    """
    pool = request.app.state.db_pool  # pyright: ignore[reportAny]
    assert isinstance(pool, AsyncConnectionPool), "Could not retrieve pool with correct type from app state."
    pool = cast(AsyncConnectionPool, pool)  # needed for default generics to kick in below
    async with pool.connection() as conn:
        yield conn
