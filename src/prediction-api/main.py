from typing import Optional

from fastapi import FastAPI
from datetime import datetime, UTC

app = FastAPI()

# use AsyncConnectionPool like shown here: https://blog.danielclayton.co.uk/posts/database-connections-with-fastapi/
# configure connection string with pydantic-settings: https://fastapi.tiangolo.com/advanced/settings/


@app.get("/predictions")
def get_predictions(at: Optional[datetime] = None, horizon: Optional[int] = None):
    now = datetime.now(UTC)

    return {
        "time": [now, now, now],
        "temp_bern": [1, 2, 3],
    }
