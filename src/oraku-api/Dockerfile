# must be built from project root
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder
# bytecode -> faster startup (but larger size?). copy linkmode -> can copy venv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# use python from image
ENV UV_PYTHON_DOWNLOADS=0

WORKDIR /app

# copy just uv.lock and pyproject.toml to install deps without our source.
# annoyingly, we need to copy all packages, otherwise it thinks the lock file is outdated.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=src/aare/pyproject.toml,target=src/aare/pyproject.toml \
    --mount=type=bind,source=src/aare-train/pyproject.toml,target=src/aare-train/pyproject.toml \
    --mount=type=bind,source=src/aare-logging/pyproject.toml,target=src/aare-logging/pyproject.toml \
    --mount=type=bind,source=src/aare-timescale/pyproject.toml,target=src/aare-timescale/pyproject.toml \
    --mount=type=bind,source=src/aare-influx/pyproject.toml,target=src/aare-influx/pyproject.toml \
    --mount=type=bind,source=src/aare-plotly/pyproject.toml,target=src/aare-plotly/pyproject.toml \
    --mount=type=bind,source=src/oraku-forecast/pyproject.toml,target=src/oraku-forecast/pyproject.toml \
    --mount=type=bind,source=src/oraku-api/pyproject.toml,target=src/oraku-api/pyproject.toml \
    --mount=type=bind,source=src/oraku-influx2pg/pyproject.toml,target=src/oraku-influx2pg/pyproject.toml \
    uv sync --locked --no-install-project --no-dev --no-editable --package=aare-oraku-api

# which files/dirs are copied is controlled by the .dockerignore file.
# wouldn't need sources for oraku-forecast and aare-train, but they carry almost no weight so whatever (just worse caching).
COPY . /app

# install packages from our code etc.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable --package=aare-oraku-api

# must be same base image as builder, otherwise copying venv won't work
FROM python:3.12-slim-bookworm

WORKDIR /app

# copy the application and built venv from the builder
COPY --from=builder --chown=app:app /app /app

ENV PYTHONUNBUFFERED=1
ENV PATH="/app/.venv/bin:$PATH"
RUN chmod +x /app/src/oraku-api/entrypoint.sh
COPY src/oraku-api/deploy/app.json /app/app.json

# MUST NOT EXPOSE, otherwise dokku will want to deploy on that port!
# EXPOSE 5000

# use entrypoint to respect PORT variable by dokku
CMD ["/app/src/oraku-api/entrypoint.sh"]
# could probably also do this with the procfile, there it would respect $PORT
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "$PORT"] <- doesn't respect $PORT
