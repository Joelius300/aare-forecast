# must be built from project root
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
ENV UV_PYTHON_DOWNLOADS=0

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=src/aare/pyproject.toml,target=src/aare/pyproject.toml \
    --mount=type=bind,source=src/aare-logging/pyproject.toml,target=src/aare-logging/pyproject.toml \
    --mount=type=bind,source=src/aare-timescale/pyproject.toml,target=src/aare-timescale/pyproject.toml \
    --mount=type=bind,source=src/aare-influx/pyproject.toml,target=src/aare-influx/pyproject.toml \
    --mount=type=bind,source=src/oraku-influx2pg/pyproject.toml,target=src/oraku-influx2pg/pyproject.toml \
    uv sync --locked --no-install-project --no-dev --package=aare-oraku-influx2pg

COPY . /app

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --package=aare-oraku-influx2pg

FROM python:3.12-slim-bookworm

WORKDIR /app

COPY --from=builder --chown=app:app /app /app
COPY src/oraku-influx2pg/deploy/app.json /app/app.json
COPY src/oraku-influx2pg/deploy/Procfile /app/Procfile

ENV PYTHONUNBUFFERED=1
ENV PATH="/app/.venv/bin:$PATH"
ENV PROJECT_ROOT=/app

CMD ["python", "/app/src/oraku-influx2pg/main.py"]
