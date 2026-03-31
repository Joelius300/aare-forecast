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
    --mount=type=bind,source=src/oraku-forecast/pyproject.toml,target=src/oraku-forecast/pyproject.toml \
    --mount=type=bind,source=src/oraku-api/pyproject.toml,target=src/oraku-api/pyproject.toml \
    --mount=type=bind,source=src/oraku-influx2pg/pyproject.toml,target=src/oraku-influx2pg/pyproject.toml \
    uv sync --locked --no-install-project --no-dev --extra cpu --package=aare-oraku-forecast \
    && uv pip uninstall shap numba llvmlite networkx slicer pygit2 pygments cryptography

# which files/dirs are copied is controlled by the .dockerignore file.
# wouldn't need sources for oraku-api, but they carry almost no weight so whatever.
COPY . /app

# install packages from our code etc. and remove unneccessary (transitive) dependencies of darts
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --extra cpu --package=aare-oraku-forecast \
    && uv pip uninstall shap numba llvmlite networkx slicer pygit2 pygments cryptography

# must be same base image as builder, otherwise copying venv won't work
FROM python:3.12-slim-bookworm

WORKDIR /app

# copy the application and built venv from the builder
COPY --from=builder --chown=app:app /app /app
COPY src/oraku-forecast/deploy/app.json /app/app.json
COPY src/oraku-forecast/deploy/Procfile /app/Procfile

ENV PYTHONUNBUFFERED=1
ENV PATH="/app/.venv/bin:$PATH"
ENV PROJECT_ROOT=/app

CMD ["python", "/app/src/oraku-forecast/main.py"]
