# must be built from project root
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder
# bytecode -> faster startup (but larger size?). copy linkmode -> can copy venv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# use python from image
ENV UV_PYTHON_DOWNLOADS=0

WORKDIR /app

# copy just uv.lock and pyproject.toml to install deps without our source
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=src/aare-shared/pyproject.toml,target=src/aare-shared/pyproject.toml \
    --mount=type=bind,source=src/aare-shared/README.md,target=src/aare-shared/README.md \
    --mount=type=bind,source=src/prediction-service/pyproject.toml,target=src/prediction-api/pyproject.toml \
    --mount=type=bind,source=src/prediction-service/README.md,target=src/prediction-api/README.md \
    --mount=type=bind,source=src/prediction-api/pyproject.toml,target=src/prediction-api/pyproject.toml \
    --mount=type=bind,source=src/prediction-api/README.md,target=src/prediction-api/README.md \
    uv sync --locked --no-install-project --no-dev --package=aare-oraku-prediction-api

# which files/dirs are copied is controlled by the .dockerignore file
COPY . /app

# install packages from our code etc. and remove unneccessary (transitive) dependencies of darts
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --package=aare-oraku-prediction-api

# must be same base image as builder, otherwise copying venv won't work
FROM python:3.12-slim-bookworm

WORKDIR /app

# copy the application and built venv from the builder
COPY --from=builder --chown=app:app /app /app

ENV PATH="/app/.venv/bin:$PATH"
COPY src/prediction-api/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
# TODO add app.json with healthcheck

# use entrypoint to respect PORT variable by dokku
CMD ["/entrypoint.sh"]
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"]
