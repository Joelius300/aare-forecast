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
    --mount=type=bind,source=src/prediction-service/pyproject.toml,target=src/prediction-service/pyproject.toml \
    --mount=type=bind,source=src/prediction-service/README.md,target=src/prediction-service/README.md \
    uv sync --locked --no-install-project --no-dev --extra cpu --package=aare-oraku-prediction

# could probably do more granularly
COPY . /app
# TODO could you maybe build the aare-shared package into a wheel and then only copy/install that
#  instead of having it both as source in tree and installed in the venv?
#  Of course the deps are the bigger problem, but still. Or use una.

# install packages from our code etc.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --extra cpu --package=aare-oraku-prediction

# must be same base image as builder, otherwise copying venv won't work
FROM python:3.12-slim-bookworm

# copy the application and built venv from the builder
COPY --from=builder --chown=app:app /app /app

ENV PATH="/app/.venv/bin:$PATH"
ENV PROJECT_ROOT=/app

CMD ["python", "/app/src/prediction-service/main.py"]
