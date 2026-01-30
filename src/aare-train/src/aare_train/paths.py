import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def find_project_root(raise_not_found: bool = True, allow_env: bool = True) -> Path:
    """Traverse CWD up to the project root and return its path. Take PROJECT_ROOT env if set."""
    if allow_env and (root := os.getenv("PROJECT_ROOT")):
        logger.debug(f"Project root configured via env variable: {root}")
        return Path(root)

    cwd = Path(os.getcwd())
    cur = cwd
    while True:
        # should be enough to identify the project root
        if (cur / ".dvc").is_dir() and (cur / "uv.lock").is_file():
            return cur

        parent = cur.parent

        if parent == cur:
            # reached file system root
            if raise_not_found:
                raise ValueError("Could not find project root!")
            else:
                logger.warning(f"COULD NOT DETERMINE PROJECT ROOT, USING CWD: '{cwd}'")
                return cwd

        cur = parent


# suboptimal that this runs on import, but works and avoids refactoring many things
PROJECT_ROOT = find_project_root(raise_not_found=False)

DATA_FOLDER: Path = PROJECT_ROOT / "data"
METRICS_FOLDER = DATA_FOLDER / "metrics"
FORECAST_SAMPLES_FOLDER = DATA_FOLDER / "forecast_samples"
MODELS_FOLDER = DATA_FOLDER / "models"
OPTUNA_STORE = DATA_FOLDER / "optuna-trials.db"
OPTUNA_STORE_URI = f"sqlite:///{OPTUNA_STORE}"
