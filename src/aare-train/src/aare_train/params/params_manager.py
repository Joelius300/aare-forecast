import logging
from functools import cache
from pathlib import Path
from typing import Optional, cast

from aare_train.params.params_types import Params
from aare_train.paths import PROJECT_ROOT

logger = logging.getLogger(__name__)


@cache
def _read_dvc_params():
    import dvc.api

    return cast(Params, dvc.api.params_show())


@cache
def _read_params_file(path: str | Path):
    import yaml

    with open(path, "rt") as file:
        return cast(Params, yaml.safe_load(file))


class _ParamsManager:
    """
    A manager that returns and caches the DVC params or the values from a specified params file.
    To reduce the chance of misconfigurations, races and confusion, it's heavily restricted. By default, it's configured
    to return the DVC params, but you can alternatively set a params file as source (only prior to the first access).

    Intended to be used as singleton.
    """

    def __init__(self):
        self._params_file: Optional[Path] = None
        self._read_params_was_called = False

    @cache
    def _is_dvc(self):
        return (PROJECT_ROOT / ".dvc").is_dir()

    @property
    def is_dvc(self):
        """Check if we're currently running in a DVC context. ONLY EVALUATED ONCE."""
        # cached_property is writable, so use property explicitly
        return self._is_dvc()

    @property
    def params_file(self):
        """Get the path to the params file that is set currently."""
        return self._params_file

    # want explicit setter function for clarity
    def set_params_file(self, params_file_path: str | Path):
        """Sets the path of the params file. Can only be set once and will be used from then on!"""
        if self._read_params_was_called:
            raise ValueError("Cannot set params file after the params manager was already used!")

        if self._params_file is not None:
            raise ValueError(f"Params file is already set to '{self._params_file}', cannot change it!")

        if self.is_dvc:
            logger.warning("Custom params file was set despite running in a DVC context! DVC params will be ignored.")

        params_file_path = Path(params_file_path)
        if not params_file_path.is_file():
            raise ValueError(f"The params file '{params_file_path}' does not exist or isn't a valid file.")

        self._params_file = params_file_path

    def read_params(self, *, ensure_dvc=False) -> Params:
        """
        Returns the params with appropriate typing from the configured source.
        Set ensure_dvc to true to raise an error if not running in a DVC context.
        """
        self._read_params_was_called = True

        if not self.is_dvc:
            if ensure_dvc:
                raise ValueError("Called read_params with ensure_dvc outside of a DVC context!")

            if self._params_file is None:
                raise ValueError("Called read_params outside of a DVC context but no params file path is set!")

            try:
                return _read_params_file(self._params_file)
            except Exception as e:
                raise ValueError(f"Could not read params from file '{self.params_file}': {e}") from e

        try:
            return _read_dvc_params()
        except Exception as e:
            raise ValueError(f"Could not read DVC params: {e}") from e


params_manager_singleton = _ParamsManager()
"""DO NOT USE DIRECTLY. The singleton manager to configure param handling for the application."""
