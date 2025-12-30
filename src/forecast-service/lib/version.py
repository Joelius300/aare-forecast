from importlib.metadata import PackageNotFoundError

version = None
try:
    import importlib.metadata

    # this works fine IF the package is installed. In our current setup, we don't install the aare-oraku-* app packages.
    version = importlib.metadata.version(__package__ or __name__)
except PackageNotFoundError:
    import tomllib
    from pathlib import Path

    # so instead we literally read the pyproject.toml. I don't like it but hey it works,
    # and it's cheap to have in the docker image as well, so it doesn't really matter I guess.
    path = Path(__file__).parent.parent / "pyproject.toml"
    with open(path, "rb") as file:
        version = tomllib.load(file)["project"]["version"]
        assert isinstance(version, str), "version isn't a string"

__version__ = version
