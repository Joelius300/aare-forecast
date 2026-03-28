from importlib.metadata import version, PackageNotFoundError

try:
    # cannot derive distribution name from import name (__package__) unfortunately
    __version__ = version("aare-oraku-api")
except PackageNotFoundError:
    __version__ = "0.0.0"
