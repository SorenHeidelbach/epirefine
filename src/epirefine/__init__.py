from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("epirefine")
except PackageNotFoundError:  # running from a source tree without installation
    __version__ = "unknown"
