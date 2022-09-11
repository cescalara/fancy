name = "fancy"

from .interfaces.data import Data
from .analysis import Analysis
from .interfaces.stan import Model

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions
