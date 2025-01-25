name = "fancy"

from .interfaces.data import Data
from .analysis import Analysis
from .interfaces.model import Model
from .simulation.simulation import Simulation

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions

from . import _version
__version__ = _version.get_versions()['version']
