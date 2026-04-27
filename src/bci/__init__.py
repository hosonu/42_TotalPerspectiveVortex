"""BCI pipeline building blocks.

CSP internals are implemented in this project (see :mod:`bci.csp`).
"""

from bci.csp import CustomCSP
from bci.data import epochs_to_Xy
from bci.features import LogVarianceTransformer
from bci.pipeline import make_motor_imagery_pipeline

__all__ = [
    "CustomCSP",
    "LogVarianceTransformer",
    "epochs_to_Xy",
    "make_motor_imagery_pipeline",
]
