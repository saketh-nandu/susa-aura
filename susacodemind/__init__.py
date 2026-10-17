"""
SUSA-1B CodeMind — Backward Compatibility Layer.
DEPRECATED: CodeMind has evolved into SUSA AURA.
Please update imports from `susacodemind` to `aura`.
"""

import warnings
from aura import __author__, __version__

warnings.warn(
    "The 'susacodemind' package is deprecated and will be removed in a future release. "
    "Please import from 'aura' instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["__version__", "__author__"]
