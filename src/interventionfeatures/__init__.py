"""
interventionfeatures: Mechanistic interpretability using CSS and activation patching.

Main components:
- core: CSS direction finding, model interventions, data handling
- analysis: Feature explanation and faithfulness testing
- config: Hydra-based configuration
"""

__version__ = "0.2.0"

from .config import Config, set_global_seed, setup_device
from .core.css import CSSDirectionFinder
from .core.data_handler import TransformerDataHandler
from .core.model import IntervenableTransformerSegment

__all__ = [
    "Config",
    "CSSDirectionFinder",
    "TransformerDataHandler",
    "IntervenableTransformerSegment",
    "set_global_seed",
    "setup_device",
]
