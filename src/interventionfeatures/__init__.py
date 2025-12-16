"""
interventionfeatures: Mechanistic interpretability using CSS and activation patching.

Main components:
- core: CSS direction finding, model interventions, data handling
- core.task_css: Task-targeted CSS direction finding for benchmarks
- analysis: Feature explanation and faithfulness testing
- config: Hydra-based configuration
"""

__version__ = "0.3.0"

from .config import Config, DiscoverConfig, set_global_seed, setup_device
from .core.css import CSSDirectionFinder
from .core.data_handler import TransformerDataHandler
from .core.model import IntervenableTransformerSegment
from .core.task_css import TaskTargetedCSSFinder
from .core.task_data_handler import TaskDataHandler

__all__ = [
    "Config",
    "DiscoverConfig",
    "CSSDirectionFinder",
    "TaskTargetedCSSFinder",
    "TaskDataHandler",
    "TransformerDataHandler",
    "IntervenableTransformerSegment",
    "set_global_seed",
    "setup_device",
]
