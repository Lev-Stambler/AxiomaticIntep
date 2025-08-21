"""
FourierFeatures: A package for mechanistic interpretability using Fourier features and Causal Scrubbing Search (CSS).

This package provides tools for:
- Finding interpretable directions in neural networks using CSS
- Analyzing activation patterns and feature representations
- Manual and automated explanation of neural network features
- Faithfulness evaluation of explanations

Main components:
- core: Core algorithms (CSS, model interventions, data handling)
- analysis: Analysis and interpretation tools (explainers, faithfulness testing)
- utils: Utilities (configuration, visualization, math utilities)
- cli: Command-line interface
"""

__version__ = "0.1.0"
__author__ = ""

from .analysis.explainer import FeatureExplainer

# Import main classes for convenience
from .core.css import CSSDirectionFinder
from .core.data_handler import TransformerDataHandler
from .core.model import IntervenableTransformerSegment
from .utils.config import MainConfig

__all__ = [
    "CSSDirectionFinder",
    "TransformerDataHandler",
    "IntervenableTransformerSegment",
    "FeatureExplainer",
    "MainConfig",
]
