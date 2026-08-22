# Copyright 2018 Christoph Heindl.
# Copyright 2026 Yurui Ming.
# Licensed under MIT License
# ============================================================

from ._version import __version__
from .figure import figure_tensor, blittable_figure_tensor
from .create import create_figure, create_figures
from . import plots

__all__ = ["figure_tensor", "blittable_figure_tensor", 
           "create_figure", "create_figures", 
           "plots", "__version__"]
