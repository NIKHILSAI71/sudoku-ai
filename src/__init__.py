"""Source code for production Sudoku AI system."""

__version__ = "2.0.0"
__author__ = "Sudoku AI Team"

from . import core
from . import models
from . import training
from . import inference
from . import data
from . import utils

__all__ = [
    'core',
    'models',
    'training',
    'inference',
    'data',
    'utils'
]
