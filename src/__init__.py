"""Mammography Detection Package

This package contains modules for mammography image detection and classification
using deep learning techniques.

Modules:
    - preprocessing: Image preprocessing and enhancement
    - feature_extraction: Feature extraction from mammography images
    - models: Deep learning model architectures
    - train: Model training and evaluation
"""

__version__ = '1.0.0'
__author__ = 'Your Name'

from . import preprocessing
from . import feature_extraction
from . import models
from . import train

__all__ = ['preprocessing', 'feature_extraction', 'models', 'train']
