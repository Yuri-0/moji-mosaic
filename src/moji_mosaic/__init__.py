"""
Moji-mosaic: AI-powered text-to-artistic-image generator.

This package provides functionality to:
1. Generate mosaic images from text
2. Extract keywords using BERT models
3. Generate artistic images using diffusion models
"""

__version__ = "2.0.0"
__author__ = "Yuri"

from .core.mosaic_generator import MosaicGenerator
from .core.keyword_extractor import KeywordExtractor
from .core.diffusion_generator import DiffusionGenerator
from .pipeline import MojiMosaicPipeline

__all__ = [
    "MosaicGenerator",
    "KeywordExtractor", 
    "DiffusionGenerator",
    "MojiMosaicPipeline",
]
