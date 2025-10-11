"""Translation modules for multi-language subtitle generation."""

from .base_translator import BaseTranslator
from .huggingface_translator import HuggingFaceTranslator
from .translation_pipeline import TranslationPipeline

__all__ = [
    'BaseTranslator',
    'HuggingFaceTranslator', 
    'TranslationPipeline'
]