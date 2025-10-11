"""Translation modules for multi-language subtitle generation."""

# Base/Interface classes (in /base/ directory - for inheritance only)
# Concrete implementations (user-facing)
from .huggingface_translator import HuggingFaceTranslator
from .interface import BaseTranslator, PyTorchTranslator
from .sakura_translator import SakuraTranslator

# Pipeline coordinator
from .translation_pipeline import TranslationPipeline

__all__ = [
    # Base/Interface classes
    "BaseTranslator",
    "PyTorchTranslator",
    # Concrete translators
    "HuggingFaceTranslator",
    "SakuraTranslator",
    # Pipeline
    "TranslationPipeline",
]
