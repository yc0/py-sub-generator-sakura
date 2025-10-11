"""Translation modules for multi-language subtitle generation."""

# Concrete implementations (user-facing)
from .huggingface_translator import HuggingFaceTranslator
from .sakura_translator_llama_cpp import SakuraTranslator

# Pipeline coordinator
from .translation_pipeline import TranslationPipeline

__all__ = [
    # Concrete translators
    "HuggingFaceTranslator",
    "SakuraTranslator",
    # Pipeline
    "TranslationPipeline",
]
