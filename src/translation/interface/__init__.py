"""Interface classes for translation implementations."""

from .base_translator import BaseTranslator
from .pytorch_translator import PyTorchTranslator

__all__ = [
    'BaseTranslator',
    'PyTorchTranslator'
]