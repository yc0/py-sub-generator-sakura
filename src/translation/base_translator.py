"""Base class for translation implementations."""

from abc import ABC, abstractmethod
from typing import Callable, List, Optional

from ..models.subtitle_data import TranslationResult


class BaseTranslator(ABC):
    """Abstract base class for translation implementations."""

    def __init__(self, model_name: str, source_lang: str, target_lang: str, device: str = "auto", **kwargs):
        """Initialize the translator.
        
        Args:
            model_name: Name of the model to use
            source_lang: Source language code (e.g., 'ja', 'en')
            target_lang: Target language code (e.g., 'zh', 'en')
            device: Device to use ('cpu', 'cuda', 'mps', 'auto')
        """
        self.model_name = model_name
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.device = self._resolve_device(device)
        self.is_loaded = False

    @abstractmethod
    def load_model(self) -> bool:
        """Load the model. Returns True if successful."""
        pass

    @abstractmethod
    def translate_text(self, text: str, progress_callback: Optional[Callable[[float], None]] = None) -> TranslationResult:
        """Translate a single text string."""
        pass

    @abstractmethod
    def translate_batch(self, texts: List[str], progress_callback: Optional[Callable[[float], None]] = None) -> List[TranslationResult]:
        """Translate a batch of texts."""
        pass

    @abstractmethod
    def unload_model(self):
        """Unload the model from memory"""
        pass

    def _resolve_device(self, device):
        """Resolve device string to actual device"""
        if device == "auto":
            import torch
            if torch.backends.mps.is_available():
                return "mps"
            elif torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return device
