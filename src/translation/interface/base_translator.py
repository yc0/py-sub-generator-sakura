"""Base class for translation implementations."""

import logging
from abc import ABC, abstractmethod
from typing import Callable, List, Optional

from ...models.subtitle_data import SubtitleSegment, TranslationResult

logger = logging.getLogger(__name__)


class BaseTranslator(ABC):
    """Abstract base class for translation implementations."""

    def __init__(
        self,
        model_name: str,
        source_lang: str,
        target_lang: str,
        device: str = "auto",
        **kwargs,
    ):
        """Initialize translator.

        Args:
            model_name: Name/path of translation model
            source_lang: Source language code
            target_lang: Target language code
            device: Device to run on ('auto', 'cpu', 'cuda', 'mps')
            **kwargs: Additional model-specific parameters
        """
        self.model_name = model_name
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.device = self._resolve_device(device)
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        self.kwargs = kwargs

    @abstractmethod
    def load_model(self) -> bool:
        """Load the translation model.

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    def translate_text(
        self, text: str, progress_callback: Optional[Callable[[float], None]] = None
    ) -> TranslationResult:
        """Translate a single text string.

        Args:
            text: Text to translate
            progress_callback: Optional progress callback

        Returns:
            Translation result
        """
        pass

    @abstractmethod
    def translate_batch(
        self,
        texts: List[str],
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> List[TranslationResult]:
        """Translate multiple texts in batch.

        Args:
            texts: List of texts to translate
            progress_callback: Optional progress callback

        Returns:
            List of translation results
        """
        pass

    def translate_segments(
        self,
        segments: List[SubtitleSegment],
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> List[TranslationResult]:
        """Translate subtitle segments.

        Args:
            segments: List of subtitle segments
            progress_callback: Optional progress callback

        Returns:
            List of translation results
        """
        texts = [segment.text for segment in segments]
        return self.translate_batch(texts, progress_callback)

    def _resolve_device(self, device: str) -> str:
        """Resolve device string to actual device.

        Args:
            device: Device specification

        Returns:
            Resolved device string
        """
        if device == "auto":
            try:
                import torch

                # Priority: CUDA > MPS (Apple Silicon) > CPU
                if torch.cuda.is_available():
                    return "cuda"
                elif (
                    hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                ):
                    return "mps"
                else:
                    return "cpu"
            except ImportError:
                return "cpu"
        return device

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text before translation.

        Args:
            text: Input text

        Returns:
            Preprocessed text
        """
        # Basic text cleaning
        text = text.strip()

        # Remove excessive whitespace
        import re

        text = re.sub(r"\s+", " ", text)

        return text

    def _postprocess_text(self, text: str) -> str:
        """Postprocess translated text.

        Args:
            text: Translated text

        Returns:
            Postprocessed text
        """
        # Basic cleaning
        text = text.strip()

        # Remove extra spaces
        import re

        text = re.sub(r"\s+", " ", text)

        return text

    def unload_model(self):
        """Unload model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        self.is_loaded = False

        # Force garbage collection
        try:
            import gc

            gc.collect()

            # Clear CUDA cache if available
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def __enter__(self):
        """Context manager entry."""
        self.load_model()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.unload_model()
