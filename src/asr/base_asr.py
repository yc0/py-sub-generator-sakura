"""Base class for ASR implementations."""

from abc import ABC, abstractmethod
from typing import Callable, List, Optional

from ..models.subtitle_data import SubtitleSegment
from ..models.video_data import AudioData


class BaseASR(ABC):
    """Abstract base class for ASR implementations."""

    def __init__(self, model_name: str, device: str = "auto", **kwargs):
        """Initialize ASR model.

        Args:
            model_name: Name/path of the ASR model
            device: Device to run on ('auto', 'cpu', 'cuda', 'mps')
            **kwargs: Additional model-specific parameters
        """
        self.model_name = model_name
        self.device = self._resolve_device(device)
        self.model = None
        self.processor = None
        self.is_loaded = False

    @abstractmethod
    def load_model(self) -> bool:
        """Load the ASR model.

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    def transcribe_audio(
        self,
        audio_data: AudioData,
        language: str = "ja",
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> List[SubtitleSegment]:
        """Transcribe audio to text with timestamps.

        Args:
            audio_data: Audio data to transcribe
            language: Source language code
            progress_callback: Optional callback for progress updates

        Returns:
            List of subtitle segments with timestamps
        """
        pass

    @abstractmethod
    def transcribe_batch(
        self,
        audio_chunks: List[AudioData],
        language: str = "ja",
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> List[SubtitleSegment]:
        """Transcribe multiple audio chunks.

        Args:
            audio_chunks: List of audio chunks to transcribe
            language: Source language code
            progress_callback: Optional callback for progress updates

        Returns:
            List of subtitle segments with timestamps
        """
        pass

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

    def unload_model(self):
        """Unload model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
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
