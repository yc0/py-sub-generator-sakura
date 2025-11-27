"""Voice Activity Detection helper using WebRTC VAD."""

import logging
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from ..models.video_data import AudioData

logger = logging.getLogger(__name__)


@dataclass
class Frame:
    """Represents a fixed-size audio frame for VAD."""

    payload: bytes
    timestamp: float
    duration: float


class VADProcessor:
    """Splits audio into speech-only intervals using WebRTC VAD."""

    def __init__(
        self,
        enabled: bool = True,
        mode: int = 3,
        frame_duration_ms: int = 30,
        padding_ms: int = 300,
        min_segment_duration: float = 0.5,
    ) -> None:
        self.enabled = enabled
        self.mode = mode
        self.frame_duration_ms = frame_duration_ms
        self.padding_ms = padding_ms
        self.min_segment_duration = min_segment_duration
        self.vad = self._initialize_vad()

    def _initialize_vad(self):
        if not self.enabled:
            return None

        try:
            import webrtcvad

            vad = webrtcvad.Vad(self.mode)
            return vad
        except ImportError:
            logger.warning("webrtcvad not installed; VAD disabled")
            self.enabled = False
            return None
        except Exception as exc:
            logger.warning("Failed to initialize webrtcvad: %s", exc)
            self.enabled = False
            return None

    def get_speech_intervals(self, audio_data: AudioData) -> List[Tuple[float, float]]:
        """Return speech intervals detected inside the given audio data."""
        if not self.enabled or not self.vad:
            return [(0.0, audio_data.duration)] if audio_data.duration > 0 else []

        frames = self._frame_generator(audio_data)
        if not frames:
            return []

        intervals: List[Tuple[float, float]] = []
        speech_start = None

        for frame in frames:
            try:
                is_speech = self.vad.is_speech(frame.payload, audio_data.sample_rate)
            except Exception as exc:
                logger.debug("VAD frame processing failed: %s", exc)
                is_speech = False

            if is_speech and speech_start is None:
                speech_start = frame.timestamp
            elif not is_speech and speech_start is not None:
                intervals.append((speech_start, frame.timestamp + frame.duration))
                speech_start = None

        if speech_start is not None:
            last_frame = frames[-1]
            intervals.append((speech_start, last_frame.timestamp + last_frame.duration))

        total_duration = audio_data.duration
        merged = self._merge_intervals(intervals)
        padded = [self._apply_padding(start, end, total_duration) for start, end in merged]

        speech_segments: List[Tuple[float, float]] = []
        for start, end in padded:
            if end - start >= self.min_segment_duration:
                speech_segments.append((start, end))

        logger.info("VAD detected %d speech intervals", len(speech_segments))
        return speech_segments

    def _frame_generator(self, audio_data: AudioData) -> List[Frame]:
        duration_seconds = audio_data.duration
        if duration_seconds <= 0:
            return []

        sample_rate = audio_data.sample_rate
        frame_sample_count = int(sample_rate * self.frame_duration_ms / 1000)
        if frame_sample_count <= 0:
            return []

        bytes_per_frame = frame_sample_count * 2
        int16_audio = self._to_int16(audio_data.audio_array)

        frames: List[Frame] = []
        offset = 0
        timestamp = 0.0
        frame_duration = frame_sample_count / sample_rate

        while offset + frame_sample_count <= len(int16_audio):
            frame_samples = int16_audio[offset : offset + frame_sample_count]
            frames.append(Frame(frame_samples.tobytes(), timestamp, frame_duration))
            timestamp += frame_duration
            offset += frame_sample_count

        # Capture remaining tail as a shorter frame
        if offset < len(int16_audio):
            remaining = int16_audio[offset:]
            frame_duration = len(remaining) / sample_rate
            if frame_duration >= self.frame_duration_ms / 1000:
                frames.append(Frame(remaining.tobytes(), timestamp, frame_duration))

        return frames

    def _to_int16(self, audio_array: np.ndarray) -> np.ndarray:
        clipped = np.clip(audio_array, -1.0, 1.0)
        return (clipped * 32767).astype(np.int16)

    def _merge_intervals(self, intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        if not intervals:
            return []

        sorted_intervals = sorted(intervals, key=lambda x: x[0])
        merged: List[Tuple[float, float]] = [sorted_intervals[0]]

        for current_start, current_end in sorted_intervals[1:]:
            last_start, last_end = merged[-1]
            if current_start <= last_end:
                merged[-1] = (last_start, max(last_end, current_end))
            else:
                merged.append((current_start, current_end))

        return merged

    def _apply_padding(self, start: float, end: float, total_duration: float) -> Tuple[float, float]:
        padding = self.padding_ms / 1000
        padded_start = max(0.0, start - padding)
        padded_end = min(total_duration, end + padding)
        return padded_start, padded_end