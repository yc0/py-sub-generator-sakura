"""Audio processing utilities for video files."""

import logging
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

from ..models.video_data import AudioData, VideoFile

logger = logging.getLogger(__name__)


class AudioProcessor:
    """Processes audio extraction and preparation for ASR."""

    def __init__(self, target_sample_rate: int = 16000):
        """Initialize audio processor.

        Args:
            target_sample_rate: Target sample rate for ASR models (default: 16kHz)
        """
        self.target_sample_rate = target_sample_rate
        self.hwaccel = self._detect_hardware_acceleration()

    def _detect_hardware_acceleration(self) -> Optional[str]:
        """Detect the best available hardware acceleration.
        
        Returns:
            Hardware acceleration method or None for software fallback
        """
        try:
            import subprocess
            import platform
            
            # Get available hardware accelerators
            result = subprocess.run(
                ["ffmpeg", "-hwaccels"], 
                capture_output=True, 
                text=True, 
                check=True
            )
            available_hwaccels = result.stdout.lower()
            
            # Priority order: CUDA > VideoToolbox > software
            if "cuda" in available_hwaccels:
                logger.info("Using CUDA hardware acceleration")
                return "cuda"
            elif "videotoolbox" in available_hwaccels and platform.system() == "Darwin":
                logger.info("Using VideoToolbox hardware acceleration (Apple Silicon)")
                return "videotoolbox"
            else:
                logger.info("Using software decoding (no hardware acceleration available)")
                return None
                
        except Exception as e:
            logger.warning(f"Could not detect hardware acceleration: {e}")
            return None

    def extract_audio_from_video(self, video_file: VideoFile) -> Optional[AudioData]:
        """Extract audio from video file.

        Args:
            video_file: VideoFile object containing video metadata

        Returns:
            AudioData object or None if extraction fails
        """
        try:
            import ffmpeg

            logger.info(f"Extracting audio from: {video_file.filename}")

            # Create temporary audio file
            temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            temp_audio_path = temp_audio.name
            temp_audio.close()

            # Use ffmpeg with hardware acceleration to extract audio
            self._extract_audio_with_fallback(video_file.file_path, temp_audio_path)

            # Load audio data
            audio_data = self.load_audio_file(Path(temp_audio_path))

            # Cleanup temp file
            Path(temp_audio_path).unlink()

            return audio_data

        except ImportError:
            logger.error(
                "ffmpeg-python not installed. Please install it: pip install ffmpeg-python"
            )
            return None
        except Exception as e:
            logger.error(f"Error extracting audio: {e}")
            return None

    def _extract_audio_with_fallback(self, input_path: Path, output_path: str) -> None:
        """Extract audio with hardware acceleration and software fallback.
        
        Args:
            input_path: Input video file path
            output_path: Output audio file path
        """
        import ffmpeg
        
        # Try hardware acceleration first
        if self.hwaccel:
            try:
                (
                    ffmpeg.input(str(input_path), hwaccel=self.hwaccel)
                    .output(
                        output_path,
                        acodec="pcm_s16le",  # 16-bit PCM
                        ar=self.target_sample_rate,  # Target sample rate
                        ac=1,  # Mono channel
                    )
                    .overwrite_output()
                    .run(quiet=True)
                )
                logger.debug(f"Audio extracted using {self.hwaccel} acceleration")
                return
            except Exception as e:
                logger.warning(f"Hardware acceleration failed ({self.hwaccel}): {e}")
                logger.info("Falling back to software decoding")
        
        # Fallback to software decoding
        (
            ffmpeg.input(str(input_path))
            .output(
                output_path,
                acodec="pcm_s16le",  # 16-bit PCM
                ar=self.target_sample_rate,  # Target sample rate
                ac=1,  # Mono channel
            )
            .overwrite_output()
            .run(quiet=True)
        )
        logger.debug("Audio extracted using software decoding")

    def load_audio_file(self, audio_path: Path) -> Optional[AudioData]:
        """Load audio file and return AudioData object.

        Args:
            audio_path: Path to audio file

        Returns:
            AudioData object or None if loading fails
        """
        try:
            import librosa

            # Load audio file
            audio_array, sample_rate = librosa.load(
                str(audio_path), sr=self.target_sample_rate, mono=True
            )

            # Calculate duration
            duration = len(audio_array) / sample_rate

            return AudioData(
                audio_array=audio_array,
                sample_rate=sample_rate,
                duration=duration,
                channels=1,
                bit_depth=16,
            )

        except ImportError:
            logger.error(
                "librosa not installed. Please install it: pip install librosa"
            )
            return None
        except Exception as e:
            logger.error(f"Error loading audio file: {e}")
            return None

    def preprocess_audio_for_asr(self, audio_data: AudioData) -> np.ndarray:
        """Preprocess audio data for ASR models.

        Args:
            audio_data: AudioData object

        Returns:
            Preprocessed audio array
        """
        try:
            audio_array = audio_data.audio_array

            # Normalize audio to [-1, 1] range
            if audio_array.dtype != np.float32:
                audio_array = audio_array.astype(np.float32)

            # Normalize to prevent clipping
            max_val = np.abs(audio_array).max()
            if max_val > 0:
                audio_array = audio_array / max_val

            # Remove silence from beginning and end
            audio_array = self._trim_silence(audio_array)

            logger.info(
                f"Audio preprocessed: {len(audio_array)} samples, "
                f"{audio_data.sample_rate}Hz, {audio_data.duration:.2f}s"
            )

            return audio_array

        except Exception as e:
            logger.error(f"Error preprocessing audio: {e}")
            return audio_data.audio_array

    def _trim_silence(
        self, audio_array: np.ndarray, threshold: float = 0.01
    ) -> np.ndarray:
        """Remove silence from beginning and end of audio.

        Args:
            audio_array: Audio array to trim
            threshold: Silence threshold (0-1)

        Returns:
            Trimmed audio array
        """
        try:
            import librosa

            # Find non-silent portions
            intervals = librosa.effects.split(
                audio_array,
                top_db=20,  # Consider anything below -20dB as silence
                frame_length=2048,
                hop_length=512,
            )

            if len(intervals) > 0:
                # Concatenate all non-silent intervals
                trimmed_audio = np.concatenate(
                    [audio_array[start:end] for start, end in intervals]
                )
                return trimmed_audio
            else:
                return audio_array

        except ImportError:
            logger.warning("librosa not available, skipping silence trimming")
            return audio_array
        except Exception as e:
            logger.warning(f"Error trimming silence: {e}")
            return audio_array

    def split_audio_chunks(
        self, audio_data: AudioData, chunk_duration: float = 30.0, overlap: float = 1.0
    ) -> list[AudioData]:
        """Split audio into overlapping chunks for processing.

        Args:
            audio_data: AudioData to split
            chunk_duration: Duration of each chunk in seconds
            overlap: Overlap between chunks in seconds

        Returns:
            List of AudioData chunks
        """
        try:
            audio_array = audio_data.audio_array
            sample_rate = audio_data.sample_rate

            chunk_samples = int(chunk_duration * sample_rate)
            overlap_samples = int(overlap * sample_rate)
            step_samples = chunk_samples - overlap_samples

            chunks = []
            start = 0

            while start < len(audio_array):
                end = min(start + chunk_samples, len(audio_array))
                chunk_audio = audio_array[start:end]

                # Skip very short chunks
                if len(chunk_audio) < sample_rate:  # Less than 1 second
                    break

                chunk_duration_actual = len(chunk_audio) / sample_rate

                chunks.append(
                    AudioData(
                        audio_array=chunk_audio,
                        sample_rate=sample_rate,
                        duration=chunk_duration_actual,
                        channels=audio_data.channels,
                        bit_depth=audio_data.bit_depth,
                    )
                )

                start += step_samples

            logger.info(f"Audio split into {len(chunks)} chunks")
            return chunks

        except Exception as e:
            logger.error(f"Error splitting audio: {e}")
            return [audio_data]
