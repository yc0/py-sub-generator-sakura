"""Core subtitle generation orchestrator."""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..asr.whisper_asr import WhisperASR
from ..models.subtitle_data import SubtitleFile, SubtitleSegment
from ..models.video_data import AudioData, VideoFile
from ..translation.translation_pipeline import TranslationPipeline
from ..utils.audio_processor import AudioProcessor
from ..utils.config import Config
from ..utils.file_handler import FileHandler
from ..utils.logger import LoggerMixin

logger = logging.getLogger(__name__)


class SubtitleGenerator(LoggerMixin):
    """Main subtitle generation orchestrator."""

    def __init__(self, config: Config):
        """Initialize subtitle generator.

        Args:
            config: Application configuration
        """
        self.config = config
        self.file_handler = FileHandler()
        self.audio_processor = AudioProcessor(
            target_sample_rate=self.config.get("asr.sample_rate", 16000)
        )

        # Initialize ASR
        asr_config = self.config.get_asr_config()
        self.asr = WhisperASR(
            model_name=asr_config.get("model_name", "openai/whisper-large-v3"),
            device=asr_config.get("device", "auto"),
            language=asr_config.get("language", "ja"),
        )

        # Initialize translation pipeline
        self.translation_pipeline = TranslationPipeline(config)

        # Processing state
        self.current_video = None
        self.current_audio = None
        self.temp_files = []

    def process_video_file(
        self,
        video_path: Path,
        target_languages: List[str] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> Optional[SubtitleFile]:
        """Process video file to generate subtitles with translations.

        Args:
            video_path: Path to video file
            target_languages: List of target languages ['en', 'zh']
            progress_callback: Callback for progress updates (stage, progress)

        Returns:
            SubtitleFile with translations or None if failed
        """
        if target_languages is None:
            target_languages = ["en", "zh"]

        try:
            self.logger.info(f"Starting subtitle generation for: {video_path}")


            # Stage 1: Validate and load video
            if progress_callback:
                progress_callback("validation", 0.0)
            video_file = self._validate_and_load_video(video_path)
            if not video_file:
                return None
            if progress_callback:
                progress_callback("validation", 1.0)

            # Stage 2: Extract audio
            if progress_callback:
                progress_callback("audio_extraction", 0.0)
            audio_data = self._extract_audio(video_file)
            if not audio_data:
                return None
            if progress_callback:
                progress_callback("audio_extraction", 1.0)

            # Stage 3: Perform ASR
            if progress_callback:
                progress_callback("asr", 0.0)
            segments = self._perform_asr(
                audio_data,
                progress_callback=progress_callback,
            )
            # Defensive: log and check types before boolean check
            try:
                for idx, seg in enumerate(segments):
                    import numpy as np
                    if isinstance(seg, np.ndarray):
                        self.logger.error(f"[SubtitleGenerator] Segment {idx} is a numpy array, which is invalid. Segment: {seg}")
                    elif not hasattr(seg, 'start_time') or not hasattr(seg, 'end_time'):
                        self.logger.error(f"[SubtitleGenerator] Segment {idx} is not a SubtitleSegment: {type(seg)} {seg}")
            except Exception as e:
                import traceback
                self.logger.error(f"[SubtitleGenerator] Exception while checking segment types: {e}\n{traceback.format_exc()}")
            if not segments:
                self.logger.error("ASR failed to produce any segments")
                return None
            # Do not set progress_callback("asr", 1.0) here; let ASR/translation pipeline handle fine-grained progress

            # Create subtitle file
            subtitle_file = SubtitleFile(
                segments=segments,
                video_file=video_path,
                source_language="ja",
                target_languages=target_languages,
            )

            # Stage 4: Perform translation
            if target_languages:
                if progress_callback:
                    progress_callback("translation", 0.0)
                subtitle_file = self.translation_pipeline.translate_subtitle_file(
                    subtitle_file,
                    target_languages=target_languages,
                    progress_callback=progress_callback,
                )
                if progress_callback:
                    progress_callback("translation", 1.0)

            self.logger.info(f"Subtitle generation completed: {len(segments)} segments")
            return subtitle_file

        except Exception as e:
            self.logger.error(f"Error processing video file: {e}")
            return None

        finally:
            # Cleanup temp files
            self._cleanup_temp_files()

    def _validate_and_load_video(self, video_path: Path) -> Optional[VideoFile]:
        """Validate and load video file with metadata.

        Args:
            video_path: Path to video file

        Returns:
            VideoFile object or None if invalid
        """
        try:
            # Basic validation
            video_file = self.file_handler.create_video_file_object(video_path)
            if not video_file:
                return None

            # Extract detailed metadata
            video_file = self.file_handler.get_video_metadata(video_file)

            self.current_video = video_file
            self.logger.info(f"Video loaded: {video_file.get_display_info()}")

            return video_file

        except Exception as e:
            self.logger.error(f"Error loading video: {e}")
            return None

    def _extract_audio(self, video_file: VideoFile) -> Optional[AudioData]:
        """Extract audio from video file.

        Args:
            video_file: VideoFile object

        Returns:
            AudioData object or None if extraction failed
        """
        try:
            # Extract audio
            audio_data = self.audio_processor.extract_audio_from_video(video_file)
            if not audio_data:
                return None

            # Preprocess for ASR
            preprocessed_audio = self.audio_processor.preprocess_audio_for_asr(
                audio_data
            )
            audio_data.audio_array = preprocessed_audio

            self.current_audio = audio_data
            self.logger.info(
                f"Audio extracted: {audio_data.duration:.2f}s, {audio_data.sample_rate}Hz"
            )

            return audio_data

        except Exception as e:
            self.logger.error(f"Error extracting audio: {e}")
            return None

    def _perform_asr(
        self,
        audio_data: AudioData,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> List[SubtitleSegment]:
        """Perform automatic speech recognition on audio.

        Args:
            audio_data: AudioData to transcribe
            progress_callback: Optional progress callback (stage, progress)

        Returns:
            List of subtitle segments
        """
        try:
            # Load ASR model
            if not self.asr.load_model():
                self.logger.error("Failed to load ASR model")
                return []

            # Check if we need to split audio into chunks
            chunk_length = self.config.get("asr.chunk_length", 8)  # Reduced for Japanese content
            overlap = self.config.get("asr.overlap", 0.5)  # Reduced overlap

            if (
                audio_data.duration > chunk_length * 1.5
            ):  # Use chunking for longer audio
                self.logger.info(f"Using chunked ASR: {chunk_length}s chunks with {overlap}s overlap")

                # Split audio into chunks
                audio_chunks = self.audio_processor.split_audio_chunks(
                    audio_data, chunk_duration=chunk_length, overlap=overlap
                )

                # Transcribe chunks
                segments = self.asr.transcribe_batch(
                    audio_chunks,
                    language=self.config.get("asr.language", "ja"),
                    progress_callback=progress_callback,
                )
            else:
                # Process entire audio at once
                segments = self.asr.transcribe_audio(
                    audio_data,
                    language=self.config.get("asr.language", "ja"),
                    progress_callback=progress_callback,
                )

            # Post-process segments for better subtitle readability
            segments = self._post_process_subtitle_segments(segments)

            self.logger.info(f"ASR completed: {len(segments)} segments generated")
            return segments

        except Exception as e:
            self.logger.error(f"Error performing ASR: {e}")
            return []

        finally:
            # Unload ASR model to free memory
            self.asr.unload_model()

    def _post_process_subtitle_segments(
        self, segments: List[SubtitleSegment]
    ) -> List[SubtitleSegment]:
        """Post-process subtitle segments for better readability and timing.

        Args:
            segments: Raw subtitle segments from ASR

        Returns:
            Processed subtitle segments with improved timing and readability
        """
        if not segments:
            return segments

        processed_segments = []
        max_subtitle_length = self.config.get("asr.max_subtitle_length", 25)  # Max characters per subtitle
        subtitle_display_duration = self.config.get("asr.subtitle_display_duration", 2.5)  # Max display time in seconds

        for segment in segments:
            # Skip segments that are too short or empty
            if not segment.text or not segment.text.strip():
                continue

            text = segment.text.strip()
            start_time = segment.start_time
            end_time = segment.end_time
            duration = end_time - start_time

            # If segment is too long in text or duration, split it
            if len(text) > max_subtitle_length or duration > subtitle_display_duration:
                # Calculate how many parts we need
                text_parts = self._split_text_by_length(text, max_subtitle_length)
                time_parts = self._split_time_by_duration(start_time, end_time, len(text_parts))

                # Create new segments for each part
                for i, (text_part, (part_start, part_end)) in enumerate(zip(text_parts, time_parts)):
                    processed_segments.append(SubtitleSegment(
                        start_time=part_start,
                        end_time=part_end,
                        text=text_part,
                        confidence=segment.confidence if hasattr(segment, 'confidence') else 1.0
                    ))
            else:
                # Segment is fine as-is, but ensure minimum display time
                min_display_time = 1.0  # Minimum 1 second display
                if duration < min_display_time:
                    # Extend the end time slightly, but don't overlap with next segment
                    extended_end = min(start_time + min_display_time, end_time + 0.5)
                    segment.end_time = extended_end

                processed_segments.append(segment)

        # Sort segments by start time
        processed_segments.sort(key=lambda x: x.start_time)

        self.logger.info(f"Post-processed {len(segments)} segments into {len(processed_segments)} readable subtitles")
        return processed_segments

    def _split_text_by_length(self, text: str, max_length: int) -> List[str]:
        """Split text into parts that don't exceed max_length characters.

        Args:
            text: Text to split
            max_length: Maximum characters per part

        Returns:
            List of text parts
        """
        if len(text) <= max_length:
            return [text]

        words = text.split()
        parts = []
        current_part = ""

        for word in words:
            # Check if adding this word would exceed the limit
            if len(current_part) + len(word) + 1 > max_length:  # +1 for space
                if current_part:
                    parts.append(current_part.strip())
                    current_part = word
                else:
                    # Word itself is too long, split it
                    parts.append(word[:max_length])
                    remaining = word[max_length:]
                    if remaining:
                        current_part = remaining
            else:
                if current_part:
                    current_part += " " + word
                else:
                    current_part = word

        if current_part:
            parts.append(current_part.strip())

        return parts if parts else [text]

    def _split_time_by_duration(
        self, start_time: float, end_time: float, num_parts: int
    ) -> List[tuple[float, float]]:
        """Split time duration into equal parts.

        Args:
            start_time: Start time of the segment
            end_time: End time of the segment
            num_parts: Number of parts to split into

        Returns:
            List of (start, end) time tuples
        """
        if num_parts <= 1:
            return [(start_time, end_time)]

        total_duration = end_time - start_time
        part_duration = total_duration / num_parts

        time_parts = []
        current_time = start_time

        for i in range(num_parts):
            part_start = current_time
            part_end = min(current_time + part_duration, end_time)
            time_parts.append((part_start, part_end))
            current_time = part_end

        return time_parts

    def export_subtitles(
        self, subtitle_file: SubtitleFile, output_dir: Path, formats: List[str] = None
    ) -> Dict[str, Path]:
        """Export subtitle file to various formats.

        Args:
            subtitle_file: SubtitleFile to export
            output_dir: Output directory
            formats: List of formats to export

        Returns:
            Dictionary mapping format names to file paths
        """
        try:
            return self.translation_pipeline.export_translated_subtitles(
                subtitle_file, output_dir, formats
            )
        except Exception as e:
            self.logger.error(f"Error exporting subtitles: {e}")
            return {}

    def _cleanup_temp_files(self):
        """Clean up temporary files."""
        if self.temp_files:
            self.file_handler.cleanup_temp_files(self.temp_files)
            self.temp_files.clear()

    def get_processing_info(self) -> Dict[str, Any]:
        """Get information about current processing state.

        Returns:
            Dictionary with processing information
        """
        info = {
            "video_loaded": self.current_video is not None,
            "audio_extracted": self.current_audio is not None,
            "asr_loaded": self.asr.is_loaded,
            "translation_ready": self.translation_pipeline.multi_stage_translator
            is not None,
        }

        if self.current_video:
            info["video_info"] = self.current_video.get_display_info()

        if self.current_audio:
            info["audio_info"] = {
                "duration": self.current_audio.duration,
                "sample_rate": self.current_audio.sample_rate,
                "channels": self.current_audio.channels,
            }

        return info

    def cleanup(self):
        """Clean up resources and temporary files."""
        try:
            # Unload models
            self.asr.unload_model()
            self.translation_pipeline.unload_models()

            # Clean temp files
            self._cleanup_temp_files()

            # Reset state
            self.current_video = None
            self.current_audio = None

            self.logger.info("Subtitle generator cleanup completed")

        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
