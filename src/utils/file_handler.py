"""File handling utilities for video and subtitle files."""

import logging
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional

from ..models.video_data import VideoFile

logger = logging.getLogger(__name__)


class FileHandler:
    """Handles file operations for videos and subtitles."""

    SUPPORTED_VIDEO_FORMATS = {
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".wmv",
        ".flv",
        ".webm",
        ".m4v",
        ".mpg",
        ".mpeg",
    }

    SUPPORTED_AUDIO_FORMATS = {".wav", ".mp3", ".aac", ".flac", ".ogg", ".m4a"}

    def __init__(self, temp_dir: Optional[Path] = None):
        """Initialize file handler with optional temp directory."""
        self.temp_dir = Path(temp_dir) if temp_dir else Path(tempfile.gettempdir())
        self.temp_dir.mkdir(exist_ok=True)

    def validate_video_file(self, file_path: Path) -> bool:
        """Validate if file is a supported video format."""
        if not file_path.exists():
            logger.error(f"File does not exist: {file_path}")
            return False

        if file_path.suffix.lower() not in self.SUPPORTED_VIDEO_FORMATS:
            logger.error(f"Unsupported video format: {file_path.suffix}")
            return False

        return True

    def create_video_file_object(self, file_path: Path) -> Optional[VideoFile]:
        """Create VideoFile object with basic metadata."""
        try:
            if not self.validate_video_file(file_path):
                return None

            stat = file_path.stat()

            return VideoFile(
                file_path=file_path, filename=file_path.name, file_size=stat.st_size
            )

        except Exception as e:
            logger.error(f"Error creating VideoFile object: {e}")
            return None

    def get_video_metadata(self, video_file: VideoFile) -> VideoFile:
        """Extract detailed metadata from video file using ffprobe."""
        try:
            import ffmpeg

            probe = ffmpeg.probe(str(video_file.file_path))

            # Extract video stream info
            video_stream = next(
                (
                    stream
                    for stream in probe["streams"]
                    if stream["codec_type"] == "video"
                ),
                None,
            )

            # Extract audio stream info
            audio_stream = next(
                (
                    stream
                    for stream in probe["streams"]
                    if stream["codec_type"] == "audio"
                ),
                None,
            )

            # Update video file object
            if video_stream:
                video_file.video_codec = video_stream.get("codec_name")
                video_file.resolution = (
                    int(video_stream.get("width", 0)),
                    int(video_stream.get("height", 0)),
                )
                video_file.fps = eval(video_stream.get("r_frame_rate", "0/1"))

            if audio_stream:
                video_file.audio_codec = audio_stream.get("codec_name")

            # Duration from format info
            format_info = probe.get("format", {})
            if "duration" in format_info:
                video_file.duration = float(format_info["duration"])

            video_file.metadata = probe

        except ImportError:
            logger.warning(
                "ffmpeg-python not available, skipping detailed metadata extraction"
            )
        except Exception as e:
            logger.error(f"Error extracting video metadata: {e}")

        return video_file

    def create_temp_file(self, suffix: str = "") -> Path:
        """Create a temporary file and return its path."""
        temp_file = tempfile.NamedTemporaryFile(
            suffix=suffix, dir=self.temp_dir, delete=False
        )
        temp_file.close()
        return Path(temp_file.name)

    def cleanup_temp_files(self, file_paths: List[Path]):
        """Clean up temporary files."""
        for file_path in file_paths:
            try:
                if file_path.exists():
                    file_path.unlink()
                    logger.debug(f"Cleaned up temp file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to cleanup temp file {file_path}: {e}")

    def save_subtitle_file(
        self, content: str, output_path: Path, format_type: str = "srt"
    ) -> bool:
        """Save subtitle content to file."""
        try:
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Add extension if not present
            if not output_path.suffix:
                output_path = output_path.with_suffix(f".{format_type}")

            # Write content
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(content)

            logger.info(f"Subtitle file saved: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving subtitle file: {e}")
            return False

    def get_available_space(self, path: Path) -> int:
        """Get available disk space in bytes."""
        try:
            stat = shutil.disk_usage(path)
            return stat.free
        except Exception as e:
            logger.error(f"Error checking disk space: {e}")
            return 0
