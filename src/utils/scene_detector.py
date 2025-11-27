"""Scene detection helpers for segmenting videos."""

import logging
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from ..models.video_data import VideoFile

logger = logging.getLogger(__name__)


@dataclass
class SceneSegment:
    """Represents one scene burst with timing metadata."""

    start_time: float
    end_time: float
    score: float


class SceneDetector:
    """Detects scene cuts using FFmpeg's scene detection filter."""

    def __init__(
        self,
        enabled: bool = True,
        threshold: float = 0.35,
        min_scene_length: float = 1.0,
        max_scenes: int = 60,
        ffmpeg_path: str = "ffmpeg",
    ) -> None:
        self.enabled = enabled
        self.threshold = threshold
        self.min_scene_length = min_scene_length
        self.max_scenes = max_scenes
        self.ffmpeg_path = ffmpeg_path

    def detect_scenes(self, video_file: VideoFile) -> List[SceneSegment]:
        """Return a list of scene segments for the provided video."""
        duration = video_file.duration or self._probe_duration(video_file.file_path)
        if not duration or duration <= 0:
            duration = 0.0

        if not self.enabled:
            return [SceneSegment(0.0, duration, 0.0)]

        try:
            change_points = self._run_scene_detection(video_file.file_path)
        except Exception as exc:
            logger.warning("Scene detection failed, falling back to single segment: %s", exc)
            return [SceneSegment(0.0, duration, 0.0)]

        segments: List[SceneSegment] = []
        prev_time = 0.0
        for time_stamp, score in change_points:
            if time_stamp <= prev_time:
                continue
            if time_stamp - prev_time < self.min_scene_length:
                prev_time = time_stamp
                continue
            segments.append(SceneSegment(prev_time, time_stamp, score))
            prev_time = time_stamp
            if len(segments) >= self.max_scenes:
                break

        if duration > prev_time + self.min_scene_length:
            segments.append(SceneSegment(prev_time, duration, 0.0))
        elif not segments and duration > 0:
            segments.append(SceneSegment(0.0, duration, 0.0))

        logger.info(
            "Scene detector produced %d segments (video duration: %.2fs)",
            len(segments),
            duration,
        )
        return segments

    def _run_scene_detection(self, video_path: Path) -> List[tuple[float, float]]:
        """Invoke FFmpeg scene detection and parse change timestamps."""
        cmd = [
            self.ffmpeg_path,
            "-hide_banner",
            "-nostats",
            "-i",
            str(video_path),
            "-filter_complex",
            f"select='gt(scene,{self.threshold})',metadata=print",
            "-an",
            "-f",
            "null",
            "-",
        ]
        logger.debug("Running FFmpeg scene detection: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)

        stderr = result.stderr or ""
        time_pattern = re.compile(r"pts_time[:=]?(\d+(?:\.\d+)?)")
        score_pattern = re.compile(r"lavfi\.scene_score[:=]?(\d+(?:\.\d+)?)")

        change_points: List[Tuple[float, float]] = []
        last_score = 0.0
        for line in stderr.splitlines():
            time_match = time_pattern.search(line)
            score_match = score_pattern.search(line)
            if score_match:
                last_score = float(score_match.group(1))
            if time_match:
                time_value = float(time_match.group(1))
                change_points.append((time_value, last_score))
                last_score = 0.0

        if not change_points:
            logger.warning("No scene changes detected, returning single segment")

        return change_points[: self.max_scenes]

    def _probe_duration(self, video_path: Path) -> float:
        """Query FFprobe for video duration when metadata is missing."""
        try:
            cmd = [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return float(result.stdout.strip())
        except Exception as exc:
            logger.warning("Failed to probe duration via ffprobe: %s", exc)
            return 0.0
