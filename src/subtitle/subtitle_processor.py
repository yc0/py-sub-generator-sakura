"""Subtitle processing utilities for post-processing and optimization."""

import logging
import re
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path

from ..models.subtitle_data import SubtitleSegment, SubtitleFile, TranslationResult
from ..utils.logger import LoggerMixin

logger = logging.getLogger(__name__)


class SubtitleProcessor(LoggerMixin):
    """Utilities for post-processing and optimizing subtitle segments."""
    
    def __init__(self,
                 max_chars_per_line: int = 42,
                 max_lines_per_subtitle: int = 2,
                 min_duration: float = 1.0,
                 max_duration: float = 7.0,
                 merge_threshold: float = 0.5):
        """Initialize subtitle processor.
        
        Args:
            max_chars_per_line: Maximum characters per line
            max_lines_per_subtitle: Maximum lines per subtitle
            min_duration: Minimum subtitle duration in seconds
            max_duration: Maximum subtitle duration in seconds  
            merge_threshold: Threshold for merging adjacent segments (seconds)
        """
        self.max_chars_per_line = max_chars_per_line
        self.max_lines_per_subtitle = max_lines_per_subtitle
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.merge_threshold = merge_threshold
    
    def process_segments(self, segments: List[SubtitleSegment]) -> List[SubtitleSegment]:
        """Process and optimize subtitle segments.
        
        Args:
            segments: Raw subtitle segments
            
        Returns:
            Processed and optimized segments
        """
        try:
            # Step 1: Clean text content
            segments = self._clean_segments(segments)
            
            # Step 2: Merge very short adjacent segments
            segments = self._merge_short_segments(segments)
            
            # Step 3: Split overly long segments
            segments = self._split_long_segments(segments)
            
            # Step 4: Optimize timing
            segments = self._optimize_timing(segments)
            
            # Step 5: Format text for display
            segments = self._format_text_for_display(segments)
            
            self.logger.info(f"Processed {len(segments)} subtitle segments")
            return segments
            
        except Exception as e:
            self.logger.error(f"Error processing segments: {e}")
            return segments
    
    def _clean_segments(self, segments: List[SubtitleSegment]) -> List[SubtitleSegment]:
        """Clean text content in segments.
        
        Args:
            segments: Input segments
            
        Returns:
            Segments with cleaned text
        """
        cleaned_segments = []
        
        for segment in segments:
            text = self._clean_text(segment.text)
            
            # Skip empty or very short segments
            if len(text.strip()) < 2:
                continue
            
            cleaned_segment = SubtitleSegment(
                start_time=segment.start_time,
                end_time=segment.end_time,
                text=text,
                confidence=segment.confidence,
                speaker_id=segment.speaker_id
            )
            
            cleaned_segments.append(cleaned_segment)
        
        return cleaned_segments
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content.
        
        Args:
            text: Raw text
            
        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove common speech recognition artifacts
        artifacts = [
            r'\[音楽\]', r'\[Music\]', r'\[MUSIC\]',
            r'\[拍手\]', r'\[Applause\]', r'\[APPLAUSE\]',
            r'\[笑い\]', r'\[Laughter\]', r'\[LAUGHTER\]',
            r'♪+', r'♫+', r'♬+', r'♩+',
            r'\(音楽\)', r'\(Music\)', r'\(MUSIC\)',
            r'あー+', r'えー+', r'うー+',  # Japanese filler words
            r'um+', r'uh+', r'ah+',  # English filler words
        ]
        
        for pattern in artifacts:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Remove excessive punctuation
        text = re.sub(r'[。！？]{3,}', '。', text)
        text = re.sub(r'[.!?]{3,}', '.', text)
        text = re.sub(r'[,，]{2,}', '、', text)
        
        # Normalize Japanese punctuation
        text = text.replace('､', '、').replace('｡', '。')
        
        # Remove leading/trailing punctuation
        text = text.strip('、。,.!?')
        
        return text.strip()
    
    def _merge_short_segments(self, segments: List[SubtitleSegment]) -> List[SubtitleSegment]:
        """Merge adjacent short segments.
        
        Args:
            segments: Input segments
            
        Returns:
            Segments with short ones merged
        """
        if not segments:
            return segments
        
        merged_segments = []
        current_segment = segments[0]
        
        for next_segment in segments[1:]:
            current_duration = current_segment.end_time - current_segment.start_time
            next_duration = next_segment.end_time - next_segment.start_time
            gap = next_segment.start_time - current_segment.end_time
            
            # Check if we should merge
            should_merge = (
                (current_duration < self.min_duration or next_duration < self.min_duration) and
                gap <= self.merge_threshold and
                len(current_segment.text) + len(next_segment.text) < self.max_chars_per_line * self.max_lines_per_subtitle
            )
            
            if should_merge:
                # Merge segments
                merged_text = f"{current_segment.text} {next_segment.text}".strip()
                
                current_segment = SubtitleSegment(
                    start_time=current_segment.start_time,
                    end_time=next_segment.end_time,
                    text=merged_text,
                    confidence=min(current_segment.confidence or 1.0, next_segment.confidence or 1.0) if current_segment.confidence and next_segment.confidence else None,
                    speaker_id=current_segment.speaker_id
                )
            else:
                merged_segments.append(current_segment)
                current_segment = next_segment
        
        # Add the last segment
        merged_segments.append(current_segment)
        
        return merged_segments
    
    def _split_long_segments(self, segments: List[SubtitleSegment]) -> List[SubtitleSegment]:
        """Split overly long segments.
        
        Args:
            segments: Input segments
            
        Returns:
            Segments with long ones split
        """
        split_segments = []
        
        for segment in segments:
            duration = segment.end_time - segment.start_time
            text_length = len(segment.text)
            
            # Check if segment is too long
            if duration > self.max_duration or text_length > self.max_chars_per_line * self.max_lines_per_subtitle:
                # Split the segment
                sub_segments = self._split_segment(segment)
                split_segments.extend(sub_segments)
            else:
                split_segments.append(segment)
        
        return split_segments
    
    def _split_segment(self, segment: SubtitleSegment) -> List[SubtitleSegment]:
        """Split a long segment into smaller parts.
        
        Args:
            segment: Segment to split
            
        Returns:
            List of smaller segments
        """
        text = segment.text
        duration = segment.end_time - segment.start_time
        max_chars = self.max_chars_per_line * self.max_lines_per_subtitle
        
        # If text is not too long, just split by time
        if len(text) <= max_chars:
            mid_time = segment.start_time + duration / 2
            
            return [
                SubtitleSegment(
                    start_time=segment.start_time,
                    end_time=mid_time,
                    text=text,
                    confidence=segment.confidence,
                    speaker_id=segment.speaker_id
                )
            ]
        
        # Split by text and distribute time proportionally
        sub_segments = []
        
        # Try to split at sentence boundaries first
        sentences = re.split(r'[。！？.!?]', text)
        if len(sentences) > 1:
            # Split at sentence boundaries
            current_pos = 0
            for i, sentence in enumerate(sentences[:-1]):  # Last split is usually empty
                sentence = sentence.strip() + text[current_pos + len(sentence)]  # Add back punctuation
                if sentence.strip():
                    char_ratio = len(sentence) / len(text)
                    segment_duration = duration * char_ratio
                    
                    sub_segments.append(SubtitleSegment(
                        start_time=segment.start_time + current_pos / len(text) * duration,
                        end_time=segment.start_time + (current_pos + len(sentence)) / len(text) * duration,
                        text=sentence.strip(),
                        confidence=segment.confidence,
                        speaker_id=segment.speaker_id
                    ))
                
                current_pos += len(sentence)
        else:
            # Split at word boundaries
            words = text.split()
            words_per_segment = max(1, len(words) // 2)
            
            for i in range(0, len(words), words_per_segment):
                segment_words = words[i:i + words_per_segment]
                segment_text = ' '.join(segment_words)
                
                start_ratio = i / len(words)
                end_ratio = min((i + len(segment_words)) / len(words), 1.0)
                
                sub_segments.append(SubtitleSegment(
                    start_time=segment.start_time + start_ratio * duration,
                    end_time=segment.start_time + end_ratio * duration,
                    text=segment_text,
                    confidence=segment.confidence,
                    speaker_id=segment.speaker_id
                ))
        
        return sub_segments if sub_segments else [segment]
    
    def _optimize_timing(self, segments: List[SubtitleSegment]) -> List[SubtitleSegment]:
        """Optimize timing of segments.
        
        Args:
            segments: Input segments
            
        Returns:
            Segments with optimized timing
        """
        if not segments:
            return segments
        
        optimized_segments = []
        
        for i, segment in enumerate(segments):
            # Ensure minimum duration
            duration = segment.end_time - segment.start_time
            if duration < self.min_duration:
                # Extend end time
                new_end_time = segment.start_time + self.min_duration
                
                # Make sure we don't overlap with next segment
                if i < len(segments) - 1:
                    next_segment = segments[i + 1]
                    if new_end_time > next_segment.start_time:
                        new_end_time = next_segment.start_time - 0.1  # Leave small gap
                
                segment = SubtitleSegment(
                    start_time=segment.start_time,
                    end_time=max(new_end_time, segment.end_time),
                    text=segment.text,
                    confidence=segment.confidence,
                    speaker_id=segment.speaker_id
                )
            
            optimized_segments.append(segment)
        
        return optimized_segments
    
    def _format_text_for_display(self, segments: List[SubtitleSegment]) -> List[SubtitleSegment]:
        """Format text for optimal display.
        
        Args:
            segments: Input segments
            
        Returns:
            Segments with formatted text
        """
        formatted_segments = []
        
        for segment in segments:
            formatted_text = self._format_text_lines(segment.text)
            
            formatted_segment = SubtitleSegment(
                start_time=segment.start_time,
                end_time=segment.end_time,
                text=formatted_text,
                confidence=segment.confidence,
                speaker_id=segment.speaker_id
            )
            
            formatted_segments.append(formatted_segment)
        
        return formatted_segments
    
    def _format_text_lines(self, text: str) -> str:
        """Format text into appropriate lines for subtitle display.
        
        Args:
            text: Input text
            
        Returns:
            Formatted text with line breaks
        """
        if len(text) <= self.max_chars_per_line:
            return text
        
        # Try to break at natural points
        words = text.split()
        lines = []
        current_line = ""
        
        for word in words:
            test_line = f"{current_line} {word}".strip()
            
            if len(test_line) <= self.max_chars_per_line:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
                
                # If we already have max lines, break
                if len(lines) >= self.max_lines_per_subtitle:
                    break
        
        # Add remaining text
        if current_line:
            lines.append(current_line)
        
        # Limit to max lines
        lines = lines[:self.max_lines_per_subtitle]
        
        return '\n'.join(lines)
    
    def analyze_segments(self, segments: List[SubtitleSegment]) -> Dict[str, Any]:
        """Analyze subtitle segments and return statistics.
        
        Args:
            segments: Segments to analyze
            
        Returns:
            Analysis statistics
        """
        if not segments:
            return {}
        
        durations = [seg.end_time - seg.start_time for seg in segments]
        text_lengths = [len(seg.text) for seg in segments]
        
        total_duration = segments[-1].end_time - segments[0].start_time if segments else 0
        
        stats = {
            "total_segments": len(segments),
            "total_duration": total_duration,
            "avg_segment_duration": sum(durations) / len(durations),
            "min_segment_duration": min(durations),
            "max_segment_duration": max(durations),
            "avg_text_length": sum(text_lengths) / len(text_lengths),
            "min_text_length": min(text_lengths),
            "max_text_length": max(text_lengths),
            "segments_too_short": sum(1 for d in durations if d < self.min_duration),
            "segments_too_long": sum(1 for d in durations if d > self.max_duration),
            "segments_text_too_long": sum(1 for l in text_lengths if l > self.max_chars_per_line * self.max_lines_per_subtitle)
        }
        
        return stats