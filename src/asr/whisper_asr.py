"""Whisper ASR implementation using Hugging Face transformers."""

import logging
from typing import List, Optional, Callable, Dict, Any
import numpy as np

from .base_asr import BaseASR
from ..models.video_data import AudioData
from ..models.subtitle_data import SubtitleSegment
from ..utils.logger import LoggerMixin

logger = logging.getLogger(__name__)


class WhisperASR(BaseASR, LoggerMixin):
    """Whisper ASR implementation using transformers pipeline."""
    
    def __init__(self, 
                 model_name: str = "openai/whisper-large-v3",
                 device: str = "auto",
                 batch_size: int = 1,
                 return_timestamps: bool = True,
                 chunk_length_s: int = 30,
                 **kwargs):
        """Initialize Whisper ASR.
        
        Args:
            model_name: Whisper model name from Hugging Face
            device: Device to run on
            batch_size: Batch size for inference
            return_timestamps: Whether to return word-level timestamps
            chunk_length_s: Length of audio chunks in seconds
            **kwargs: Additional pipeline parameters
        """
        super().__init__(model_name, device, **kwargs)
        
        self.batch_size = batch_size
        self.return_timestamps = return_timestamps
        self.chunk_length_s = chunk_length_s
        self.pipeline_kwargs = kwargs
        
        # Pipeline will be created in load_model
        self.pipeline = None
    
    def load_model(self) -> bool:
        """Load Whisper model using transformers pipeline.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            from transformers import pipeline
            import torch
            
            self.logger.info(f"Loading Whisper model: {self.model_name}")
            
            # Determine torch_dtype based on device
            torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
            
            # Create pipeline
            self.pipeline = pipeline(
                "automatic-speech-recognition",
                model=self.model_name,
                device=0 if self.device == "cuda" else -1,
                torch_dtype=torch_dtype,
                return_timestamps=self.return_timestamps,
                chunk_length_s=self.chunk_length_s,
                **self.pipeline_kwargs
            )
            
            self.is_loaded = True
            self.logger.info(f"Whisper model loaded successfully on {self.device}")
            return True
            
        except ImportError as e:
            self.logger.error(f"Required dependencies not installed: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Error loading Whisper model: {e}")
            return False
    
    def transcribe_audio(self, 
                        audio_data: AudioData,
                        language: str = "ja",
                        progress_callback: Optional[Callable[[float], None]] = None) -> List[SubtitleSegment]:
        """Transcribe audio using Whisper.
        
        Args:
            audio_data: Audio data to transcribe
            language: Source language code
            progress_callback: Optional progress callback
            
        Returns:
            List of subtitle segments with timestamps
        """
        if not self.is_loaded:
            if not self.load_model():
                return []
        
        try:
            self.logger.info(f"Transcribing audio: {audio_data.duration:.2f}s")
            
            if progress_callback:
                progress_callback(0.0)
            
            # Prepare transcription parameters
            generate_kwargs = {
                "language": language,
                "task": "transcribe"
            }
            
            # Run transcription
            result = self.pipeline(
                audio_data.audio_array,
                generate_kwargs=generate_kwargs,
                batch_size=self.batch_size
            )
            
            if progress_callback:
                progress_callback(0.8)
            
            # Convert to SubtitleSegment objects
            segments = self._convert_to_segments(result)
            
            if progress_callback:
                progress_callback(1.0)
            
            self.logger.info(f"Transcription completed: {len(segments)} segments")
            return segments
            
        except Exception as e:
            self.logger.error(f"Error during transcription: {e}")
            return []
    
    def transcribe_batch(self,
                        audio_chunks: List[AudioData],
                        language: str = "ja",
                        progress_callback: Optional[Callable[[float], None]] = None) -> List[SubtitleSegment]:
        """Transcribe multiple audio chunks.
        
        Args:
            audio_chunks: List of audio chunks
            language: Source language code
            progress_callback: Optional progress callback
            
        Returns:
            Combined list of subtitle segments
        """
        if not self.is_loaded:
            if not self.load_model():
                return []
        
        all_segments = []
        time_offset = 0.0
        
        try:
            total_chunks = len(audio_chunks)
            
            for i, chunk in enumerate(audio_chunks):
                self.logger.info(f"Processing chunk {i+1}/{total_chunks}")
                
                # Transcribe chunk
                chunk_segments = self.transcribe_audio(chunk, language)
                
                # Adjust timestamps with offset
                for segment in chunk_segments:
                    segment.start_time += time_offset
                    segment.end_time += time_offset
                
                all_segments.extend(chunk_segments)
                
                # Update time offset (subtract overlap to avoid gaps)
                time_offset += chunk.duration - 1.0  # Assuming 1s overlap
                
                # Update progress
                if progress_callback:
                    progress = (i + 1) / total_chunks
                    progress_callback(progress)
            
            self.logger.info(f"Batch transcription completed: {len(all_segments)} segments")
            return all_segments
            
        except Exception as e:
            self.logger.error(f"Error during batch transcription: {e}")
            return []
    
    def _convert_to_segments(self, whisper_result: Dict[str, Any]) -> List[SubtitleSegment]:
        """Convert Whisper pipeline result to SubtitleSegment objects.
        
        Args:
            whisper_result: Result from Whisper pipeline
            
        Returns:
            List of SubtitleSegment objects
        """
        segments = []
        
        try:
            # Handle different result formats
            if "chunks" in whisper_result:
                # Chunked result with timestamps
                for chunk in whisper_result["chunks"]:
                    segment = SubtitleSegment(
                        start_time=chunk["timestamp"][0] or 0.0,
                        end_time=chunk["timestamp"][1] or 0.0,
                        text=chunk["text"].strip(),
                        confidence=None  # Whisper doesn't return confidence scores
                    )
                    if segment.text:  # Only add non-empty segments
                        segments.append(segment)
            
            elif "text" in whisper_result:
                # Simple text result without timestamps
                text = whisper_result["text"].strip()
                if text:
                    segment = SubtitleSegment(
                        start_time=0.0,
                        end_time=0.0,  # Will need to be estimated
                        text=text,
                        confidence=None
                    )
                    segments.append(segment)
            
            # Post-process segments
            segments = self._post_process_segments(segments)
            
        except Exception as e:
            self.logger.error(f"Error converting Whisper result: {e}")
        
        return segments
    
    def _post_process_segments(self, segments: List[SubtitleSegment]) -> List[SubtitleSegment]:
        """Post-process segments for better quality.
        
        Args:
            segments: Raw segments from ASR
            
        Returns:
            Post-processed segments
        """
        processed_segments = []
        
        for segment in segments:
            # Clean up text
            text = segment.text.strip()
            
            # Skip very short or empty segments
            if len(text) < 2:
                continue
            
            # Basic text cleaning
            text = self._clean_text(text)
            
            # Update segment
            processed_segment = SubtitleSegment(
                start_time=segment.start_time,
                end_time=segment.end_time,
                text=text,
                confidence=segment.confidence,
                speaker_id=segment.speaker_id
            )
            
            processed_segments.append(processed_segment)
        
        return processed_segments
    
    def _clean_text(self, text: str) -> str:
        """Clean transcribed text.
        
        Args:
            text: Raw transcribed text
            
        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = " ".join(text.split())
        
        # Remove common artifacts
        artifacts = [
            "♪", "♫", "♬", "♩",  # Music notes
            "[音楽]", "[Music]", "[MUSIC]",  # Music tags
            "[拍手]", "[Applause]", "[APPLAUSE]"  # Applause tags
        ]
        
        for artifact in artifacts:
            text = text.replace(artifact, "").strip()
        
        return text
    
    def unload_model(self):
        """Unload Whisper model and free memory."""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
        
        super().unload_model()
        
        self.logger.info("Whisper model unloaded")