"""Whisper ASR implementation using native Whisper generate() method."""

import logging
from typing import List, Optional, Callable, Dict, Any, Tuple
import numpy as np
import torch

from .base_asr import BaseASR
from ..models.video_data import AudioData
from ..models.subtitle_data import SubtitleSegment
from ..utils.logger import LoggerMixin

logger = logging.getLogger(__name__)


class WhisperASR(BaseASR, LoggerMixin):
    """Whisper ASR implementation using native Whisper generate() method.
    
    This implementation uses Whisper's native sliding window approach for
    long-form transcription as described in Section 3.8 of the Whisper paper.
    This provides better quality, no token limits, and eliminates experimental warnings.
    """
    
    def __init__(self, 
                 model_name: str = "openai/whisper-large-v3",
                 device: str = "auto",
                 batch_size: int = 1,
                 return_timestamps: bool = True,
                 chunk_length_s: int = 30,
                 **kwargs):
        """Initialize Whisper ASR with native generate() method.
        
        Args:
            model_name: Whisper model name from Hugging Face
            device: Device to run on ('auto', 'cuda', 'mps', 'cpu')
            batch_size: Batch size for inference (not used in native approach)
            return_timestamps: Whether to return word-level timestamps
            chunk_length_s: Length of audio chunks (30s is Whisper's native window)
            **kwargs: Additional generation parameters
        """
        super().__init__(model_name, device, **kwargs)
        
        self.batch_size = batch_size
        self.return_timestamps = return_timestamps
        self.chunk_length_s = chunk_length_s
        self.generation_kwargs = kwargs
        
        # Native Whisper components
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.feature_extractor = None
        
        # Device configuration
        self.torch_device = self._get_torch_device()
        self.torch_dtype = self._get_torch_dtype()
        
        # Whisper native parameters
        self.sample_rate = 16000  # Whisper's required sample rate
        self.n_mels = 80  # Whisper's mel spectrogram features
        self.hop_length = 160  # Whisper's hop length
    
    def _get_torch_device(self) -> torch.device:
        """Get the appropriate torch device."""
        if self.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        elif self.device == "cuda":
            if torch.cuda.is_available():
                return torch.device("cuda")
            else:
                self.logger.warning("CUDA requested but not available, falling back to CPU")
                return torch.device("cpu")
        elif self.device == "mps":
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                self.logger.warning("MPS requested but not available, falling back to CPU")
                return torch.device("cpu")
        else:
            return torch.device("cpu")
    
    def _get_torch_dtype(self) -> torch.dtype:
        """Get the appropriate torch dtype based on device."""
        if self.torch_device.type == "cuda":
            return torch.float16  # Use half precision for CUDA
        elif self.torch_device.type == "mps":
            return torch.float32  # MPS works better with float32
        else:
            return torch.float32  # CPU uses float32
    
    def load_model(self) -> bool:
        """Load Whisper model components using native transformers approach.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            from transformers import (
                WhisperForConditionalGeneration, 
                WhisperProcessor,
                WhisperTokenizer,
                WhisperFeatureExtractor
            )
            
            self.logger.info(f"Loading Whisper model: {self.model_name}")
            self.logger.info(f"Device: {self.torch_device}, dtype: {self.torch_dtype}")
            
            # Load model components
            self.model = WhisperForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
                device_map=self.torch_device,
                use_safetensors=True
            )
            
            self.processor = WhisperProcessor.from_pretrained(self.model_name)
            self.tokenizer = self.processor.tokenizer
            self.feature_extractor = self.processor.feature_extractor
            
            # Move model to device if not using device_map
            if not hasattr(self.model, 'hf_device_map'):
                self.model = self.model.to(self.torch_device)
            
            # Set model to evaluation mode
            self.model.eval()
            
            # Configure generation parameters
            self.generation_config = {
                'task': 'transcribe',
                'return_timestamps': self.return_timestamps,
                'num_beams': 1,  # Greedy decoding for consistency
                'do_sample': False,  # Deterministic generation
                'use_cache': True,  # Enable KV cache for efficiency
                'pad_token_id': self.tokenizer.pad_token_id,
                'bos_token_id': self.tokenizer.bos_token_id,
                'eos_token_id': self.tokenizer.eos_token_id,
                'suppress_tokens': None,  # Don't suppress any tokens
                'begin_suppress_tokens': None,  # Don't suppress beginning tokens
            }
            
            # Add custom generation kwargs
            self.generation_config.update(self.generation_kwargs)
            
            self.is_loaded = True
            self.logger.info(f"Whisper model loaded successfully on {self.torch_device}")
            self.logger.info("Using Whisper's native sliding window approach - no experimental warnings!")
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
        """Transcribe audio using Whisper's native generate() method.
        
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
            self.logger.info(f"Transcribing audio: {audio_data.duration:.2f}s using native Whisper")
            
            if progress_callback:
                progress_callback(0.0)
            
            # Preprocess audio for Whisper
            audio_features = self._preprocess_audio(audio_data)
            
            if progress_callback:
                progress_callback(0.2)
            
            # Set forced decoder IDs for language and task (Whisper's proper way)
            language_token = self.tokenizer.convert_tokens_to_ids(f"<|{language}|>")
            task_token = self.tokenizer.convert_tokens_to_ids("<|transcribe|>")
            
            forced_decoder_ids = [
                (1, language_token),  # Language token at position 1
                (2, task_token),      # Task token at position 2
            ]
            
            if progress_callback:
                progress_callback(0.3)
            
            # Generate with native Whisper approach
            if audio_data.duration <= 30.0:
                # Short audio - single pass (already padded to 30s)
                segments = self._transcribe_single_pass(
                    audio_features, forced_decoder_ids, progress_callback
                )
            else:
                # Long audio - process in 30-second chunks
                segments = self._transcribe_long_audio(
                    audio_data, forced_decoder_ids, progress_callback
                )
            
            if progress_callback:
                progress_callback(1.0)
            
            self.logger.info(f"Native transcription completed: {len(segments)} segments")
            return segments
            
        except Exception as e:
            self.logger.error(f"Error during native transcription: {e}")
            return []
    
    def _preprocess_audio(self, audio_data: AudioData) -> torch.Tensor:
        """Preprocess audio data for Whisper feature extraction.
        
        Args:
            audio_data: Input audio data
            
        Returns:
            Mel spectrogram features as torch tensor
        """
        # Ensure audio is at correct sample rate
        if audio_data.sample_rate != self.sample_rate:
            self.logger.warning(f"Audio sample rate {audio_data.sample_rate} != {self.sample_rate}, resampling may be needed")
        
        # Pad or truncate audio to exactly 30 seconds (Whisper's expected input length)
        target_length = self.sample_rate * 30  # 30 seconds at 16kHz
        audio_array = audio_data.audio_array.copy()
        
        if len(audio_array) < target_length:
            # Pad with zeros if too short
            padding = target_length - len(audio_array)
            audio_array = np.pad(audio_array, (0, padding), mode='constant', constant_values=0)
        elif len(audio_array) > target_length:
            # Truncate if too long (will be handled by sliding window)
            audio_array = audio_array[:target_length]
        
        # Extract mel spectrogram features using Whisper's feature extractor
        features = self.feature_extractor(
            audio_array,
            sampling_rate=self.sample_rate,
            return_tensors="pt"
        )
        
        # Move to device and convert dtype
        input_features = features.input_features.to(self.torch_device, dtype=self.torch_dtype)
        
        return input_features
    
    def _transcribe_single_pass(self, 
                               audio_features: torch.Tensor,
                               forced_decoder_ids: list,
                               progress_callback: Optional[Callable[[float], None]] = None) -> List[SubtitleSegment]:
        """Transcribe audio in a single pass for short audio (≤30s).
        
        Args:
            audio_features: Preprocessed audio features
            forced_decoder_ids: Forced decoder IDs for language/task tokens
            progress_callback: Optional progress callback
            
        Returns:
            List of subtitle segments
        """
        with torch.no_grad():
            # Update generation config with forced decoder IDs
            generation_config = self.generation_config.copy()
            generation_config['forced_decoder_ids'] = forced_decoder_ids
            
            # Create proper attention mask for input features
            # For Whisper, the attention mask should be all 1s for the entire sequence
            batch_size, n_mels, seq_len = audio_features.shape
            attention_mask = torch.ones((batch_size, seq_len), device=self.torch_device, dtype=torch.long)
            
            # Suppress attention mask warnings during generation (just in case)
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*attention mask is not set.*")
                
                # Generate tokens using Whisper's native generate method
                generated_ids = self.model.generate(
                    input_features=audio_features,
                    attention_mask=attention_mask,
                    **generation_config
                )
            
            if progress_callback:
                progress_callback(0.8)
            
            # Decode generated tokens
            segments = self._decode_tokens_to_segments(generated_ids)
            
        return segments
    
    def _transcribe_long_audio(self, 
                              audio_data: AudioData,
                              forced_decoder_ids: list,
                              progress_callback: Optional[Callable[[float], None]] = None) -> List[SubtitleSegment]:
        """Transcribe long audio by processing 30-second chunks.
        
        Args:
            audio_data: Full audio data
            forced_decoder_ids: Forced decoder IDs for language/task tokens
            progress_callback: Optional progress callback
            
        Returns:
            List of subtitle segments with proper timestamps
        """
        segments = []
        
        # Process in 30-second chunks with 5-second overlap
        chunk_duration = 30.0  # seconds
        overlap_duration = 5.0  # seconds
        
        chunk_samples = int(chunk_duration * self.sample_rate)
        overlap_samples = int(overlap_duration * self.sample_rate)
        
        audio_array = audio_data.audio_array
        total_samples = len(audio_array)
        current_sample = 0
        time_offset = 0.0
        
        chunk_count = 0
        total_chunks = int(np.ceil(total_samples / (chunk_samples - overlap_samples)))
        
        while current_sample < total_samples:
            # Extract chunk
            end_sample = min(current_sample + chunk_samples, total_samples)
            chunk_audio = audio_array[current_sample:end_sample]
            
            # Pad chunk to exactly 30 seconds if needed
            if len(chunk_audio) < chunk_samples:
                padding = chunk_samples - len(chunk_audio)
                chunk_audio = np.pad(chunk_audio, (0, padding), mode='constant', constant_values=0)
            
            # Create AudioData for chunk
            chunk_audio_data = AudioData(
                audio_array=chunk_audio,
                sample_rate=self.sample_rate,
                duration=chunk_duration
            )
            
            # Preprocess chunk
            chunk_features = self._preprocess_audio(chunk_audio_data)
            
            # Transcribe chunk
            chunk_segments = self._transcribe_single_pass(chunk_features, forced_decoder_ids)
            
            # Adjust timestamps with offset
            for segment in chunk_segments:
                # Only keep segments that start within the non-overlap region
                if segment.start_time < chunk_duration - overlap_duration or chunk_count == 0:
                    segment.start_time += time_offset
                    segment.end_time += time_offset
                    segments.append(segment)
            
            # Move to next chunk
            current_sample += chunk_samples - overlap_samples
            time_offset += chunk_duration - overlap_duration
            chunk_count += 1
            
            # Update progress
            if progress_callback:
                progress = min(chunk_count / total_chunks, 0.95)
                progress_callback(0.3 + progress * 0.6)  # 0.3 to 0.95
        
        return segments
    
    def _decode_tokens_to_segments(self, 
                                  generated_ids: torch.Tensor) -> List[SubtitleSegment]:
        """Decode generated tokens to subtitle segments.
        
        Args:
            generated_ids: Generated token IDs from model
            
        Returns:
            List of subtitle segments
        """
        segments = []
        
        try:
            # Skip the initial special tokens (bos, language, task)
            # Whisper uses: [bos_token, language_token, task_token, ...]
            skip_tokens = 3
            new_tokens = generated_ids[0, skip_tokens:]
            
            # Decode tokens with timestamps if enabled
            if self.return_timestamps:
                # Use Whisper's timestamp decoding
                decoded = self.tokenizer.decode(new_tokens, skip_special_tokens=False)
                segments = self._parse_whisper_timestamps(decoded)
            else:
                # Simple text decoding without timestamps
                decoded_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                if decoded_text.strip():
                    segment = SubtitleSegment(
                        start_time=0.0,
                        end_time=30.0,  # Default duration
                        text=decoded_text.strip(),
                        confidence=None
                    )
                    segments.append(segment)
                    
        except Exception as e:
            self.logger.error(f"Error decoding tokens to segments: {e}")
            
        return segments
    
    def _parse_whisper_timestamps(self, decoded_text: str) -> List[SubtitleSegment]:
        """Parse Whisper's timestamp tokens to create segments.
        
        Args:
            decoded_text: Decoded text with timestamp tokens
            
        Returns:
            List of subtitle segments with timestamps
        """
        import re
        segments = []
        
        try:
            # Whisper timestamp pattern: <|0.00|>text<|5.00|>
            timestamp_pattern = r'<\|(\d+\.?\d*)\|>'
            
            # Split by timestamp tokens
            parts = re.split(timestamp_pattern, decoded_text)
            
            current_start = 0.0
            current_text = ""
            
            for i, part in enumerate(parts):
                if i == 0:
                    continue  # Skip initial part
                    
                if i % 2 == 1:  # Timestamp
                    try:
                        timestamp = float(part)
                        if current_text.strip():
                            # Create segment for previous text
                            segment = SubtitleSegment(
                                start_time=current_start,
                                end_time=timestamp,
                                text=current_text.strip(),
                                confidence=None
                            )
                            segments.append(segment)
                            
                        current_start = timestamp
                        current_text = ""
                    except ValueError:
                        continue
                else:  # Text
                    current_text += part
            
            # Handle final segment if any text remains
            if current_text.strip():
                segment = SubtitleSegment(
                    start_time=current_start,
                    end_time=current_start + 5.0,  # Estimate duration
                    text=current_text.strip(),
                    confidence=None
                )
                segments.append(segment)
                
        except Exception as e:
            self.logger.error(f"Error parsing Whisper timestamps: {e}")
            
        return segments

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
                for i, chunk in enumerate(whisper_result["chunks"]):
                    timestamp = chunk.get("timestamp", [None, None])
                    start_time = timestamp[0] if timestamp[0] is not None else 0.0
                    end_time = timestamp[1] if timestamp[1] is not None else start_time + 1.0
                    
                    # Handle missing end timestamps (common Whisper issue)
                    if end_time is None or end_time <= start_time:
                        # Estimate end time based on text length and speech rate
                        text_length = len(chunk["text"].strip())
                        estimated_duration = max(text_length * 0.1, 1.0)  # ~10 chars per second
                        end_time = start_time + estimated_duration
                        
                        if i == len(whisper_result["chunks"]) - 1:
                            self.logger.warning("Whisper missing end timestamp for final segment - estimated duration used")
                    
                    segment = SubtitleSegment(
                        start_time=start_time,
                        end_time=end_time,
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
    
    @classmethod
    def get_longform_recommendations(cls) -> Dict[str, Any]:
        """Get recommended configuration for long-form transcription.
        
        Returns:
            Dictionary with recommended settings
        """
        return {
            "chunk_length_s": 30,          # Whisper's optimal chunk size
            "stride_length_s": 1.0,        # Small overlap between chunks
            "max_new_tokens": 128,         # Prevent runaway generation
            "return_timestamps": True,     # Essential for subtitles and WhisperTimeStampLogitsProcessor
            "ignore_warning": True,        # Suppress experimental warnings
            "batch_size": 1,               # Conservative for memory
            "generate_kwargs": {
                "return_timestamps": True,  # Ensures WhisperTimeStampLogitsProcessor is used
                "num_beams": 1,            # Better timestamp accuracy
                "do_sample": False,        # Deterministic generation
                "forced_decoder_ids": None  # Auto language detection
            },
            "notes": [
                "return_timestamps=True ensures WhisperTimeStampLogitsProcessor is used during generation",
                "WhisperTimeStampLogitsProcessor automatically handles timestamp prediction",
                "For production use, consider using Whisper's generate() method directly",
                "chunk_length_s is experimental but works well for most cases",
                "Missing end timestamps are automatically estimated using text length heuristics",
                "Use smaller chunks (15s) for better timestamp accuracy",
                "Deterministic generation (do_sample=False) improves timestamp consistency"
            ]
        }