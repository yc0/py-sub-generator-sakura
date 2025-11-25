"""Whisper ASR implementation using native Whisper generate() method."""

from typing import Callable, List, Optional

import numpy as np
import re
import torch

from ..models.subtitle_data import SubtitleSegment
from ..models.video_data import AudioData
from ..utils.logger import LoggerMixin
from .base_asr import BaseASR
from transformers import WhisperTimeStampLogitsProcessor, pipeline

class WhisperASR(LoggerMixin):
    def unload_model(self):
        """No-op for compatibility. Does nothing."""
        pass

    def transcribe_batch(self, audio_chunks, language=None, progress_callback=None):
        """Transcribe a list of AudioData chunks, returning one segment per chunk (real or empty), preserving chunk count and order."""
        all_segments = []
        time_offset = 0.0
        total_chunks = len(audio_chunks)
        for idx, audio_data in enumerate(audio_chunks):
            chunk_start = getattr(audio_data, "start_time", time_offset)
            segments = self.transcribe_audio(audio_data, language=language, progress_callback=progress_callback)
            self.logger.debug(
                f"Chunk {idx}: start_time={chunk_start:.2f}s duration={audio_data.duration:.2f}s segments={len(segments)}"
            )
            # If speech detected, merge all text in this chunk into one segment
            if segments:
                merged_text = " ".join([seg.text for seg in segments if seg.text])
                if not merged_text.strip():
                    pass
                # Use the earliest start and latest end among all segments
                start = min(seg.start_time for seg in segments)
                end = max(seg.end_time for seg in segments)
                candidate = SubtitleSegment(
                    start_time=chunk_start + start,
                    end_time=chunk_start + end,
                    text=merged_text,
                    confidence=None,
                    speaker_id=None,
                )
                if merged_text.strip() and not self._is_duplicate_segment(
                    all_segments, candidate
                ):
                    all_segments.append(candidate)
            else:
                # No speech: insert empty segment for this chunk's window
                all_segments.append(
                    SubtitleSegment(
                        start_time=chunk_start,
                        end_time=chunk_start + audio_data.duration,
                        text="",
                        confidence=None,
                        speaker_id=None
                    )
                )
            time_offset = chunk_start + audio_data.duration
            if progress_callback:
                progress_callback("asr", (idx + 1) / total_chunks if total_chunks else 1.0)
        return all_segments
    def load_model(self):
        """No-op for compatibility. Pipeline loads on demand."""
        return True
    def transcribe_audio(self, audio_data, language=None, progress_callback=None):
        """Transcribe AudioData and return a list of SubtitleSegment for compatibility with old interface."""
        import tempfile
        import soundfile as sf
        # Write audio to a temporary wav file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            sf.write(tmp.name, audio_data.audio_array, audio_data.sample_rate)
            lang = language or self.language
            # Use pipeline to get segments with timestamps
            self.logger.debug(f"[WhisperASR] Transcribing audio of duration {audio_data.duration:.2f}s with language={lang}")
            # if progress_callback:
            #     progress_callback("asr", 0.0)
            generate_kwargs = {
                "language": lang,
                "return_timestamps": True,
                "logits_processor": [self.timestamp_logits_processor],
            }
            result = self.pipe(tmp.name, generate_kwargs=generate_kwargs)
            self.logger.debug(f"[WhisperASR] Pipeline output: {result}")
            # if progress_callback:
            #     progress_callback("asr", 1.0)
            segments = []
            # Try to extract timestamps from result['chunks'] if available
            if isinstance(result, dict) and "chunks" in result and result["chunks"]:
                for chunk in result["chunks"]:
                    # Newer transformers may provide 'timestamp' or 'timestamps' or 'offsets'
                    start = None
                    end = None
                    if "timestamp" in chunk and isinstance(chunk["timestamp"], list):
                        start, end = chunk["timestamp"]
                    elif "timestamps" in chunk and isinstance(chunk["timestamps"], list):
                        start, end = chunk["timestamps"]
                    elif "offsets" in chunk and isinstance(chunk["offsets"], dict):
                        start = chunk["offsets"].get("start", 0)
                        end = chunk["offsets"].get("end", 0)
                    else:
                        start = 0.0
                        end = float(getattr(audio_data, "duration", 0))
                    segments.append(SubtitleSegment(
                        start_time=start,
                        end_time=end,
                        text=chunk.get("text", ""),
                        confidence=chunk.get("confidence", None)
                    ))
            elif isinstance(result, dict) and "text" in result:
                # Fallback: one segment for the whole text
                segments.append(SubtitleSegment(
                    start_time=0.0,
                    end_time=float(getattr(audio_data, "duration", 0)),
                    text=result["text"],
                    confidence=None
                ))
            return segments

    def _normalize_text(self, text: str) -> str:
        """Normalize text for duplicate comparison."""
        if not text:
            return ""
        normalized = text.strip().lower()
        normalized = re.sub(r"[\.。,，!！?？\"]", "", normalized)
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized

    def _is_duplicate_segment(
        self, segments: List[SubtitleSegment], candidate: SubtitleSegment
    ) -> bool:
        """Detect if candidate is a duplicate of the last emitted segment."""
        if not segments or not candidate.text.strip():
            return False

        last = segments[-1]
        if not last.text.strip():
            return False

        overlap = min(last.end_time, candidate.end_time) - max(
            last.start_time, candidate.start_time
        )
        normalized_last = self._normalize_text(last.text)
        normalized_candidate = self._normalize_text(candidate.text)

        return (
            normalized_last == normalized_candidate
            and overlap > 0
            and abs(last.end_time - candidate.end_time) < 0.5
        )
    """Minimal Whisper ASR using Hugging Face pipeline with hardware acceleration support."""
    def __init__(
        self,
        model_name="openai/whisper-large-v3",
        device=None,
        language="ja",
        return_timestamps: bool = True,
        chunk_length_s: Optional[float] = None,
        **pipeline_kwargs,
    ):
        super().__init__()
        self.model_name = model_name
        self.language = language
        self.return_timestamps = return_timestamps
        self.chunk_length_s = chunk_length_s
        # device: 'cuda', 'mps', 'cpu', or None (auto)
        if device is None or device == "auto":
            import torch
            self.logger.debug(f"[WhisperASR] torch.cuda.is_available(): {torch.cuda.is_available()}")
            mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            self.logger.debug(f"[WhisperASR] torch.backends.mps.is_available(): {mps_available}")
            if torch.cuda.is_available():
                device = "cuda"
            elif mps_available:
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        self.logger.debug(f"[WhisperASR] Using model name: {self.model_name}, device: {self.device}")
        pipeline_args = {
            "model": self.model_name,
            "device": self.device,
            "return_timestamps": self.return_timestamps,
            "model_kwargs": {"torch_dtype": "auto"},
        }
        if self.chunk_length_s is not None:
            pipeline_args["chunk_length_s"] = self.chunk_length_s
        pipeline_args.update(pipeline_kwargs)

        self.pipe = pipeline("automatic-speech-recognition", **pipeline_args)

        forced_decoder_ids = getattr(
            self.pipe.model.generation_config, "forced_decoder_ids", None
        )
        begin_index = len(forced_decoder_ids) if forced_decoder_ids else 1
        self.timestamp_logits_processor = WhisperTimeStampLogitsProcessor(
            self.pipe.model.generation_config, begin_index
        )

    def transcribe(self, audio_path: str) -> str:
        """Transcribe an audio file and return the recognized text."""
        result = self.pipe(audio_path, generate_kwargs={"language": self.language})
        return result["text"] if isinstance(result, dict) and "text" in result else str(result)

    def transcribe_segments(self, audio_path: str) -> list:
        """Transcribe an audio file and return segments (if supported by the model)."""
        result = self.pipe(audio_path, generate_kwargs={"language": self.language, "return_timestamps": True})
        if isinstance(result, dict) and "chunks" in result:
            return result["chunks"]
        return []

class DeprecatedWhisperASR(BaseASR, LoggerMixin):
    """Whisper ASR implementation using native Whisper generate() method.

    This implementation uses Whisper's native sliding window approach for
    long-form transcription as described in Section 3.8 of the Whisper paper.
    This provides better quality, no token limits, and eliminates experimental warnings.
    """
    def __init__(
        self,
        model_name: str = "openai/whisper-large-v3",
        device: str = "auto",
        batch_size: int = 1,
        return_timestamps: bool = True,
        chunk_length_s: int = 30,
        **kwargs,
    ):
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
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        elif self.device == "cuda":
            if torch.cuda.is_available():
                return torch.device("cuda")
            else:
                self.logger.warning(
                    "CUDA requested but not available, falling back to CPU"
                )
                return torch.device("cpu")
        elif self.device == "mps":
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                self.logger.warning(
                    "MPS requested but not available, falling back to CPU"
                )
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
                WhisperFeatureExtractor,
                WhisperForConditionalGeneration,
                WhisperProcessor,
                WhisperTokenizer,
            )

            self.logger.info(f"Loading Whisper model: {self.model_name}")
            self.logger.info(f"Device: {self.torch_device}, dtype: {self.torch_dtype}")

            # Load model components
            self.model = WhisperForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
                device_map=self.torch_device,
                use_safetensors=True,
            )

            self.processor = WhisperProcessor.from_pretrained(self.model_name)
            self.tokenizer = self.processor.tokenizer
            self.feature_extractor = self.processor.feature_extractor

            # Move model to device if not using device_map
            if not hasattr(self.model, "hf_device_map"):
                self.model = self.model.to(self.torch_device)

            # Set model to evaluation mode
            self.model.eval()

            # Configure generation parameters for Whisper
            self.generation_config = {
                "task": "transcribe",
                "language": "ja",  # Set language explicitly
                "return_timestamps": self.return_timestamps,
                "num_beams": 1,  # Greedy decoding for consistency
                "do_sample": False,  # Deterministic generation
                "use_cache": True,  # Enable KV cache for efficiency
                "pad_token_id": self.tokenizer.pad_token_id,
                "bos_token_id": self.tokenizer.bos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "suppress_tokens": None,  # Don't suppress any tokens
                "begin_suppress_tokens": None,  # Don't suppress beginning tokens
            }

            # Add custom generation kwargs (filter out invalid ones)
            valid_kwargs = {}
            invalid_params = ['name', 'model_name', 'device', 'batch_size', 'chunk_length_s']
            for key, value in self.generation_kwargs.items():
                if key not in invalid_params:
                    valid_kwargs[key] = value

            self.generation_config.update(valid_kwargs)

            self.is_loaded = True
            self.logger.info(
                f"Whisper model loaded successfully on {self.torch_device}"
            )
            self.logger.info(
                "Using Whisper's native sliding window approach - no experimental warnings!"
            )
            return True

        except ImportError as e:
            self.logger.error(f"Required dependencies not installed: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Error loading Whisper model: {e}")
            return False

    def transcribe_audio(
        self,
        audio_data: AudioData,
        language: str = "ja",
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> List[SubtitleSegment]:
        self.logger.debug("[ASR] Entered transcribe_audio")
        """Transcribe audio using Whisper's native generate() method.

        Args:
            audio_data: Audio data to transcribe
            language: Source language code
            progress_callback: Optional progress callback

        Returns:
            List of subtitle segments with timestamps
        """
        self.logger.debug("[ASR] Entered transcribe_audio")
        if not self.is_loaded:
            if not self.load_model():
                return []

        try:
            self.logger.info(
                f"Transcribing audio: {audio_data.duration:.2f}s using native Whisper"
            )
            # Log first 10 samples for inspection
            if hasattr(audio_data, 'audio_array') and audio_data.audio_array is not None:
                self.logger.debug(f"[ASR] First 10 audio samples: {audio_data.audio_array[:10]}")
            else:
                self.logger.warning("[ASR] audio_data.audio_array is None!")

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
                (2, task_token),  # Task token at position 2
            ]
            self.logger.debug(f"[ASR] Forced decoder IDs: {forced_decoder_ids}")
            self.logger.debug(f"[ASR] Model loaded: {self.is_loaded}, Model device: {self.torch_device}, Model dtype: {self.torch_dtype}")
            self.logger.debug(f"[ASR] Generation config: {self.generation_kwargs}")

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


            self.logger.info(
                f"Native transcription completed: {len(segments)} segments"
            )

            if len(segments) == 0:
                self.logger.warning("No segments produced - this indicates the model generated no meaningful transcription")
                audio_array = getattr(audio_data, 'audio_array', None)
                shape = audio_array.shape if audio_array is not None else None
                self.logger.debug(f"[ASR] segments empty. audio_data: duration={audio_data.duration}, sample_rate={audio_data.sample_rate}, shape={shape}")
                # Log a sample of the input features for further diagnosis
                try:
                    audio_features = self._preprocess_audio(audio_data)
                    self.logger.debug(f"[ASR] Sample of input features: {audio_features.flatten()[:10]}")
                except Exception as e:
                    self.logger.error(f"[ASR] Could not log input features: {e}")
            # Defensive: check for ambiguous truth value in segments
            try:
                for idx, seg in enumerate(segments):
                    if isinstance(seg, np.ndarray):
                        self.logger.error(f"[ASR] Segment {idx} is a numpy array, which is invalid. Segment: {seg}")
                    # Avoid using seg in a boolean context
            except Exception as e:
                import traceback
                self.logger.error(f"[ASR] Exception while iterating segments: {e}\n{traceback.format_exc()}")
            # Diagnostic: log type and value of segments before returning
            self.logger.debug(f"[ASR] Returning segments of type: {type(segments)}, value: {repr(segments)}")
            return segments

        except Exception as e:
            import traceback
            # Diagnostic: log type and value of segments if defined
            if 'segments' in locals():
                self.logger.error(f"[ASR] Exception caught. segments type: {type(segments)}, value: {repr(segments)}")
            self.logger.error(f"Error during native transcription: {e}\nTraceback: {traceback.format_exc()}")
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
            self.logger.warning(
                f"Audio sample rate {audio_data.sample_rate} != {self.sample_rate}, resampling may be needed"
            )

        # Pad or truncate audio to exactly 30 seconds (Whisper's expected input length)
        target_length = self.sample_rate * 30  # 30 seconds at 16kHz
        audio_array = audio_data.audio_array.copy()

        if len(audio_array) < target_length:
            # Pad with zeros if too short
            padding = target_length - len(audio_array)
            audio_array = np.pad(
                audio_array, (0, padding), mode="constant", constant_values=0
            )
        elif len(audio_array) > target_length:
            # Truncate if too long (will be handled by sliding window)
            audio_array = audio_array[:target_length]

        # Extract mel spectrogram features using Whisper's feature extractor
        features = self.feature_extractor(
            audio_array, sampling_rate=self.sample_rate, return_tensors="pt"
        )

        # Move to device and convert dtype
        input_features = features.input_features.to(
            self.torch_device, dtype=self.torch_dtype
        )

        return input_features

    def _transcribe_single_pass(
    
        self,
        audio_features: torch.Tensor,
        forced_decoder_ids: list,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> List[SubtitleSegment]:
        self.logger.debug("[ASR] Entered _transcribe_single_pass")
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
            generation_config["forced_decoder_ids"] = forced_decoder_ids

            # Create proper attention mask for input features
            # For Whisper, the attention mask should be all 1s for the entire sequence
            batch_size, n_mels, seq_len = audio_features.shape
            attention_mask = torch.ones(
                (batch_size, seq_len), device=self.torch_device, dtype=torch.long
            )

            # Suppress attention mask warnings during generation (just in case)
            import warnings

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message=".*attention mask is not set.*"
                )

                # Generate tokens using Whisper's native generate method
                generated_ids = self.model.generate(
                    input_features=audio_features,
                    attention_mask=attention_mask,
                    **generation_config,
                )

            if progress_callback:
                progress_callback(0.8)

            # Decode generated tokens
            segments = self._decode_tokens_to_segments(generated_ids)

        return segments

    def _transcribe_long_audio(
        self,
        audio_data: AudioData,
        forced_decoder_ids: list,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> List[SubtitleSegment]:
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
                chunk_audio = np.pad(
                    chunk_audio, (0, padding), mode="constant", constant_values=0
                )

            # Create AudioData for chunk
            chunk_audio_data = AudioData(
                audio_array=chunk_audio,
                sample_rate=self.sample_rate,
                duration=chunk_duration,
            )

            # Preprocess chunk
            chunk_features = self._preprocess_audio(chunk_audio_data)

            # Transcribe chunk
            chunk_segments = self._transcribe_single_pass(
                chunk_features, forced_decoder_ids
            )

            # Adjust timestamps with offset
            for segment in chunk_segments:
                # Only keep segments that start within the non-overlap region
                if (
                    segment.start_time < chunk_duration - overlap_duration
                    or chunk_count == 0
                ):
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

    def _decode_tokens_to_segments(
        self, generated_ids: torch.Tensor
    ) -> List[SubtitleSegment]:
        self.logger.debug("[ASR] Entered _decode_tokens_to_segments")
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

            self.logger.debug(f"[ASR] Decoding tokens: {new_tokens.tolist()}")

            # Decode tokens with timestamps if enabled
            if self.return_timestamps:
                # Use Whisper's timestamp decoding
                decoded = self.tokenizer.decode(new_tokens, skip_special_tokens=False)
                self.logger.debug(f"[ASR] Decoded text (with timestamps): '{decoded}'")
                segments = self._parse_whisper_timestamps(decoded)
            else:
                # Simple text decoding without timestamps
                decoded_text = self.tokenizer.decode(
                    new_tokens, skip_special_tokens=True
                )
                self.logger.debug(f"[ASR] Decoded text (no timestamps): '{decoded_text}'")

                if decoded_text.strip():
                    segment = SubtitleSegment(
                        start_time=0.0,
                        end_time=30.0,  # Default duration
                        text=decoded_text.strip(),
                        confidence=None,
                    )
                    segments.append(segment)
                    self.logger.debug(f"[ASR] Created segment: {segment}")
                else:
                    self.logger.warning("Decoded text is empty or whitespace only")
            # Defensive: check for ambiguous truth value in segments
            try:
                for idx, seg in enumerate(segments):
                    if isinstance(seg, np.ndarray):
                        self.logger.error(f"[ASR] _decode_tokens_to_segments: Segment {idx} is a numpy array, which is invalid. Segment: {seg}")
            except Exception as e:
                import traceback
                self.logger.error(f"[ASR] Exception while iterating segments in _decode_tokens_to_segments: {e}\n{traceback.format_exc()}")
            self.logger.debug(f"[ASR] Returning segments from _decode_tokens_to_segments: {segments}")
        except Exception as e:
            import traceback
            self.logger.error(f"Error decoding tokens to segments: {e}\nTraceback: {traceback.format_exc()}")
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
            # Clean the decoded text (remove special tokens that aren't timestamps)
            cleaned_text = decoded_text.strip()

            # Remove common non-timestamp special tokens
            cleaned_text = re.sub(r'<\|startoftranscript\|>', '', cleaned_text)
            cleaned_text = re.sub(r'<\|endoftranscript\|>', '', cleaned_text)
            cleaned_text = re.sub(r'<\|notimestamps\|>', '', cleaned_text)
            cleaned_text = re.sub(r'<\|ja\|>', '', cleaned_text)  # Language token
            cleaned_text = cleaned_text.strip()

            # Whisper timestamp pattern: <|0.00|>text<|5.00|>
            timestamp_pattern = r"<\|(\d+\.?\d*)\|>"

            # Check if we have timestamp tokens
            timestamp_matches = re.findall(timestamp_pattern, cleaned_text)

            if not timestamp_matches:
                # No timestamp tokens found - create a single segment with the entire text
                if cleaned_text:
                    segment = SubtitleSegment(
                        start_time=0.0,
                        end_time=5.0,  # Default 5-second duration
                        text=cleaned_text,
                        confidence=None,
                    )
                    segments.append(segment)
                return segments

            # Split by timestamp tokens
            parts = re.split(timestamp_pattern, cleaned_text)

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
                                confidence=None,
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
                    confidence=None,
                )
                segments.append(segment)
            # Defensive: check for ambiguous truth value in segments
            try:
                for idx, seg in enumerate(segments):
                    if isinstance(seg, np.ndarray):
                        self.logger.error(f"[ASR] _parse_whisper_timestamps: Segment {idx} is a numpy array, which is invalid. Segment: {seg}")
            except Exception as e:
                import traceback
                self.logger.error(f"[ASR] Exception while iterating segments in _parse_whisper_timestamps: {e}\n{traceback.format_exc()}")
        except Exception as e:
            import traceback
            self.logger.error(f"Error parsing Whisper timestamps: {e}\nTraceback: {traceback.format_exc()}")
        return segments

    def transcribe_batch(
        self,
        audio_chunks: List[AudioData],
        language: str = "ja",
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> List[SubtitleSegment]:
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

            self.logger.info(
                f"Batch transcription completed: {len(all_segments)} segments"
            )
            return all_segments

        except Exception as e:
            self.logger.error(f"Error during batch transcription: {e}")
            return []

    def unload_model(self):
        """Unload Whisper model and free memory."""
        # Clean up native Whisper components
        if hasattr(self, 'model') and self.model is not None:
            del self.model
            self.model = None

        if hasattr(self, 'processor') and self.processor is not None:
            del self.processor
            self.processor = None

        super().unload_model()

        self.logger.info("Whisper model unloaded")
