"""Hugging Face transformer-based translation implementation."""

import logging
from typing import Callable, Dict, List, Optional

from ..models.subtitle_data import TranslationResult
from ..utils.logger import LoggerMixin
from .base_translator import BaseTranslator

logger = logging.getLogger(__name__)


class HuggingFaceTranslator(BaseTranslator, LoggerMixin):
    """Translation using Hugging Face transformers pipeline."""

    def __init__(
        self,
        model_name: str,
        source_lang: str,
        target_lang: str,
        device: str = "auto",
        batch_size: int = 8,
        max_length: int = 512,
        **kwargs,
    ):
        """Initialize Hugging Face translator.

        Args:
            model_name: Model name from Hugging Face Hub
            source_lang: Source language code
            target_lang: Target language code
            device: Device to run on
            batch_size: Batch size for translation
            max_length: Maximum sequence length
            **kwargs: Additional pipeline parameters
        """
        super().__init__(model_name, source_lang, target_lang, device, **kwargs)

        self.batch_size = batch_size
        self.max_length = max_length
        self.pipeline_kwargs = kwargs

        # Check if this is an NLLB model
        self.is_nllb = "nllb" in model_name.lower()

        # Map standard language codes to NLLB codes if needed
        if self.is_nllb:
            self.nllb_source_lang = self._get_nllb_lang_code(source_lang, is_source=True)
            self.nllb_target_lang = self._get_nllb_lang_code(target_lang, is_source=False)
            self.logger.info(f"NLLB detected: {source_lang}({self.nllb_source_lang}) -> {target_lang}({self.nllb_target_lang})")

        # Pipeline will be created in load_model
        self.pipeline = None

    def _get_nllb_lang_code(self, lang_code: str, is_source: bool = True) -> str:
        """Map standard language codes to NLLB language codes.
        
        Args:
            lang_code: Standard language code (e.g., 'en', 'ja', 'zh')
            is_source: Whether this is for source language
            
        Returns:
            NLLB language code
        """
        # NLLB language code mapping
        nllb_codes = {
            'en': 'eng_Latn',
            'ja': 'jpn_Jpan',
            'zh': 'zho_Hant',  # Default to Traditional Chinese
            'zh-Hans': 'zho_Hans',  # Simplified Chinese
            'zh-Hant': 'zho_Hant',  # Traditional Chinese
        }

        # Check for Traditional Chinese preference in config
        if lang_code == 'zh' and hasattr(self, '_config'):
            zh_variant = getattr(self._config, 'get_translation_config', lambda: {})().get('zh_target_variant', 'zho_Hant')
            return zh_variant

        return nllb_codes.get(lang_code, lang_code)

    def load_model(self) -> bool:
        """Load translation model using transformers pipeline.

        Returns:
            True if successful, False otherwise
        """
        try:
            import torch
            from transformers import pipeline

            self.logger.info(f"Loading translation model: {self.model_name}")
            self.logger.info(f"Translation: {self.source_lang} -> {self.target_lang}")

            # Determine torch_dtype and device for pipeline
            if self.device == "cuda" or self.device.startswith("cuda"):
                pipeline_device = 0  # Use first CUDA device
                torch_dtype = torch.float16
            elif self.device == "mps":
                pipeline_device = "mps"  # Use MPS device directly
                torch_dtype = torch.float32  # MPS works better with float32
            else:
                pipeline_device = "cpu"  # Explicitly use "cpu" string instead of -1
                torch_dtype = torch.float32

            # Create pipeline with explicit device specification to avoid meta device
            # Use model_kwargs to control device mapping more directly
            model_kwargs = {
                "torch_dtype": torch_dtype,
                "device_map": None,  # Explicitly disable auto device mapping
                "low_cpu_mem_usage": False,  # Disable to avoid meta tensor issues
            }

            self.pipeline = pipeline(
                "translation",
                model=self.model_name,
                device=pipeline_device,
                max_length=self.max_length,
                model_kwargs=model_kwargs,
                **self.pipeline_kwargs,
            )

            self.is_loaded = True
            self.logger.info(f"Translation model loaded successfully on {self.device}")
            return True

        except ImportError as e:
            self.logger.error(f"Required dependencies not installed: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Error loading translation model: {e}")
            return False

    def translate_text(
        self, text: str, progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> TranslationResult:
        """Translate a single text string.

        Args:
            text: Text to translate
            progress_callback: Optional progress callback

        Returns:
            Translation result
        """
        if not self.is_loaded:
            if not self.load_model():
                return TranslationResult(
                    original_text=text,
                    translated_text=text,  # Return original if translation fails
                    source_language=self.source_lang,
                    target_language=self.target_lang,
                    confidence=0.0,
                    translation_model=self.model_name,
                )

        try:
            if progress_callback:
                progress_callback("translation", 0.0)

            # Preprocess text
            processed_text = self._preprocess_text(text)

            if not processed_text.strip():
                return TranslationResult(
                    original_text=text,
                    translated_text="",
                    source_language=self.source_lang,
                    target_language=self.target_lang,
                    confidence=1.0,
                    translation_model=self.model_name,
                )

            if progress_callback:
                progress_callback("translation", 0.2)

            # Translate
            if self.is_nllb:
                result = self.pipeline(processed_text, src_lang=self.nllb_source_lang, tgt_lang=self.nllb_target_lang)
            else:
                result = self.pipeline(processed_text)

            if progress_callback:
                progress_callback("translation", 0.8)

            # Extract translated text
            if isinstance(result, list) and len(result) > 0:
                translated_text = result[0].get("translation_text", processed_text)
            elif isinstance(result, dict):
                translated_text = result.get("translation_text", processed_text)
            else:
                translated_text = str(result)

            # Postprocess
            translated_text = self._postprocess_text(translated_text)

            if progress_callback:
                progress_callback("translation", 1.0)

            return TranslationResult(
                original_text=text,
                translated_text=translated_text,
                source_language=self.source_lang,
                target_language=self.target_lang,
                confidence=None,  # HF models don't typically return confidence
                translation_model=self.model_name,
            )

        except Exception as e:
            self.logger.error(f"Error translating text: {e}")
            return TranslationResult(
                original_text=text,
                translated_text=text,  # Return original on error
                source_language=self.source_lang,
                target_language=self.target_lang,
                confidence=0.0,
                translation_model=self.model_name,
            )

    def translate_batch(
        self,
        texts: List[str],
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> List[TranslationResult]:
        """Translate multiple texts in batch.

        Args:
            texts: List of texts to translate
            progress_callback: Optional progress callback

        Returns:
            List of translation results
        """
        if not self.is_loaded:
            if not self.load_model():
                return [
                    TranslationResult(
                        original_text=text,
                        translated_text=text,
                        source_language=self.source_lang,
                        target_language=self.target_lang,
                        confidence=0.0,
                        translation_model=self.model_name,
                    )
                    for text in texts
                ]

        try:
            results = []
            total_texts = len(texts)

            # Process in batches
            for i in range(0, total_texts, self.batch_size):
                batch_texts = texts[i : i + self.batch_size]

                # Preprocess batch
                processed_texts = [self._preprocess_text(text) for text in batch_texts]

                # Filter out empty texts
                non_empty_indices = []
                non_empty_texts = []
                for j, text in enumerate(processed_texts):
                    if text.strip():
                        non_empty_indices.append(j)
                        non_empty_texts.append(text)

                # Translate non-empty texts
                if non_empty_texts:
                    if self.is_nllb:
                        batch_results = self.pipeline(non_empty_texts, src_lang=self.nllb_source_lang, tgt_lang=self.nllb_target_lang)
                    else:
                        batch_results = self.pipeline(non_empty_texts)

                    # Ensure batch_results is a list
                    if not isinstance(batch_results, list):
                        batch_results = [batch_results]
                else:
                    batch_results = []

                # Process results for this batch
                batch_translation_results = []
                result_idx = 0

                for j, original_text in enumerate(batch_texts):
                    if j in non_empty_indices and result_idx < len(batch_results):
                        # Get translation from results
                        result = batch_results[result_idx]
                        if isinstance(result, dict):
                            translated_text = result.get(
                                "translation_text", original_text
                            )
                        else:
                            translated_text = str(result)

                        translated_text = self._postprocess_text(translated_text)
                        result_idx += 1
                    else:
                        # Empty or failed translation
                        translated_text = (
                            "" if not processed_texts[j].strip() else original_text
                        )

                    batch_translation_results.append(
                        TranslationResult(
                            original_text=original_text,
                            translated_text=translated_text,
                            source_language=self.source_lang,
                            target_language=self.target_lang,
                            confidence=None,
                            translation_model=self.model_name,
                        )
                    )

                results.extend(batch_translation_results)

                # Update progress
                if progress_callback:
                    progress = min((i + len(batch_texts)) / total_texts, 1.0)
                    progress_callback("translation", progress)

            self.logger.info(f"Batch translation completed: {len(results)} texts")
            return results

        except Exception as e:
            self.logger.error(f"Error in batch translation: {e}")
            # Return original texts on error
            return [
                TranslationResult(
                    original_text=text,
                    translated_text=text,
                    source_language=self.source_lang,
                    target_language=self.target_lang,
                    confidence=0.0,
                    translation_model=self.model_name,
                )
                for text in texts
            ]

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text before translation."""
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text.strip()

    def _postprocess_text(self, text: str) -> str:
        """Postprocess text after translation."""
        # Clean up translation output
        text = text.strip()
        # Remove any translation artifacts
        if text.startswith('[') and text.endswith(']'):
            text = text[1:-1]
        return text

    def unload_model(self):
        """Unload translation model and free memory."""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None

        self.is_loaded = False

        self.logger.info("Translation model unloaded")


class MultiStageTranslator:
    """Multi-stage translator for Japanese -> English -> Traditional Chinese."""

    def __init__(
        self,
        ja_en_model: str = "Helsinki-NLP/opus-mt-ja-en",
        en_zh_model: str = "Helsinki-NLP/opus-mt-en-zh",
        device: str = "auto",
        **kwargs,
    ):
        """Initialize multi-stage translator.

        Args:
            ja_en_model: Japanese to English model
            en_zh_model: English to Chinese model
            device: Device to run on
            **kwargs: Additional parameters for translators
        """
        self.ja_en_translator = HuggingFaceTranslator(
            model_name=ja_en_model,
            source_lang="ja",
            target_lang="en",
            device=device,
            **kwargs,
        )

        self.en_zh_translator = HuggingFaceTranslator(
            model_name=en_zh_model,
            source_lang="en",
            target_lang="zh",
            device=device,
            **kwargs,
        )

        self.logger = logging.getLogger(__name__)

    def load_models(self) -> bool:
        """Load both translation models.

        Returns:
            True if both models loaded successfully
        """
        ja_en_loaded = self.ja_en_translator.load_model()
        en_zh_loaded = self.en_zh_translator.load_model()

        return ja_en_loaded and en_zh_loaded

    def translate_to_both(
        self,
        texts: List[str],
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> Dict[str, List[TranslationResult]]:
        """Translate Japanese texts to both English and Chinese.

        Args:
            texts: List of Japanese texts
            progress_callback: Optional progress callback

        Returns:
            Dictionary with 'en' and 'zh' translation results
        """
        try:
            # Translate to English
            if progress_callback:
                progress_callback("translation", 0.0)

            en_results = self.ja_en_translator.translate_batch(texts, progress_callback=progress_callback)

            if progress_callback:
                progress_callback("translation", 0.5)

            # Extract English texts for further translation
            en_texts = [result.translated_text for result in en_results]

            # Translate English to Chinese
            zh_results = self.en_zh_translator.translate_batch(en_texts, progress_callback=progress_callback)

            if progress_callback:
                progress_callback("translation", 1.0)

            return {"en": en_results, "zh": zh_results}

        except Exception as e:
            self.logger.error(f"Error in multi-stage translation: {e}")
            return {"en": [], "zh": []}

    def unload_models(self):
        """Unload both translation models."""
        self.ja_en_translator.unload_model()
        self.en_zh_translator.unload_model()

    def __enter__(self):
        """Context manager entry."""
        self.load_models()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.unload_models()
