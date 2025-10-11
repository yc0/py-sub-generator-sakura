"""Translation pipeline for coordinating multi-language translation."""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from src.translation.sakura_translator_llama_cpp import SakuraTranslator
from src.utils.chinese_converter import convert_to_traditional

from ..models.subtitle_data import SubtitleFile, SubtitleSegment, TranslationResult
from ..utils.config import Config
from ..utils.logger import LoggerMixin
from .huggingface_translator import MultiStageTranslator

logger = logging.getLogger(__name__)


class TranslationPipeline(LoggerMixin):
    """Coordinates translation workflow for subtitle generation."""

    def __init__(self, config: Config):
        """Initialize translation pipeline.

        Args:
            config: Application configuration
        """
        self.config = config
        self.translation_config = config.get_translation_config()
        self.sakura_config = config.get_sakura_config()

        # Initialize translators
        self.multi_stage_translator = None
        self.sakura_translator = None
        self._initialize_translators()

    def _initialize_translators(self):
        """Initialize translation models based on configuration."""
        try:
            # Check if SakuraLLM is enabled
            if self.config.is_sakura_enabled():
                self.logger.info("🌸 Initializing SakuraLLM translator")
                self.sakura_translator = SakuraTranslator.create_from_config(
                    self.config
                )
                self.logger.info(
                    "🌸 SakuraLLM translator initialized for Japanese→Chinese"
                )
            else:
                self.logger.info("Initializing standard multi-stage translator")
                self.multi_stage_translator = MultiStageTranslator(
                    ja_en_model=self.translation_config.get(
                        "ja_to_en_model", "Helsinki-NLP/opus-mt-ja-en"
                    ),
                    en_zh_model=self.translation_config.get(
                        "en_to_zh_model", "Helsinki-NLP/opus-mt-en-zh"
                    ),
                    device=self.translation_config.get("device", "auto"),
                    batch_size=self.translation_config.get("batch_size", 8),
                    max_length=self.translation_config.get("max_length", 512),
                )
                self.logger.info("Standard translation pipeline initialized")

        except Exception as e:
            self.logger.error(f"Error initializing translation pipeline: {e}")
            self.multi_stage_translator = None
            self.sakura_translator = None

    def load_models(self) -> bool:
        """Load all translation models.

        Returns:
            True if successful, False otherwise
        """
        if self.sakura_translator is None and self.multi_stage_translator is None:
            self._initialize_translators()

        # Load SakuraLLM if enabled
        if self.sakura_translator:
            self.logger.info("🌸 Loading SakuraLLM model...")
            return self.sakura_translator.load_model()

        # Load standard translator if SakuraLLM not enabled
        if self.multi_stage_translator:
            self.logger.info("Loading standard translation models...")
            return self.multi_stage_translator.load_models()

        return False

    def translate_subtitle_file(
        self,
        subtitle_file: SubtitleFile,
        target_languages: List[str] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> SubtitleFile:
        """Translate entire subtitle file to target languages.

        Args:
            subtitle_file: SubtitleFile to translate
            target_languages: List of target language codes ['en', 'zh']
            progress_callback: Callback for progress updates (stage, progress)

        Returns:
            Updated SubtitleFile with translations
        """
        if target_languages is None:
            target_languages = ["en", "zh"]  # Default to English and Chinese

        try:
            # Load models if not already loaded
            if not self.load_models():
                self.logger.error("Failed to load translation models")
                return subtitle_file

            # Extract texts from segments
            texts = [segment.text for segment in subtitle_file.segments]

            if not texts:
                self.logger.warning("No texts to translate")
                return subtitle_file

            self.logger.info(f"Translating {len(texts)} segments to {target_languages}")

            # Perform translation based on enabled translator
            if progress_callback:
                progress_callback("translation", 0.0)

            if self.sakura_translator:
                # Use enhanced SakuraLLM pipeline: ja → zh-Hans → zh-Hant
                self.logger.info("🌸 Using SakuraLLM pipeline: ja → zh-Hans → zh-Hant")
                translation_results = {}

                # Step 1: Japanese → Simplified Chinese (SakuraLLM)
                if "zh" in target_languages:
                    self.logger.info("Step 1: Japanese → Simplified Chinese (SakuraLLM)")
                    zh_hans_results = self.sakura_translator.translate_batch(
                        texts,
                        progress_callback=lambda p: (
                            progress_callback("translation", p * 0.7)  # 70% for SakuraLLM
                            if progress_callback
                            else None
                        ),
                    )

                    # Step 2: Simplified Chinese → Traditional Chinese (Character conversion)
                    self.logger.info("Step 2: Simplified Chinese → Traditional Chinese (OpenCC character conversion)")

                    zh_hant_results = []
                    for i, result in enumerate(zh_hans_results):
                        # Convert simplified to traditional using reliable OpenCC
                        traditional_text = convert_to_traditional(result.translated_text)

                        # Create new result with traditional text
                        zh_hant_result = TranslationResult(
                            original_text=result.original_text,
                            translated_text=traditional_text,
                            source_lang=result.source_lang,
                            target_lang="zh-Hant",  # Mark as Traditional Chinese
                            confidence=result.confidence,
                        )
                        zh_hant_results.append(zh_hant_result)

                        if progress_callback:
                            progress_callback("conversion", 0.7 + (i + 1) / len(zh_hans_results) * 0.3)

                    translation_results["zh"] = zh_hant_results
                    self.logger.info(f"✅ SakuraLLM pipeline completed: {len(zh_hant_results)} translations")

                # Handle English translation if requested
                if "en" in target_languages:
                    self.logger.info("Japanese→English: Using standard translator alongside SakuraLLM")
                    if self.multi_stage_translator and self.multi_stage_translator.load_models():
                        en_results = self.multi_stage_translator.translate_to_language(texts, "en")
                        translation_results["en"] = en_results
                    else:
                        self.logger.warning("Standard translator not available for English translation")
                        translation_results["en"] = []

                if not translation_results:
                    self.logger.warning("No target languages supported by SakuraLLM pipeline")
                else:
                    self.logger.warning(
                        "SakuraLLM only supports Japanese→Chinese. Skipping other languages."
                    )
                    translation_results = {}
            else:
                # Use standard multi-stage translator
                translation_results = self.multi_stage_translator.translate_to_both(
                    texts,
                    progress_callback=lambda p: (
                        progress_callback("translation", p)
                        if progress_callback
                        else None
                    ),
                )

            # Add translations to subtitle file
            for lang_code, results in translation_results.items():
                if lang_code in target_languages and results:
                    subtitle_file.add_translation(lang_code, results)
                    self.logger.info(
                        f"Added {len(results)} translations for {lang_code}"
                    )

            if progress_callback:
                progress_callback("translation", 1.0)

            return subtitle_file

        except Exception as e:
            self.logger.error(f"Error in translation pipeline: {e}")
            return subtitle_file

    def translate_segments_to_language(
        self,
        segments: List[SubtitleSegment],
        target_language: str,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> List[TranslationResult]:
        """Translate segments to a specific target language.

        Args:
            segments: List of subtitle segments
            target_language: Target language code
            progress_callback: Optional progress callback

        Returns:
            List of translation results
        """
        try:
            texts = [segment.text for segment in segments]

            # Use SakuraLLM for Chinese if enabled
            if target_language == "zh" and self.sakura_translator:
                self.logger.info("🌸 Using SakuraLLM for Japanese→Chinese translation")
                return self.sakura_translator.translate_batch(texts, progress_callback)

            # Use standard translator for other cases
            elif self.multi_stage_translator:
                if target_language == "en":
                    # Direct Japanese to English
                    if not self.multi_stage_translator.ja_en_translator.is_loaded:
                        if (
                            not self.multi_stage_translator.ja_en_translator.load_model()
                        ):
                            return []

                    return self.multi_stage_translator.ja_en_translator.translate_batch(
                        texts, progress_callback
                    )

                elif target_language == "zh":
                    # Two-stage: ja->en->zh (when SakuraLLM not enabled)
                    results = self.multi_stage_translator.translate_to_both(
                        texts, progress_callback
                    )
                    return results.get("zh", [])

                else:
                    self.logger.error(f"Unsupported target language: {target_language}")
                    return []

            else:
                self.logger.error("No translation models available")
                return []

        except Exception as e:
            self.logger.error(f"Error translating to {target_language}: {e}")
            return []

    def export_translated_subtitles(
        self, subtitle_file: SubtitleFile, output_dir: Path, formats: List[str] = None
    ) -> Dict[str, Path]:
        """Export translated subtitles to files.

        Args:
            subtitle_file: SubtitleFile with translations
            output_dir: Output directory
            formats: List of formats to export ['srt'] (more can be added later)

        Returns:
            Dictionary mapping language_format to file paths
        """
        if formats is None:
            formats = ["srt"]

        output_files = {}
        generate_both = self.config.get_generate_both_languages()
        original_suffix = self.config.get_original_language_suffix()
        translated_suffix = self.config.get_translated_language_suffix()

        try:
            output_dir.mkdir(parents=True, exist_ok=True)

            # Base filename (without extension)
            base_name = (
                subtitle_file.video_file.stem
                if subtitle_file.video_file
                else "subtitles"
            )

            # Export original (Japanese) if user wants both languages
            if generate_both:
                for fmt in formats:
                    if fmt == "srt":
                        original_content = subtitle_file.export_srt()
                        original_file = output_dir / f"{base_name}{original_suffix}.srt"

                        with open(original_file, "w", encoding="utf-8") as f:
                            f.write(original_content)

                        output_files[f"original_{fmt}"] = original_file
                        self.logger.info(f"Exported Japanese subtitles: {original_file}")

            # Export translations
            for lang_code, translations in subtitle_file.translations.items():
                if translations:
                    for fmt in formats:
                        if fmt == "srt":
                            translated_content = subtitle_file.export_srt(lang_code)

                            # Use custom suffix if generating both, otherwise use language code
                            if generate_both and lang_code in ["zh", "zh-cn", "zh-tw"]:
                                suffix = translated_suffix
                            else:
                                suffix = f"_{lang_code}"

                            translated_file = output_dir / f"{base_name}{suffix}.srt"

                            with open(translated_file, "w", encoding="utf-8") as f:
                                f.write(translated_content)

                            output_files[f"{lang_code}_{fmt}"] = translated_file
                            self.logger.info(
                                f"Exported {lang_code} subtitles: {translated_file}"
                            )

            return output_files

        except Exception as e:
            self.logger.error(f"Error exporting subtitles: {e}")
            return output_files

    def get_supported_languages(self) -> Dict[str, str]:
        """Get supported language codes and names.

        Returns:
            Dictionary mapping language codes to names
        """
        return {"ja": "Japanese", "en": "English", "zh": "Traditional Chinese"}

    def unload_models(self):
        """Unload all translation models to free memory."""
        if self.sakura_translator:
            self.sakura_translator.unload_model()
            self.logger.info("🌸 SakuraLLM model unloaded")

        if self.multi_stage_translator:
            self.multi_stage_translator.unload_models()
            self.logger.info("Multi-stage translation models unloaded")

        self.logger.info("All translation models unloaded")

    def get_active_translator_info(self) -> Dict[str, Any]:
        """Get information about the currently active translator.

        Returns:
            Dictionary with translator information
        """
        if self.sakura_translator:
            return {
                "type": "SakuraLLM",
                "model": self.sakura_config.get("model_name"),
                "specialization": "Japanese → Chinese (High Quality)",
                "device": self.sakura_config.get("device"),
                "enabled": True,
                "icon": "🌸",
            }
        elif self.multi_stage_translator:
            return {
                "type": "Multi-Stage",
                "ja_en_model": self.translation_config.get("ja_to_en_model"),
                "en_zh_model": self.translation_config.get("en_to_zh_model"),
                "specialization": "Japanese → English → Chinese",
                "device": self.translation_config.get("device"),
                "enabled": True,
                "icon": "🔄",
            }
        else:
            return {
                "type": "None",
                "enabled": False,
                "error": "No translator initialized",
            }

    def is_sakura_active(self) -> bool:
        """Check if SakuraLLM is the active translator.

        Returns:
            True if SakuraLLM is active, False otherwise
        """
        return self.sakura_translator is not None and self.config.is_sakura_enabled()

    def get_supported_language_pairs(self) -> Dict[str, List[str]]:
        """Get supported language pairs for the active translator.

        Returns:
            Dictionary with source and target languages
        """
        if self.sakura_translator:
            return {
                "source": ["ja"],
                "target": ["zh"],
                "note": "SakuraLLM specializes in Japanese→Chinese translation",
            }
        elif self.multi_stage_translator:
            return {
                "source": ["ja"],
                "target": ["en", "zh"],
                "note": "Multi-stage: ja→en and ja→en→zh",
            }
        else:
            return {"source": [], "target": [], "note": "No translator available"}

    def __enter__(self):
        """Context manager entry."""
        self.load_models()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.unload_models()
