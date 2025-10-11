"""Translation pipeline for coordinating multi-language translation."""

import logging
from typing import List, Dict, Optional, Callable, Any
from pathlib import Path

from .huggingface_translator import HuggingFaceTranslator, MultiStageTranslator
from ..models.subtitle_data import SubtitleSegment, TranslationResult, SubtitleFile
from ..utils.config import Config
from ..utils.logger import LoggerMixin

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
        
        # Initialize translators
        self.multi_stage_translator = None
        self._initialize_translators()
    
    def _initialize_translators(self):
        """Initialize translation models."""
        try:
            self.multi_stage_translator = MultiStageTranslator(
                ja_en_model=self.translation_config.get("ja_to_en_model", "Helsinki-NLP/opus-mt-ja-en"),
                en_zh_model=self.translation_config.get("en_to_zh_model", "Helsinki-NLP/opus-mt-en-zh"),
                device=self.translation_config.get("device", "auto"),
                batch_size=self.translation_config.get("batch_size", 8),
                max_length=self.translation_config.get("max_length", 512)
            )
            
            self.logger.info("Translation pipeline initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing translation pipeline: {e}")
            self.multi_stage_translator = None
    
    def load_models(self) -> bool:
        """Load all translation models.
        
        Returns:
            True if successful, False otherwise
        """
        if self.multi_stage_translator is None:
            self._initialize_translators()
        
        if self.multi_stage_translator:
            return self.multi_stage_translator.load_models()
        
        return False
    
    def translate_subtitle_file(self,
                               subtitle_file: SubtitleFile,
                               target_languages: List[str] = None,
                               progress_callback: Optional[Callable[[str, float], None]] = None) -> SubtitleFile:
        """Translate entire subtitle file to target languages.
        
        Args:
            subtitle_file: SubtitleFile to translate
            target_languages: List of target language codes ['en', 'zh']
            progress_callback: Callback for progress updates (stage, progress)
            
        Returns:
            Updated SubtitleFile with translations
        """
        if target_languages is None:
            target_languages = ['en', 'zh']  # Default to English and Chinese
        
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
            
            # Perform multi-stage translation
            if progress_callback:
                progress_callback("translation", 0.0)
            
            translation_results = self.multi_stage_translator.translate_to_both(
                texts,
                progress_callback=lambda p: progress_callback("translation", p) if progress_callback else None
            )
            
            # Add translations to subtitle file
            for lang_code, results in translation_results.items():
                if lang_code in target_languages and results:
                    subtitle_file.add_translation(lang_code, results)
                    self.logger.info(f"Added {len(results)} translations for {lang_code}")
            
            if progress_callback:
                progress_callback("translation", 1.0)
            
            return subtitle_file
            
        except Exception as e:
            self.logger.error(f"Error in translation pipeline: {e}")
            return subtitle_file
    
    def translate_segments_to_language(self,
                                     segments: List[SubtitleSegment],
                                     target_language: str,
                                     progress_callback: Optional[Callable[[float], None]] = None) -> List[TranslationResult]:
        """Translate segments to a specific target language.
        
        Args:
            segments: List of subtitle segments
            target_language: Target language code
            progress_callback: Optional progress callback
            
        Returns:
            List of translation results
        """
        try:
            # For now, we support ja->en and en->zh through multi-stage
            if target_language == 'en':
                # Direct Japanese to English
                if not self.multi_stage_translator.ja_en_translator.is_loaded:
                    if not self.multi_stage_translator.ja_en_translator.load_model():
                        return []
                
                texts = [segment.text for segment in segments]
                return self.multi_stage_translator.ja_en_translator.translate_batch(
                    texts, progress_callback
                )
            
            elif target_language == 'zh':
                # Two-stage: ja->en->zh
                texts = [segment.text for segment in segments]
                results = self.multi_stage_translator.translate_to_both(
                    texts, progress_callback
                )
                return results.get('zh', [])
            
            else:
                self.logger.error(f"Unsupported target language: {target_language}")
                return []
                
        except Exception as e:
            self.logger.error(f"Error translating to {target_language}: {e}")
            return []
    
    def export_translated_subtitles(self,
                                  subtitle_file: SubtitleFile,
                                  output_dir: Path,
                                  formats: List[str] = None) -> Dict[str, Path]:
        """Export translated subtitles to files.
        
        Args:
            subtitle_file: SubtitleFile with translations
            output_dir: Output directory
            formats: List of formats to export ['srt'] (more can be added later)
            
        Returns:
            Dictionary mapping language_format to file paths
        """
        if formats is None:
            formats = ['srt']
        
        output_files = {}
        
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Base filename (without extension)
            base_name = subtitle_file.video_file.stem if subtitle_file.video_file else "subtitles"
            
            # Export original (Japanese)
            for fmt in formats:
                if fmt == 'srt':
                    original_content = subtitle_file.export_srt()
                    original_file = output_dir / f"{base_name}_ja.srt"
                    
                    with open(original_file, 'w', encoding='utf-8') as f:
                        f.write(original_content)
                    
                    output_files['ja_srt'] = original_file
                    self.logger.info(f"Exported Japanese subtitles: {original_file}")
            
            # Export translations
            for lang_code, translations in subtitle_file.translations.items():
                if translations:
                    for fmt in formats:
                        if fmt == 'srt':
                            translated_content = subtitle_file.export_srt(lang_code)
                            translated_file = output_dir / f"{base_name}_{lang_code}.srt"
                            
                            with open(translated_file, 'w', encoding='utf-8') as f:
                                f.write(translated_content)
                            
                            output_files[f'{lang_code}_{fmt}'] = translated_file
                            self.logger.info(f"Exported {lang_code} subtitles: {translated_file}")
            
            return output_files
            
        except Exception as e:
            self.logger.error(f"Error exporting subtitles: {e}")
            return output_files
    
    def get_supported_languages(self) -> Dict[str, str]:
        """Get supported language codes and names.
        
        Returns:
            Dictionary mapping language codes to names
        """
        return {
            'ja': 'Japanese',
            'en': 'English', 
            'zh': 'Traditional Chinese'
        }
    
    def unload_models(self):
        """Unload all translation models to free memory."""
        if self.multi_stage_translator:
            self.multi_stage_translator.unload_models()
        
        self.logger.info("Translation models unloaded")
    
    def __enter__(self):
        """Context manager entry."""
        self.load_models()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.unload_models()