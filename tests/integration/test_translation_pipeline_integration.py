#!/usr/bin/env python3
"""
Integration test for translation pipeline with standard multi-stage translator
"""
import pytest
import logging
from pathlib import Path

from src.utils.config import Config
from src.translation.translation_pipeline import TranslationPipeline
from src.models.subtitle_data import SubtitleFile, SubtitleSegment

# Set up logging
logging.basicConfig(level=logging.INFO)


@pytest.mark.integration
@pytest.mark.slow  
@pytest.mark.model_download
def test_translation_pipeline_integration():
    """Integration test for translation pipeline with sample text."""
    
    # Load config
    config = Config()
    
    # Create translation pipeline
    pipeline = TranslationPipeline(config)
    
    # Create a sample subtitle file
    sample_segments = [
        SubtitleSegment(
            start_time=0.0,
            end_time=5.0,
            text="こんにちは、世界！これは日本語のテストです。"
        ),
        SubtitleSegment(
            start_time=5.0,
            end_time=10.0,
            text="今日はいい天気ですね。"
        )
    ]
    
    subtitle_file = SubtitleFile(
        segments=sample_segments,
        source_language="ja"
    )
    
    # Test loading models
    success = pipeline.load_models()
    assert success, "Failed to load translation models"
    
    # Test translation to multiple languages
    target_languages = ["en", "zh"]
    result = pipeline.translate(subtitle_file, target_languages)
    
    # Verify results
    assert result is not None, "Translation returned None"
    assert "en" in result.translations, "English translation missing"
    assert "zh" in result.translations, "Chinese translation missing"
    
    # Verify translations exist
    en_translations = result.translations["en"]
    zh_translations = result.translations["zh"]
    
    assert len(en_translations) == 2, f"Expected 2 EN translations, got {len(en_translations)}"
    assert len(zh_translations) == 2, f"Expected 2 ZH translations, got {len(zh_translations)}"
    
    # Verify translations are not empty and different from original
    for translation in en_translations:
        assert translation.translated_text.strip(), "English translation is empty"
        assert translation.translated_text != translation.original_text, "English translation identical to original"
        
    for translation in zh_translations:
        assert translation.translated_text.strip(), "Chinese translation is empty"
        assert translation.translated_text != translation.original_text, "Chinese translation identical to original"


if __name__ == "__main__":
    """Run as standalone script for manual testing"""
    print("🧪 Testing Translation Pipeline Integration")
    print("=" * 50)
    
    try:
        test_translation_pipeline_integration()
        print("✅ Integration test passed!")
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        raise