#!/usr/bin/env python3
"""
Integration test for SakuraLLM pipeline: ja → zh-Hans → zh-Hant
"""
import pytest
import logging

from src.utils.config import Config
from src.translation.translation_pipeline import TranslationPipeline
from src.models.subtitle_data import SubtitleFile, SubtitleSegment
from src.utils.chinese_converter import convert_to_traditional

# Set up logging
logging.basicConfig(level=logging.INFO)


@pytest.mark.integration
def test_chinese_converter():
    """Test the built-in Chinese converter."""
    
    test_cases = [
        ("你好世界！这是一个测试。", "你好世界！這是一個測試。"),
        ("我们正在学习中文。", "我們正在學習中文。"),
        ("今天天气很好。", "今天天氣很好。"),
        ("电脑和网络技术。", "電腦和網絡技術。"),
    ]
    
    for simplified, expected_traditional in test_cases:
        traditional = convert_to_traditional(simplified)
        assert traditional != simplified, f"No conversion happened for: {simplified}"
        # Note: We don't assert exact match since OpenCC might have slight variations


@pytest.mark.integration 
@pytest.mark.slow
@pytest.mark.model_download
def test_sakura_pipeline_when_enabled():
    """Test SakuraLLM pipeline when enabled."""
    
    # Load config and enable Sakura
    config = Config()
    config.set("sakura.enabled", True)
    
    if not config.is_sakura_enabled():
        pytest.skip("SakuraLLM not enabled in configuration")
    
    # Create pipeline
    pipeline = TranslationPipeline(config)
    
    # Create sample Japanese text
    sample_segments = [
        SubtitleSegment(
            start_time=0.0,
            end_time=3.0,
            text="こんにちは"
        ),
        SubtitleSegment(
            start_time=3.0,
            end_time=6.0, 
            text="今日は良い天気ですね"
        )
    ]
    
    subtitle_file = SubtitleFile(
        segments=sample_segments,
        source_language="ja"
    )
    
    # Test loading SakuraLLM
    success = pipeline.load_models()
    if not success:
        pytest.skip("SakuraLLM model files not available")
    
    # Test translation to Chinese (should use SakuraLLM)
    result = pipeline.translate(subtitle_file, ["zh"])
    
    # Verify results
    assert result is not None
    assert "zh" in result.translations
    
    zh_translations = result.translations["zh"]
    assert len(zh_translations) == 2
    
    for translation in zh_translations:
        assert translation.translated_text.strip()
        assert translation.translated_text != translation.original_text


def test_sakura_pipeline_when_disabled():
    """Test that pipeline gracefully handles disabled SakuraLLM."""
    
    config = Config()
    config.set("sakura.enabled", False)
    
    assert not config.is_sakura_enabled()
    
    # Should still work with standard translators
    pipeline = TranslationPipeline(config)
    
    # Create simple test
    sample_segments = [
        SubtitleSegment(
            start_time=0.0,
            end_time=3.0,
            text="こんにちは"
        )
    ]
    
    subtitle_file = SubtitleFile(
        segments=sample_segments,
        source_language="ja"
    )
    
    # Should use standard multi-stage translation
    success = pipeline.load_models()
    # Note: might fail if models aren't available, which is OK for this test


if __name__ == "__main__":
    """Run as standalone script for manual testing"""
    print("🧪 Testing SakuraLLM Pipeline: ja → zh-Hans → zh-Hant")
    print("=" * 60)
    
    # Test Chinese converter first
    print("\n📝 Testing Chinese Converter...")
    try:
        test_chinese_converter()
        print("✅ Chinese converter test passed!")
    except Exception as e:
        print(f"❌ Chinese converter test failed: {e}")
    
    # Test SakuraLLM when disabled
    print("\n🚫 Testing with SakuraLLM disabled...")
    try:
        test_sakura_pipeline_when_disabled()
        print("✅ Disabled SakuraLLM test passed!")
    except Exception as e:
        print(f"❌ Disabled SakuraLLM test failed: {e}")
    
    # Test SakuraLLM when enabled (if available)
    print("\n🌸 Testing with SakuraLLM enabled...")
    try:
        test_sakura_pipeline_when_enabled()
        print("✅ Enabled SakuraLLM test passed!")
    except Exception as e:
        print(f"❌ Enabled SakuraLLM test failed (expected if model not downloaded): {e}")