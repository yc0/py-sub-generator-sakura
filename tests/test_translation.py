"""
🧪 Translation Backend Tests

Tests for HuggingFace and PyTorch translation backends.
"""

from unittest.mock import Mock

import pytest
import torch

from src.models.subtitle_data import TranslationResult
from src.translation.huggingface_translator import (
    HuggingFaceTranslator,
    MultiStageTranslator,
)
from src.translation.interface.pytorch_translator import PyTorchTranslator


class TestHuggingFaceTranslator:
    """Test HuggingFace pipeline-based translator."""

    def test_initialization(self, sample_translation_config):
        """Test translator initialization."""
        translator = HuggingFaceTranslator(
            model_name="Helsinki-NLP/opus-mt-ja-en",
            source_lang="ja",
            target_lang="en",
            device="cpu",
            batch_size=4,
        )

        assert translator.model_name == "Helsinki-NLP/opus-mt-ja-en"
        assert translator.source_lang == "ja"
        assert translator.target_lang == "en"
        assert translator.device == "cpu"
        assert translator.batch_size == 4
        assert not translator.is_loaded

    @pytest.mark.gpu
    def test_mps_device_mapping(self, skip_if_no_mps):
        """Test MPS device mapping in pipeline creation."""
        translator = HuggingFaceTranslator(
            model_name="Helsinki-NLP/opus-mt-ja-en",
            source_lang="ja",
            target_lang="en",
            device="mps",
        )

        assert translator.device == "mps"

    @pytest.mark.slow
    @pytest.mark.model_download
    def test_model_loading_cpu(self):
        """Test model loading on CPU."""
        translator = HuggingFaceTranslator(
            model_name="Helsinki-NLP/opus-mt-ja-en",
            source_lang="ja",
            target_lang="en",
            device="cpu",
            batch_size=2,
        )

        success = translator.load_model()
        assert success
        assert translator.is_loaded
        assert translator.pipeline is not None

        # Cleanup
        translator.unload_model()
        assert not translator.is_loaded
        assert translator.pipeline is None

    @pytest.mark.slow
    @pytest.mark.gpu
    @pytest.mark.model_download
    def test_model_loading_gpu(self, optimal_device, skip_if_no_gpu):
        """Test model loading on GPU."""
        translator = HuggingFaceTranslator(
            model_name="Helsinki-NLP/opus-mt-ja-en",
            source_lang="ja",
            target_lang="en",
            device=optimal_device,
            batch_size=2,
        )

        success = translator.load_model()
        assert success
        assert translator.is_loaded

        # Check device assignment
        if hasattr(translator.pipeline, "device"):
            if optimal_device == "mps":
                assert str(translator.pipeline.device) == "mps"
            elif optimal_device.startswith("cuda"):
                assert "cuda" in str(translator.pipeline.device)

        translator.unload_model()

    @pytest.mark.slow
    @pytest.mark.model_download
    def test_translation_functionality(self, sample_japanese_texts):
        """Test actual translation functionality."""
        translator = HuggingFaceTranslator(
            model_name="Helsinki-NLP/opus-mt-ja-en",
            source_lang="ja",
            target_lang="en",
            device="cpu",
            batch_size=2,
        )

        success = translator.load_model()
        assert success

        # Test single translation
        result = translator.translate_text(sample_japanese_texts[0])
        assert isinstance(result, TranslationResult)
        assert result.original_text == sample_japanese_texts[0]
        assert len(result.translated_text) > 0
        assert result.translated_text != result.original_text

        # Test batch translation
        results = translator.translate_batch(sample_japanese_texts[:3])
        assert len(results) == 3
        assert all(isinstance(r, TranslationResult) for r in results)

        translator.unload_model()

    def test_text_preprocessing(self):
        """Test text preprocessing functionality."""
        translator = HuggingFaceTranslator(
            model_name="dummy", source_lang="ja", target_lang="en", device="cpu"
        )

        # Test whitespace normalization
        text_with_spaces = "こんにちは   世界"
        processed = translator._preprocess_text(text_with_spaces)
        assert "   " not in processed

        # Test empty text handling
        empty_result = translator._preprocess_text("   ")
        assert empty_result == ""


class TestPyTorchTranslator:
    """Test PyTorch direct model translator."""

    def test_initialization(self):
        """Test PyTorch translator initialization."""
        translator = PyTorchTranslator(
            model_name="dummy-model",
            source_lang="ja",
            target_lang="zh",
            device="cpu",
            force_gpu=False,
        )

        assert translator.model_name == "dummy-model"
        assert translator.source_lang == "ja"
        assert translator.target_lang == "zh"
        assert translator.force_gpu == False
        assert translator.optimal_device == "cpu"

    def test_sakura_prompt_formatting(self):
        """Test SakuraLLM-specific prompt formatting."""
        translator = PyTorchTranslator(
            model_name="SakuraLLM/Sakura-1.5B-Qwen2.5-v1.0-GGUF",
            source_lang="ja",
            target_lang="zh",
            device="cpu",
            force_gpu=False,
        )

        test_text = "こんにちは"
        prompt = translator._create_translation_prompt(test_text)

        # Should use ChatML format for SakuraLLM
        assert "<|im_start|>system" in prompt
        assert "<|im_start|>user" in prompt
        assert "<|im_start|>assistant" in prompt
        assert "轻小说翻译模型" in prompt
        assert test_text in prompt

    def test_generic_prompt_formatting(self):
        """Test generic prompt formatting."""
        translator = PyTorchTranslator(
            model_name="some-generic-model",
            source_lang="ja",
            target_lang="en",
            device="cpu",
            force_gpu=False,
        )

        test_text = "こんにちは"
        prompt = translator._create_translation_prompt(test_text)

        # Should use generic format
        assert "Translate the following Japanese text to English" in prompt
        assert test_text in prompt

    def test_recommended_config(self):
        """Test recommended configuration generation."""
        config = PyTorchTranslator.get_recommended_config()

        assert isinstance(config, dict)
        assert "batch_size" in config
        assert "max_length" in config
        assert "torch_dtype" in config
        assert "device" in config
        assert config["device"] == "auto"
        assert config["force_gpu"] == True

    @pytest.mark.gpu
    def test_gpu_optimizations_config(self, optimal_device):
        """Test GPU-specific optimizations."""
        translator = PyTorchTranslator(
            model_name="dummy",
            source_lang="ja",
            target_lang="en",
            device=optimal_device,
            torch_dtype="float16",
        )

        assert translator.optimal_device == optimal_device
        if optimal_device != "cpu":
            assert translator.torch_dtype == torch.float16


class TestMultiStageTranslator:
    """Test multi-stage Japanese -> English -> Chinese translation."""

    def test_initialization(self, sample_translation_config):
        """Test multi-stage translator initialization."""
        translator = MultiStageTranslator(
            ja_en_model="Helsinki-NLP/opus-mt-ja-en",
            en_zh_model="Helsinki-NLP/opus-mt-en-zh",
            device="cpu",
            batch_size=2,
        )

        assert translator.ja_en_translator is not None
        assert translator.en_zh_translator is not None
        assert translator.ja_en_translator.device == "cpu"
        assert translator.en_zh_translator.device == "cpu"

    @pytest.mark.slow
    @pytest.mark.model_download
    def test_multi_stage_translation(self, sample_japanese_texts):
        """Test multi-stage translation functionality."""
        translator = MultiStageTranslator(
            ja_en_model="Helsinki-NLP/opus-mt-ja-en",
            en_zh_model="Helsinki-NLP/opus-mt-en-zh",
            device="cpu",
            batch_size=2,
        )

        success = translator.load_models()
        assert success

        # Test translation to both languages
        results = translator.translate_to_both(sample_japanese_texts[:2])

        assert "en" in results
        assert "zh" in results
        assert len(results["en"]) == 2
        assert len(results["zh"]) == 2

        # Check that translations are different from originals
        for result in results["en"]:
            assert result.translated_text != result.original_text

        translator.unload_models()


class TestTranslationResults:
    """Test translation result data structures."""

    def test_translation_result_creation(self):
        """Test TranslationResult creation and attributes."""
        result = TranslationResult(
            original_text="こんにちは",
            translated_text="Hello",
            confidence=0.95,
            source_language="ja",
            target_language="en",
        )

        assert result.original_text == "こんにちは"
        assert result.translated_text == "Hello"
        assert result.confidence == 0.95
        assert result.source_language == "ja"
        assert result.target_language == "en"


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_model_name(self):
        """Test handling of invalid model names."""
        translator = HuggingFaceTranslator(
            model_name="nonexistent/model",
            source_lang="ja",
            target_lang="en",
            device="cpu",
        )

        # Should handle gracefully without crashing
        success = translator.load_model()
        assert not success
        assert not translator.is_loaded

    def test_empty_text_translation(self):
        """Test translation of empty or whitespace-only text."""
        translator = HuggingFaceTranslator(
            model_name="dummy", source_lang="ja", target_lang="en", device="cpu"
        )

        # Mock the pipeline to avoid actual model loading
        translator.pipeline = Mock()
        translator.is_loaded = True

        result = translator.translate_text("")
        assert isinstance(result, TranslationResult)
        assert result.translated_text == ""

    def test_batch_translation_with_mixed_content(self):
        """Test batch translation with mixed empty and valid content."""
        translator = HuggingFaceTranslator(
            model_name="dummy", source_lang="ja", target_lang="en", device="cpu"
        )

        # Mock successful loading
        translator.is_loaded = True
        translator.pipeline = Mock()
        translator.pipeline.return_value = [{"translation_text": "Hello"}]

        mixed_texts = ["こんにちは", "", "   ", "さようなら"]
        results = translator.translate_batch(mixed_texts)

        assert len(results) == 4
        assert all(isinstance(r, TranslationResult) for r in results)
