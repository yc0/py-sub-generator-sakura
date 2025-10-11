"""Tests for SakuraLLM translator integration."""

from unittest.mock import Mock, patch

import pytest
import torch

from src.models.subtitle_data import TranslationResult
from src.translation.sakura_translator_llama_cpp import SakuraTranslator
from src.utils.config import Config


class TestSakuraConfig:
    """Test SakuraLLM configuration functionality."""

    @pytest.mark.unit
    def test_sakura_config_defaults(self):
        """Test SakuraLLM default configuration."""
        config = Config()
        sakura_config = config.get_sakura_config()

        assert "enabled" in sakura_config
        assert "model_name" in sakura_config
        assert "available_models" in sakura_config
        assert not config.is_sakura_enabled()  # Disabled by default

    @pytest.mark.unit
    def test_available_sakura_models(self):
        """Test available SakuraLLM models."""
        config = Config()
        models = config.get_available_sakura_models()

        # Should have standard SakuraLLM models
        expected_models = ["sakura-1.5b-v1.0", "sakura-14b-v1.0"]
        for model in expected_models:
            assert model in models
            assert "model_name" in models[model]
            assert "description" in models[model]
            assert "vram_required" in models[model]

    @pytest.mark.unit
    def test_sakura_model_selection(self):
        """Test SakuraLLM model selection."""
        config = Config()

        # Test setting a valid model
        success = config.set_sakura_model("sakura-1.5b-v1.0")
        assert success

        model_info = config.get_sakura_model_info()
        assert "SakuraLLM/Sakura-1.5B-Qwen2.5-v1.0-GGUF" in model_info["model_name"]
        assert "sakura-1.5b-qwen2.5-v1.0-q4_k_m.gguf" in model_info["model_file"]

        # Test setting invalid model
        success = config.set_sakura_model("invalid-model")
        assert not success

    @pytest.mark.unit
    def test_sakura_enable_disable(self):
        """Test enabling/disabling SakuraLLM."""
        config = Config()

        # Initially disabled
        assert not config.is_sakura_enabled()

        # Enable
        config.set("sakura.enabled", True)
        assert config.is_sakura_enabled()

        # Disable
        config.set("sakura.enabled", False)
        assert not config.is_sakura_enabled()

    @pytest.mark.unit
    def test_get_sakura_model_info(self):
        """Test getting specific model info."""
        config = Config()

        # Test getting specific model info
        model_info = config.get_sakura_model_info("sakura-14b-v1.0")
        assert model_info["model_name"] == "SakuraLLM/Sakura-14B-Qwen2.5-v1.0-GGUF"
        assert model_info["vram_required"] == "16GB"

        # Test getting current model info
        current_info = config.get_sakura_model_info()
        assert "model_name" in current_info


class TestSakuraTranslator:
    """Test SakuraTranslator functionality."""

    @pytest.mark.unit
    def test_sakura_translator_init(self):
        """Test SakuraTranslator initialization."""
        config = Config()
        config.set("sakura.enabled", True)

        # Test with config
        translator = SakuraTranslator(config=config)
        assert translator.source_lang == "ja"
        assert translator.target_lang == "zh"
        assert translator.config == config
        assert translator.use_chat_template

    @pytest.mark.unit
    def test_sakura_translator_with_model_key(self):
        """Test SakuraTranslator with specific model."""
        config = Config()

        # Test with specific model key
        translator = SakuraTranslator(config=config, model_key="sakura-1.5b-v1.0")
        assert "SakuraLLM/Sakura-1.5B-Qwen2.5-v1.0-GGUF" in translator.model_name

    @pytest.mark.unit
    def test_sakura_translator_invalid_model(self):
        """Test SakuraTranslator with invalid model."""
        config = Config()

        with pytest.raises(ValueError, match="not found"):
            SakuraTranslator(config=config, model_key="invalid-model")

    @pytest.mark.unit
    def test_create_translation_prompt(self):
        """Test SakuraLLM prompt creation."""
        config = Config()
        translator = SakuraTranslator(config=config)

        # Test ChatML format
        text = "こんにちは"
        prompt = translator._create_translation_prompt(text)

        assert "<|im_start|>system" in prompt
        assert "<|im_start|>user" in prompt
        assert "<|im_start|>assistant" in prompt
        assert "将下面的日文文本翻译成中文" in prompt
        assert text in prompt

        # Test simple format
        translator.use_chat_template = False
        simple_prompt = translator._create_translation_prompt(text)
        assert simple_prompt == f"将下面的日文文本翻译成中文：{text}"

    @pytest.mark.unit
    def test_sakura_model_info(self):
        """Test getting SakuraLLM model information."""
        config = Config()
        translator = SakuraTranslator(config=config)

        info = translator.get_model_info()
        expected_keys = [
            "model_name",
            "source_language",
            "target_language",
            "device",
            "dtype",
            "loaded",
            "specialization",
        ]

        for key in expected_keys:
            assert key in info

        assert info["source_language"] == "ja"
        assert info["target_language"] == "zh"
        assert info["specialization"] == "Japanese → Chinese (Light Novel Style)"

    @pytest.mark.unit
    def test_recommended_models(self):
        """Test SakuraLLM model recommendations."""
        recommendations = SakuraTranslator.get_recommended_models()

        expected_categories = [
            "low_vram",
            "medium_vram",
            "high_vram",
            "maximum_quality",
        ]
        for category in expected_categories:
            assert category in recommendations
            rec = recommendations[category]
            assert "model_key" in rec
            assert "vram_required" in rec
            assert "description" in rec
            assert "performance" in rec

    @pytest.mark.unit
    def test_create_from_config_disabled(self):
        """Test creating SakuraTranslator when disabled."""
        config = Config()
        config.set("sakura.enabled", False)

        with pytest.raises(ValueError, match="not enabled"):
            SakuraTranslator.create_from_config(config)

    @pytest.mark.unit
    def test_create_from_config_enabled(self):
        """Test creating SakuraTranslator when enabled."""
        config = Config()
        config.set("sakura.enabled", True)

        translator = SakuraTranslator.create_from_config(config)
        assert isinstance(translator, SakuraTranslator)

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_sakura_device_detection(self):
        """Test SakuraLLM device detection."""
        config = Config()
        translator = SakuraTranslator(config=config)

        # Should detect optimal device
        assert translator.optimal_device in [
            "cpu",
            "mps",
            "cuda",
        ] or translator.optimal_device.startswith("cuda")

        # Should prefer GPU if available
        if torch.cuda.is_available():
            assert translator.optimal_device.startswith("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            assert translator.optimal_device == "mps"


class TestSakuraTranslationResults:
    """Test SakuraLLM translation results."""

    @pytest.mark.unit
    @patch("transformers.AutoModelForCausalLM")
    @patch("transformers.AutoTokenizer")
    def test_sakura_translation_result_format(
        self, mock_tokenizer_class, mock_model_class
    ):
        """Test SakuraLLM translation result format."""
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<|endoftext|>"
        mock_tokenizer.pad_token_id = 50256
        mock_tokenizer.eos_token_id = 50256
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
        mock_tokenizer.decode.return_value = "你好"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        # Mock model
        mock_model = Mock()
        mock_model.parameters.return_value = [torch.zeros(100)]
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        mock_model_class.from_pretrained.return_value = mock_model

        # Create translator and test translation
        config = Config()
        translator = SakuraTranslator(config=config)

        # Should load successfully with mocks
        assert translator.load_model()

        # Test translation
        result = translator.translate_text("こんにちは")

        assert isinstance(result, TranslationResult)
        assert result.original_text == "こんにちは"
        assert result.source_language == "ja"
        assert result.target_language == "zh"
        assert result.confidence > 0.9  # SakuraLLM should have high confidence
        assert "SakuraLLM" in result.translation_model


class TestSakuraIntegration:
    """Test SakuraLLM integration with the overall system."""

    @pytest.mark.integration
    def test_sakura_config_save_load(self):
        """Test saving and loading SakuraLLM configuration."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "test_config.json"

            # Create and configure
            config = Config(config_file=config_file)
            config.set("sakura.enabled", True)
            config.set_sakura_model("sakura-14b-v1.0")

            # Save
            config.save_config()

            # Load new config
            config2 = Config(config_file=config_file)
            config2.load_config()

            # Verify
            assert config2.is_sakura_enabled()
            model_info = config2.get_sakura_model_info()
            assert "SakuraLLM/Sakura-14B-Qwen2.5-v1.0-GGUF" in model_info["model_name"]

    @pytest.mark.integration
    def test_sakura_with_translation_pipeline(self):
        """Test SakuraLLM integration with translation pipeline."""
        config = Config()
        config.set("sakura.enabled", True)

        # This would be tested with actual TranslationPipeline integration
        # For now, just verify the translator can be created
        translator = SakuraTranslator.create_from_config(config)
        assert translator is not None
        assert translator.source_lang == "ja"
        assert translator.target_lang == "zh"
