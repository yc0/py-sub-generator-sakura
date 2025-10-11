"""Configuration        "asr": {
    "model_name": "openai/whisper-large-v3",  # Default Whisper model
    "device": "auto",  # auto, cpu, cuda, mps (Apple Silicon)
    "batch_size": 1,
    "language": "ja",  # Japanese
    "return_timestamps": True,
    "chunk_length": 30,  # seconds
    "overlap": 1.0  # seconds
},nt for the application."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class Config:
    """Application configuration management."""

    # Default configuration
    DEFAULT_CONFIG = {
        "asr": {
            "model_name": "openai/whisper-large-v3",
            "device": "auto",  # auto, cpu, cuda, mps (Apple Silicon)
            "batch_size": 1,
            "language": "ja",  # Japanese
            "return_timestamps": True,
            "chunk_length": 30,  # seconds (native Whisper optimal window)
        },
        "translation": {
            "ja_to_en_model": "Helsinki-NLP/opus-mt-ja-en",
            "en_to_zh_model": "Helsinki-NLP/opus-mt-en-zh",
            "device": "auto",  # auto, cpu, cuda, mps (Apple Silicon)
            "batch_size": 8,
            "max_length": 512,
        },
        "sakura": {
            # SakuraLLM configuration for high-quality Japanese translation
            "enabled": False,  # Enable SakuraLLM for advanced translation
            "model_name": "SakuraLLM/Sakura-1.5B-Qwen2.5-v1.0-GGUF",  # Default Sakura model
            "model_file": "sakura-1.5b-qwen2.5-v1.0-q4_k_m.gguf",  # Specific GGUF file
            "device": "auto",  # auto, cpu, cuda, mps
            "context_length": 8192,  # Context window size
            "max_new_tokens": 512,  # Maximum tokens to generate
            "temperature": 0.1,  # Low temperature for consistent translation
            "top_p": 0.95,  # Nucleus sampling
            "repetition_penalty": 1.1,  # Prevent repetition
            "batch_size": 1,  # Start with 1 for large models
            "torch_dtype": "float16",  # Use float16 for memory efficiency
            "force_gpu": True,  # Require GPU for large models
            "use_chat_template": True,  # Use chat template for better results
            # Available SakuraLLM models (latest versions only)
            "available_models": {
                "sakura-1.5b-v1.0": {
                    "model_name": "SakuraLLM/Sakura-1.5B-Qwen2.5-v1.0-GGUF",
                    "model_file": "sakura-1.5b-qwen2.5-v1.0-q4_k_m.gguf",
                    "description": "Latest compact 1.5B parameter model (3GB VRAM)",
                    "vram_required": "3GB",
                    "release_date": "20241012",
                },
                "sakura-14b-v1.0": {
                    "model_name": "SakuraLLM/Sakura-14B-Qwen2.5-v1.0-GGUF",
                    "model_file": "sakura-14b-qwen2.5-v1.0-q4_k_m.gguf",
                    "description": "Latest large 14B parameter model (16GB VRAM)",
                    "vram_required": "16GB",
                    "release_date": "20241008",
                },
            },
        },
        "ui": {
            "window_title": "Sakura Subtitle Generator",
            "window_size": "800x600",
            "theme": "default",
            "progress_update_interval": 100,  # ms
        },
        "output": {
            "default_format": "srt",
            "include_confidence": True,
            "output_directory": "outputs",
            "temp_directory": "temp",
        },
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "file": "logs/app.log",
        },
    }

    def __init__(self, config_file: Optional[Path] = None):
        """Initialize configuration.

        Args:
            config_file: Path to configuration file (optional)
        """
        self.config_file = config_file or Path("config.json")
        self.config = self.DEFAULT_CONFIG.copy()

        # Load config if file exists
        if self.config_file.exists():
            self.load_config()
        else:
            self.save_config()

    def load_config(self) -> bool:
        """Load configuration from file.

        Returns:
            True if successful, False otherwise
        """
        try:
            with open(self.config_file, encoding="utf-8") as f:
                user_config = json.load(f)

            # Deep merge with default config
            self.config = self._deep_merge(self.DEFAULT_CONFIG, user_config)
            logger.info(f"Configuration loaded from: {self.config_file}")
            return True

        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return False

    def save_config(self) -> bool:
        """Save current configuration to file.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Create config directory if it doesn't exist
            self.config_file.parent.mkdir(parents=True, exist_ok=True)

            with open(self.config_file, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)

            logger.info(f"Configuration saved to: {self.config_file}")
            return True

        except Exception as e:
            logger.error(f"Error saving config: {e}")
            return False

    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation.

        Args:
            key_path: Dot-separated path to config value (e.g., "asr.model_name")
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        try:
            keys = key_path.split(".")
            value = self.config

            for key in keys:
                value = value[key]

            return value

        except (KeyError, TypeError):
            return default

    def set(self, key_path: str, value: Any) -> bool:
        """Set configuration value using dot notation.

        Args:
            key_path: Dot-separated path to config value
            value: Value to set

        Returns:
            True if successful, False otherwise
        """
        try:
            keys = key_path.split(".")
            config_ref = self.config

            # Navigate to parent dict
            for key in keys[:-1]:
                if key not in config_ref:
                    config_ref[key] = {}
                config_ref = config_ref[key]

            # Set value
            config_ref[keys[-1]] = value
            return True

        except Exception as e:
            logger.error(f"Error setting config {key_path}: {e}")
            return False

    def get_asr_config(self) -> Dict[str, Any]:
        """Get ASR-specific configuration."""
        return self.config.get("asr", {})

    def get_translation_config(self) -> Dict[str, Any]:
        """Get translation-specific configuration."""
        return self.config.get("translation", {})

    def get_ui_config(self) -> Dict[str, Any]:
        """Get UI-specific configuration."""
        return self.config.get("ui", {})

    def get_output_config(self) -> Dict[str, Any]:
        """Get output-specific configuration."""
        return self.config.get("output", {})

    def get_sakura_config(self) -> Dict[str, Any]:
        """Get SakuraLLM-specific configuration."""
        return self.config.get("sakura", {})

    def is_sakura_enabled(self) -> bool:
        """Check if SakuraLLM is enabled."""
        return self.get("sakura.enabled", False)

    def get_sakura_model_info(self, model_key: Optional[str] = None) -> Dict[str, Any]:
        """Get SakuraLLM model information.

        Args:
            model_key: Specific model key (e.g., 'sakura-1b8-v1.0')
                      If None, returns current configured model info

        Returns:
            Model information dictionary
        """
        if model_key:
            available_models = self.get("sakura.available_models", {})
            return available_models.get(model_key, {})

        # Return current model configuration
        return {
            "model_name": self.get("sakura.model_name"),
            "model_file": self.get("sakura.model_file"),
            "description": "Current configured model",
            "vram_required": "Unknown",
        }

    def set_sakura_model(self, model_key: str) -> bool:
        """Set SakuraLLM model by key.

        Args:
            model_key: Model key from available_models

        Returns:
            True if successful, False if model not found
        """
        model_info = self.get_sakura_model_info(model_key)
        if not model_info:
            logger.error(f"SakuraLLM model '{model_key}' not found in available models")
            return False

        # Update configuration
        success = True
        success &= self.set("sakura.model_name", model_info.get("model_name"))
        success &= self.set("sakura.model_file", model_info.get("model_file"))

        if success:
            logger.info(
                f"SakuraLLM model set to: {model_key} ({model_info.get('description')})"
            )

        return success

    def get_available_sakura_models(self) -> Dict[str, Dict[str, Any]]:
        """Get all available SakuraLLM models."""
        return self.get("sakura.available_models", {})

    def setup_directories(self):
        """Create necessary directories based on config."""
        try:
            # Output directory
            output_dir = Path(self.get("output.output_directory", "outputs"))
            output_dir.mkdir(parents=True, exist_ok=True)

            # Temp directory
            temp_dir = Path(self.get("output.temp_directory", "temp"))
            temp_dir.mkdir(parents=True, exist_ok=True)

            # Log directory
            log_file = Path(self.get("logging.file", "logs/app.log"))
            log_file.parent.mkdir(parents=True, exist_ok=True)

            logger.info("Directories created successfully")

        except Exception as e:
            logger.error(f"Error creating directories: {e}")

    def _deep_merge(self, base_dict: Dict, update_dict: Dict) -> Dict:
        """Deep merge two dictionaries.

        Args:
            base_dict: Base dictionary
            update_dict: Dictionary with updates

        Returns:
            Merged dictionary
        """
        result = base_dict.copy()

        for key, value in update_dict.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result
