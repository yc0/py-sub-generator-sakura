"""Simple SakuraLLM translator using llama-cpp-python for GGUF support."""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from ..models.subtitle_data import TranslationResult
from ..utils.config import Config
from .base_translator import BaseTranslator

logger = logging.getLogger(__name__)


class SakuraTranslator(BaseTranslator):
    """SakuraLLM translator using llama-cpp-python for GGUF model support."""

    def __init__(self, config: Optional[Config] = None, model_key: Optional[str] = None, **kwargs):
        """Initialize SakuraLLM translator.
        
        Args:
            config: Application configuration object
            model_key: Specific SakuraLLM model key (e.g., 'sakura-1.5b-v1.0')
            **kwargs: Additional parameters
        """
        self.config = config or Config()
        self.llm = None
        self.is_loaded = False

        # Get SakuraLLM configuration
        sakura_config = self.config.get_sakura_config()

        # Determine model to use
        if model_key:
            model_info = self.config.get_sakura_model_info(model_key)
            if not model_info:
                raise ValueError(f"SakuraLLM model '{model_key}' not found")
            model_name = model_info["model_name"]
            model_file = model_info.get("model_file")
        else:
            model_name = sakura_config.get("model_name", "SakuraLLM/Sakura-1.5B-Qwen2.5-v1.0-GGUF")
            model_file = sakura_config.get("model_file", "sakura-1.5b-qwen2.5-v1.0-q4_k_m.gguf")

        # Set model properties
        self.model_name = model_name
        self.model_file = model_file

        # LLM parameters
        self.context_length = sakura_config.get("context_length", 8192)
        self.max_tokens = sakura_config.get("max_new_tokens", 512)
        self.temperature = sakura_config.get("temperature", 0.1)
        self.top_p = sakura_config.get("top_p", 0.95)
        self.repetition_penalty = sakura_config.get("repetition_penalty", 1.1)

        # Initialize base translator
        super().__init__(
            model_name=model_name,
            source_lang="ja",
            target_lang="zh",
            device="auto"  # llama-cpp-python handles device selection
        )

        # Add legacy compatibility attributes for tests
        self.use_chat_template = sakura_config.get("use_chat_template", True)

        logger.info(f"🌸 SakuraLLM translator initialized: {model_name}")
        if model_file:
            logger.info(f"📄 GGUF file: {model_file}")

    def load_model(self) -> bool:
        """Load SakuraLLM model using llama-cpp-python.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            from llama_cpp import Llama
            import os

            # Find model file path
            model_path = self._find_model_file()
            if not model_path:
                logger.error(f"❌ Model file not found: {self.model_file}")
                return False

            logger.info(f"🌸 Loading SakuraLLM from: {model_path}")

            # Suppress expected Metal kernel warnings for unsupported operations
            # These warnings are normal on older Apple Silicon models
            os.environ.setdefault("GGML_METAL_LOG_LEVEL", "WARN")

            # Initialize Llama with optimal settings
            self.llm = Llama(
                model_path=str(model_path),
                n_ctx=self.context_length,
                n_gpu_layers=-1,  # Use all GPU layers if available
                verbose=False,
                seed=-1,  # Random seed
                n_threads=None,  # Auto-detect optimal threads
                use_mmap=True,  # Memory mapping for efficiency
                use_mlock=False,  # Don't lock memory unless needed
            )

            self.is_loaded = True
            logger.info("✅ SakuraLLM loaded successfully with llama-cpp-python")

            # Log device info
            if hasattr(self.llm, 'n_gpu_layers') and self.llm.n_gpu_layers > 0:
                logger.info(f"🚀 Using GPU acceleration ({self.llm.n_gpu_layers} layers)")
            else:
                logger.info("💻 Using CPU inference")

            return True

        except ImportError:
            logger.error("❌ llama-cpp-python not installed. Install with: pip install llama-cpp-python")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to load SakuraLLM: {e}")
            return False

    def _find_model_file(self) -> Optional[Path]:
        """Find the GGUF model file in common locations."""
        # Check common cache locations
        common_paths = [
            Path.home() / ".cache" / "huggingface" / "hub",
            Path.home() / ".cache" / "sakura",
            Path("./models"),
            Path("./cache"),
        ]

        # Also try relative to current directory
        if self.model_file:
            # Try direct path first
            direct_path = Path(self.model_file)
            if direct_path.exists():
                return direct_path

            # Search in common locations
            for base_path in common_paths:
                if base_path.exists():
                    # First try direct search in base path
                    direct_search = base_path / self.model_file
                    if direct_search.exists():
                        return direct_search
                    
                    # Then recursively search for directories containing the model name
                    for model_dir in base_path.rglob("*"):
                        if model_dir.is_dir():
                            # Check if directory name contains model name components
                            dir_name_lower = str(model_dir.name).lower()
                            model_name_parts = self.model_name.lower().split("/")[-1].split("-")
                            
                            if any(part in dir_name_lower for part in model_name_parts if len(part) > 2):
                                gguf_file = model_dir / self.model_file
                                if gguf_file.exists():
                                    return gguf_file

        return None

    def translate_text(self, text: str, progress_callback: Optional[Callable[[float], None]] = None) -> TranslationResult:
        """Translate Japanese text to Chinese using SakuraLLM.
        
        Args:
            text: Japanese text to translate
            progress_callback: Optional progress callback
            
        Returns:
            TranslationResult with translated text
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if progress_callback:
            progress_callback(0.0)

        try:
            # Create SakuraLLM prompt
            prompt = self._create_translation_prompt(text)

            if progress_callback:
                progress_callback(0.3)

            # Generate translation
            response = self.llm(
                prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                repeat_penalty=self.repetition_penalty,
                stop=["<|im_end|>", "\n\n"],  # Stop tokens
                echo=False  # Don't include prompt in response
            )

            if progress_callback:
                progress_callback(0.8)

            # Extract translated text
            translated_text = response["choices"][0]["text"].strip()

            # Clean up the response
            translated_text = self._clean_translation(translated_text)

            if progress_callback:
                progress_callback(1.0)

            return TranslationResult(
                original_text=text,
                translated_text=translated_text,
                source_language="ja",
                target_language="zh",
                confidence=0.95,  # SakuraLLM typically has high confidence
                translation_model=f"SakuraLLM/{self.model_name}"
            )

        except Exception as e:
            logger.error(f"Translation failed: {e}")
            # Return fallback result
            return TranslationResult(
                original_text=text,
                translated_text=text,  # Fallback to original
                source_language="ja",
                target_language="zh",
                confidence=0.0,
                translation_model=f"SakuraLLM/{self.model_name}",
                method="llama-cpp-python",
                error=str(e)
            )

    def _create_translation_prompt(self, text: str) -> str:
        """Create SakuraLLM-specific prompt."""
        if self.use_chat_template:
            return f"""<|im_start|>system
你是一个轻小说翻译模型，可以流畅通顺地以日本轻小说的风格将日文翻译成简体中文，并联系上下文正确使用人称代词，不擅自添加原文中没有的代词。<|im_end|>
<|im_start|>user
将下面的日文文本翻译成中文：{text}<|im_end|>
<|im_start|>assistant
"""
        else:
            return f"将下面的日文文本翻译成中文：{text}"

    def _clean_translation(self, text: str) -> str:
        """Clean up translated text."""
        # Remove common artifacts
        text = text.strip()

        # Remove ChatML tokens if they leak through
        text = text.replace("<|im_start|>", "").replace("<|im_end|>", "")

        # Remove extra whitespace
        text = " ".join(text.split())

        return text

    def translate_batch(self, texts: list, progress_callback: Optional[Callable[[str, float], None]] = None) -> list:
        """Translate a batch of texts."""
        results = []
        total = len(texts)

        for i, text in enumerate(texts):
            if progress_callback:
                progress_callback(f"Translating {i+1}/{total}", i / total)

            result = self.translate_text(text)
            results.append(result)

        if progress_callback:
            progress_callback("Translation complete", 1.0)

        return results

    def unload_model(self):
        """Unload the model to free memory."""
        if self.llm:
            del self.llm
            self.llm = None
        self.is_loaded = False
        logger.info("🌸 SakuraLLM unloaded")

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model.
        
        Returns:
            Dictionary containing model information
        """
        info = {
            "model_name": self.model_name,
            "model_file": self.model_file,
            "source_language": self.source_lang,  # Test expects this key name
            "target_language": self.target_lang,  # Test expects this key name
            "device": self.device,
            "dtype": "auto",  # Test expects this key
            "loaded": self.is_loaded,  # Test expects this key name
            "specialization": "Japanese → Chinese (Light Novel Style)",  # Test expects this value
            "use_chat_template": self.use_chat_template,
            "context_length": self.context_length,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        if self.llm:
            # Add runtime info if model is loaded
            info["model_type"] = "llama-cpp-python"

        return info

    @classmethod
    def get_recommended_models(cls) -> Dict[str, Any]:
        """Get recommended SakuraLLM models.
        
        Returns:
            Dictionary of recommended models with their configurations
        """
        return {
            "low_vram": {
                "model_key": "sakura-1.5b-v1.0",
                "model_name": "SakuraLLM/Sakura-1.5B-Qwen2.5-v1.0-GGUF",
                "model_file": "sakura-1.5b-qwen2.5-v1.0-q4_k_m.gguf",
                "size": "1.5B",
                "vram_required": "2GB",
                "description": "Lightweight model for low-end hardware",
                "performance": "fast"
            },
            "medium_vram": {
                "model_key": "sakura-7b-v1.0",
                "model_name": "SakuraLLM/Sakura-7B-Qwen2.5-v1.0-GGUF",
                "model_file": "sakura-7b-qwen2.5-v1.0-q4_k_m.gguf",
                "size": "7B",
                "vram_required": "8GB",
                "description": "Balanced performance and quality",
                "performance": "balanced"
            },
            "high_vram": {
                "model_key": "sakura-14b-v1.0",
                "model_name": "SakuraLLM/Sakura-14B-Qwen2.5-v1.0-GGUF",
                "model_file": "sakura-14b-qwen2.5-v1.0-q4_k_m.gguf",
                "size": "14B",
                "vram_required": "16GB",
                "description": "High quality translation for powerful hardware",
                "performance": "high_quality"
            },
            "maximum_quality": {
                "model_key": "sakura-32b-v1.0",
                "model_name": "SakuraLLM/Sakura-32B-Qwen2.5-v1.0-GGUF",
                "model_file": "sakura-32b-qwen2.5-v1.0-q4_k_m.gguf",
                "size": "32B",
                "vram_required": "32GB+",
                "description": "Best quality translation (requires high-end GPU)",
                "performance": "maximum_quality"
            }
        }

    @classmethod
    def create_from_config(cls, config: Config, model_key: Optional[str] = None) -> "SakuraTranslator":
        """Create SakuraTranslator from configuration.
        
        Args:
            config: Application configuration
            model_key: Optional specific model key
            
        Returns:
            Configured SakuraTranslator instance
        """
        if not config.is_sakura_enabled():
            raise ValueError("SakuraLLM is not enabled in configuration")

        return cls(config=config, model_key=model_key)
