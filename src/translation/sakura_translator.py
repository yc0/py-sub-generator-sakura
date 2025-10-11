"""SakuraLLM-specific translator implementation for high-quality Japanese translation."""

import logging
from typing import Any, Callable, Dict, Optional

import torch

from ..models.subtitle_data import TranslationResult
from ..utils.config import Config
from .interface.pytorch_translator import PyTorchTranslator

logger = logging.getLogger(__name__)


class SakuraTranslator(PyTorchTranslator):
    """SakuraLLM-optimized translator for high-quality Japanese translation."""

    def __init__(
        self, config: Optional[Config] = None, model_key: Optional[str] = None, **kwargs
    ):
        """Initialize SakuraLLM translator.

        Args:
            config: Application configuration object
            model_key: Specific SakuraLLM model key (e.g., 'sakura-1b8-v1.0')
            **kwargs: Additional parameters
        """
        self.config = config or Config()

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
            model_name = sakura_config.get(
                "model_name", "SakuraLLM/Sakura-1B8-Qwen2.5-v1.0-GGUF"
            )
            model_file = sakura_config.get("model_file")

        # Configure SakuraLLM-specific parameters
        sakura_kwargs = {
            "batch_size": sakura_config.get("batch_size", 1),
            "max_length": sakura_config.get("context_length", 8192),
            "torch_dtype": sakura_config.get("torch_dtype", "float16"),
            "force_gpu": sakura_config.get("force_gpu", True),
            "use_safetensors": True,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
        }

        # Update with any provided kwargs
        sakura_kwargs.update(kwargs)

        # Initialize with Japanese to Chinese translation by default
        super().__init__(
            model_name=model_name,
            source_lang="ja",  # SakuraLLM specializes in Japanese
            target_lang="zh",  # To Chinese
            device=sakura_config.get("device", "auto"),
            **sakura_kwargs,
        )

        self.model_file = model_file
        self.sakura_config = sakura_config
        self.use_chat_template = sakura_config.get("use_chat_template", True)

        # SakuraLLM generation parameters
        self.generation_config = {
            "max_new_tokens": sakura_config.get("max_new_tokens", 512),
            "temperature": sakura_config.get("temperature", 0.1),
            "top_p": sakura_config.get("top_p", 0.95),
            "repetition_penalty": sakura_config.get("repetition_penalty", 1.1),
            "do_sample": True,
            "use_cache": True,
            "pad_token_id": None,  # Will be set after tokenizer loads
            "eos_token_id": None,  # Will be set after tokenizer loads
        }

        logger.info("🌸 SakuraLLM Translator initialized")
        logger.info(f"Model: {model_name}")
        if model_file:
            logger.info(f"Model file: {model_file}")
        logger.info(f"Translation: {self.source_lang} → {self.target_lang}")
        logger.info(f"Device: {self.optimal_device}")

    def load_model(self) -> bool:
        """Load SakuraLLM model with optimizations.

        Returns:
            True if successful, False otherwise
        """
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.logger.info(f"🌸 Loading SakuraLLM model: {self.model_name}")

            # Load tokenizer with SakuraLLM-specific settings
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                use_fast=True,  # Use fast tokenizer when available
            )

            # Configure tokenizer for SakuraLLM
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Update generation config with tokenizer IDs
            self.generation_config["pad_token_id"] = self.tokenizer.pad_token_id
            self.generation_config["eos_token_id"] = self.tokenizer.eos_token_id

            # Load model with SakuraLLM optimizations
            model_kwargs = {
                "torch_dtype": self.torch_dtype,
                "low_cpu_mem_usage": True,
                "trust_remote_code": True,
                "use_safetensors": True,
            }

            # Handle device mapping for different GPU types
            if self.optimal_device.startswith("cuda"):
                model_kwargs["device_map"] = "auto"  # Auto-distribute on CUDA
                self.logger.info("🚀 Using NVIDIA CUDA acceleration")
            elif self.optimal_device == "mps":
                model_kwargs["device_map"] = None  # Manual placement for MPS
                self.logger.info("🍎 Using Apple Silicon MPS acceleration")
            else:
                if self.force_gpu:
                    raise RuntimeError("GPU required but not available!")
                model_kwargs["device_map"] = None
                self.logger.warning("⚠️ Falling back to CPU")

            # Add any model-specific kwargs
            model_kwargs.update(self.model_kwargs)

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, **model_kwargs
            )

            # Manual device placement for MPS
            if self.optimal_device == "mps" and model_kwargs["device_map"] is None:
                self.model.to(self.optimal_device)
                self.logger.info(f"Model moved to {self.optimal_device}")

            # Enable optimizations
            self.model.eval()  # Evaluation mode

            # GPU-specific optimizations
            if self.optimal_device in ["mps"] or self.optimal_device.startswith("cuda"):
                if hasattr(self.model, "half") and self.torch_dtype == torch.float16:
                    if self.optimal_device != "mps":  # MPS handles dtype automatically
                        self.model = self.model.half()
                    self.logger.info("✨ FP16 optimization enabled")

            self.is_loaded = True

            # Log model information
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )

            self.logger.info("🌸 SakuraLLM loaded successfully!")
            self.logger.info(f"📊 Total parameters: {total_params:,}")
            self.logger.info(f"🎯 Trainable parameters: {trainable_params:,}")
            self.logger.info(f"💾 Model dtype: {self.torch_dtype}")
            self.logger.info(f"🔧 Generation config: {self.generation_config}")

            return True

        except ImportError as e:
            self.logger.error(f"Required dependencies not installed: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Error loading SakuraLLM model: {e}")
            return False

    def _create_translation_prompt(self, text: str) -> str:
        """Create SakuraLLM-specific translation prompt.

        Args:
            text: Japanese text to translate

        Returns:
            Formatted prompt for SakuraLLM
        """
        if not self.use_chat_template:
            # Simple prompt format
            return f"将下面的日文文本翻译成中文：{text}"

        # SakuraLLM ChatML format for optimal results
        system_prompt = (
            "你是一个轻小说翻译模型，可以流畅通顺地以日本轻小说的风格将日文翻译成简体中文，"
            "并联系上下文正确使用人称代词，不擅自添加原文中没有的代词。"
        )

        user_prompt = f"将下面的日文文本翻译成中文：{text}"

        # Use ChatML format for SakuraLLM
        prompt = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        return prompt

    def translate_text(
        self, text: str, progress_callback: Optional[Callable[[float], None]] = None
    ) -> TranslationResult:
        """Translate Japanese text to Chinese using SakuraLLM.

        Args:
            text: Japanese text to translate
            progress_callback: Optional progress callback

        Returns:
            Translation result with high-quality Chinese translation
        """
        if not self.is_loaded:
            if not self.load_model():
                return TranslationResult(
                    original_text=text,
                    translated_text="",
                    source_language=self.source_lang,
                    target_language=self.target_lang,
                    confidence=0.0,
                    translation_model=self.model_name,
                )

        try:
            if progress_callback:
                progress_callback(0.1)

            # Create SakuraLLM-specific prompt
            prompt = self._create_translation_prompt(text)

            if progress_callback:
                progress_callback(0.2)

            # Tokenize with appropriate length
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length - self.generation_config["max_new_tokens"],
                padding=False,  # No padding for single generation
            )

            # Move to device
            inputs = {k: v.to(self.optimal_device) for k, v in inputs.items()}

            if progress_callback:
                progress_callback(0.4)

            # Generate translation with SakuraLLM optimizations
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **self.generation_config)

            if progress_callback:
                progress_callback(0.8)

            # Decode only the generated part
            input_length = inputs["input_ids"].shape[-1]
            generated_tokens = outputs[0][input_length:]

            translated_text = self.tokenizer.decode(
                generated_tokens, skip_special_tokens=True
            ).strip()

            if progress_callback:
                progress_callback(1.0)

            # Clean up translation (remove any remaining prompt artifacts)
            if self.use_chat_template and translated_text.startswith("中文："):
                translated_text = translated_text[3:].strip()

            self.logger.debug(
                f"🌸 SakuraLLM translation: '{text}' → '{translated_text}'"
            )

            return TranslationResult(
                original_text=text,
                translated_text=translated_text,
                source_language=self.source_lang,
                target_language=self.target_lang,
                confidence=0.95,  # SakuraLLM typically produces high-quality translations
                translation_model=self.model_name,
            )

        except Exception as e:
            self.logger.error(f"Error during SakuraLLM translation: {e}")
            return TranslationResult(
                original_text=text,
                translated_text="",
                source_language=self.source_lang,
                target_language=self.target_lang,
                confidence=0.0,
                translation_model=self.model_name,
            )

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current SakuraLLM model.

        Returns:
            Dictionary with model information
        """
        base_info = {
            "model_name": self.model_name,
            "model_file": self.model_file,
            "source_language": self.source_lang,
            "target_language": self.target_lang,
            "device": self.optimal_device,
            "dtype": str(self.torch_dtype),
            "loaded": self.is_loaded,
            "specialization": "Japanese → Chinese (Light Novel Style)",
            "framework": "SakuraLLM + PyTorch + Transformers",
        }

        if self.is_loaded and self.model is not None:
            total_params = sum(p.numel() for p in self.model.parameters())
            base_info["total_parameters"] = total_params
            base_info["parameters_human"] = f"{total_params / 1e9:.1f}B"

            # Memory usage estimation
            if hasattr(self.model, "get_memory_footprint"):
                try:
                    memory_mb = self.model.get_memory_footprint() / (1024 * 1024)
                    base_info["memory_usage_mb"] = memory_mb
                    base_info["memory_usage_gb"] = f"{memory_mb / 1024:.1f}GB"
                except:
                    pass

        return base_info

    @classmethod
    def get_recommended_models(cls) -> Dict[str, Dict[str, Any]]:
        """Get recommended SakuraLLM models based on hardware.

        Returns:
            Dictionary with model recommendations
        """
        return {
            "low_vram": {
                "model_key": "sakura-1b8-v1.0",
                "vram_required": "4GB",
                "description": "Best for 4-8GB VRAM GPUs",
                "performance": "Good quality, fastest inference",
            },
            "medium_vram": {
                "model_key": "sakura-7b-v1.0",
                "vram_required": "8GB",
                "description": "Best for 8-16GB VRAM GPUs",
                "performance": "High quality, good speed",
            },
            "high_vram": {
                "model_key": "sakura-14b-v1.0",
                "vram_required": "16GB",
                "description": "Best for 16-24GB VRAM GPUs",
                "performance": "Excellent quality, moderate speed",
            },
            "maximum_quality": {
                "model_key": "sakura-32b-v1.0",
                "vram_required": "32GB",
                "description": "For high-end GPUs (RTX 4090, etc.)",
                "performance": "Maximum quality, slower inference",
            },
        }

    @classmethod
    def create_from_config(
        cls, config: Config, model_key: Optional[str] = None
    ) -> "SakuraTranslator":
        """Create SakuraTranslator from configuration.

        Args:
            config: Application configuration
            model_key: Optional specific model key to use

        Returns:
            Configured SakuraTranslator instance
        """
        if not config.is_sakura_enabled():
            raise ValueError("SakuraLLM is not enabled in configuration")

        return cls(config=config, model_key=model_key)
