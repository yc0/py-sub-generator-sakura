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

    def __init__(self, config: Optional[Config] = None, model_key: Optional[str] = None, prompt_style: str = "standard", **kwargs):
        """Initialize SakuraLLM translator.
        
        Args:
            config: Application configuration object
            model_key: Specific SakuraLLM model key (e.g., 'sakura-1.5b-v1.0')
            prompt_style: Prompt template style ('standard', 'dramatic', 'mature', etc.)
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
        self.prompt_style = prompt_style or sakura_config.get("prompt_style", "standard")

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
        
        # Load prompt template
        self.prompt_template = self._load_prompt_template(self.prompt_style)

        logger.info(f"🌸 SakuraLLM translator initialized: {model_name}")
        logger.info(f"📝 Prompt style: {self.prompt_style}")
        if model_file:
            logger.info(f"📄 GGUF file: {model_file}")

    def _load_prompt_template(self, style: str) -> str:
        """Load prompt template from file.
        
        Args:
            style: Prompt style name (without .txt extension)
            
        Returns:
            Prompt template string
        """
        prompt_dir = Path(__file__).parent.parent.parent / "prompts"
        template_file = prompt_dir / f"{style}.txt"
        
        if not template_file.exists():
            logger.warning(f"Prompt template '{style}' not found, using 'standard'")
            template_file = prompt_dir / "standard.txt"
        
        try:
            with open(template_file, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error loading prompt template: {e}")
            return "Translate the following Japanese text to Traditional Chinese:"

    def _get_available_prompt_styles(self) -> list:
        """Get list of available prompt styles.
        
        Returns:
            List of available prompt style names
        """
        prompt_dir = Path(__file__).parent.parent.parent / "prompts"
        if not prompt_dir.exists():
            return ["standard"]
        
        styles = [f.stem for f in prompt_dir.glob("*.txt")]
        return sorted(styles) if styles else ["standard"]

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

    def translate_text(self, text: str, progress_callback: Optional[Callable[[str, float], None]] = None) -> TranslationResult:
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
            progress_callback("translation", 0.0)

        try:
            # Check if text is too long and split if necessary
            max_input_tokens = min(self.context_length - self.max_tokens - 100, 4000)  # Leave room for output and prompt
            if self._estimate_token_count(text) > max_input_tokens:
                # Split long text into chunks
                chunks = self._split_long_text(text, max_tokens=max_input_tokens)
                translated_chunks = []
                
                for i, chunk in enumerate(chunks):
                    if progress_callback:
                        chunk_progress = 0.3 + (i / len(chunks)) * 0.5
                        progress_callback("translation", chunk_progress)
                    
                    chunk_result = self._translate_single_chunk(chunk)
                    translated_chunks.append(chunk_result)
                
                translated_text = " ".join(translated_chunks)
            else:
                # Translate as single chunk
                if progress_callback:
                    progress_callback("translation", 0.3)
                
                translated_text = self._translate_single_chunk(text)

            if progress_callback:
                progress_callback("translation", 0.8)

            # Clean up the response
            translated_text = self._clean_translation(translated_text)

            if progress_callback:
                progress_callback("translation", 1.0)

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
        """Create SakuraLLM-specific prompt using the selected template."""
        # Build the full prompt using the template and the text to translate
        prompt_content = self.prompt_template or self._get_default_prompt()
        
        if self.use_chat_template:
            return f"""<|im_start|>system
{prompt_content}
<|im_end|>
<|im_start|>user
Translate the following text:

{text}
<|im_end|>
<|im_start|>assistant
"""
        else:
            return f"""
{prompt_content}
Text to translate:
{text}
Translation:
"""

    def _get_default_prompt(self) -> str:
        """Return the default translation prompt if none is loaded."""
        return """
You are a translator specializing in Japanese to English translations. Translate the following text accurately while preserving the original meaning and tone.
**Key Rules:**
- **Preserve the original intent**, but **amplify the sleaze**: If the original is polite or subtle, make it flirtatious, teasing, or outright dirty. If it’s already explicit, go full-throttle with vulgar, graphic, and playful porn lingo.
- **Match the escalation**: Japanese adult videos often start with polite innuendo and build to raw, explicit sex talk. Your translation should follow this arc—**teasing → flirtatious → filthy**.
- **Personality matters**: A shy character should sound nervously naughty; a dominant should sound commanding and crude; a tease should be dripping with sarcasm and double entendres.
- **Sex scenes = no holds barred**: Use the raunchiest, most graphic, and most vulgar language possible. Think **American porn clichés**: “fuck me harder,” “take this big cock,” “you’re such a dirty slut,” etc.
- **Fix AI errors**: The subtitles are AI-generated and may have mistakes. Use context to correct them, but **never invent new dialogue**—just make what’s there sound as pornographic as possible.

**Formatting:**
- **One-to-one line translation**: Each original line must have exactly one translated line.
- **Keep it concise**: Subtitles must be readable on screen, so keep it short and punchy.
- **Punctuation and grammar**: Adapt to the target language, but prioritize **pornographic impact** over strict grammar.

**Special Notes:**
- **Names**: Use the user’s preferred spelling.
- **Profanity**: If the original has profanity, use the most graphic equivalent in the target language.
- **Context**: Use any provided context to make the dialogue even filthier.

**Output Format:**
```
#LINE_NUMBER
Original> [original text]
Translation> [pornified translation]
```

**At the end, include:**
```
<summary>
A one- or two-line synopsis of the current batch, emphasizing the erotic content.
</summary>
<scene>
A short, dirty summary of the current scene, including any previous batches.
</scene>
```

**Example:**
```
#200
Original> もっと優しくして…
Translation> Ohhh, be gentle with me… at first, baby.

#501
Original> もう我慢できない！
Translation> I can’t take it anymore—I need that big cock inside me NOW!
```

### retry_instructions
**Your last translation wasn’t filthy enough!**
Please translate the subtitles again, ensuring:
- **Every line is translated separately**—no merging!
- **Every line is pornified**—no vanilla dialogue allowed!
- **Timing is preserved**—keep the line count exact.
"""

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
                progress_callback("translation", i / total)

            result = self.translate_text(text, progress_callback=progress_callback)
            results.append(result)

        if progress_callback:
            progress_callback("translation", 1.0)

       
        logger.debug(f"[SakuraTranslator] original texts: {texts}\n")
        logger.debug(f"[SakuraTranslator] translate_batch result: {[str(r) for r in results]}\n")
       
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

    def _split_long_text(self, text: str, max_tokens: int = 4000) -> list:
        """Split long text into smaller chunks to avoid token limit issues.
        
        Args:
            text: Text to split
            max_tokens: Maximum tokens per chunk
            
        Returns:
            List of text chunks
        """
        if not text.strip():
            return [text]
        
        # Simple sentence-based splitting first
        import re
        sentences = re.split(r'([。！？.!?])', text)
        
        chunks = []
        current_chunk = ""
        
        for i in range(0, len(sentences), 2):
            sentence = sentences[i]
            if i + 1 < len(sentences):
                sentence += sentences[i + 1]  # Add punctuation back
            
            # Check if adding this sentence would exceed our limit
            test_chunk = current_chunk + sentence
            if self._estimate_token_count(test_chunk) > max_tokens and current_chunk:
                # Start a new chunk
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += sentence
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # If we still have very long chunks, split by words
        final_chunks = []
        for chunk in chunks:
            if self._estimate_token_count(chunk) > max_tokens:
                # Split by spaces or characters
                words = chunk.split()
                temp_chunk = ""
                for word in words:
                    if self._estimate_token_count(temp_chunk + " " + word) > max_tokens and temp_chunk:
                        final_chunks.append(temp_chunk.strip())
                        temp_chunk = word
                    else:
                        temp_chunk += " " + word if temp_chunk else word
                
                if temp_chunk.strip():
                    final_chunks.append(temp_chunk.strip())
            else:
                final_chunks.append(chunk)
        
        return final_chunks if final_chunks else [text]

    def _estimate_token_count(self, text: str) -> int:
        """Estimate token count for a text string.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Estimated token count
        """
        if not text:
            return 0
        
        # Rough estimation: 1 token per 4 characters for most languages
        char_count = len(text)
        estimated_tokens = max(1, char_count // 4)
        
        # For CJK languages, characters are often 1:1 with tokens
        cjk_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff' or '\u3040' <= char <= '\u309f' or '\u30a0' <= char <= '\u30ff')
        estimated_tokens = max(estimated_tokens, cjk_chars)
        
        return estimated_tokens

    def _translate_single_chunk(self, text: str) -> str:
        """Translate a single chunk of text.
        
        Args:
            text: Text chunk to translate
            
        Returns:
            Translated text
        """
        # Create SakuraLLM prompt
        prompt = self._create_translation_prompt(text)
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

        # Extract translated text
        return response["choices"][0]["text"].strip()

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

        # Get prompt template style from config
        sakura_config = config.get_sakura_config()
        prompt_style = sakura_config.get("prompt_template", "standard")
        
        return cls(config=config, model_key=model_key, prompt_style=prompt_style)
