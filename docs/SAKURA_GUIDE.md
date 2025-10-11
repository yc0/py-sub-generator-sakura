# 🌸 SakuraLLM Integration Guide

This guide explains how to use **SakuraLLM** for high-quality Japanese to Chinese translation in the Sakura Subtitle Generator.

## What is SakuraLLM?

SakuraLLM is a specialized large language model designed specifically for translating Japanese text (particularly light novels and anime content) into high-quality Chinese. It provides:

- **Superior translation quality** compared to general-purpose models
- **Light novel style preservation** 
- **Context-aware pronoun usage**
- **Natural Chinese expressions**
- **GGUF format optimization** for efficient GPU inference

## Quick Start

### 1. Enable SakuraLLM in Configuration

```python
from src.utils.config import Config

# Create and configure
config = Config()
config.set("sakura.enabled", True)
config.save_config()
```

### 2. Choose Your Model

SakuraLLM offers different model sizes based on your hardware:

| Model | Parameters | VRAM Required | Best For |
|-------|------------|---------------|----------|
| `sakura-1b8-v1.0` | 1.8B | 4GB | Entry-level GPUs |
| `sakura-7b-v1.0` | 7B | 8GB | Mid-range GPUs |
| `sakura-14b-v1.0` | 14B | 16GB | High-end GPUs |  
| `sakura-32b-v1.0` | 32B | 32GB | Enthusiast GPUs |

### 3. Set Your Model

```python
# Set model based on your GPU VRAM
config.set_sakura_model("sakura-1b8-v1.0")  # For 4-8GB VRAM
config.set_sakura_model("sakura-7b-v1.0")   # For 8-16GB VRAM
config.set_sakura_model("sakura-14b-v1.0")  # For 16-24GB VRAM
```

### 4. Create and Use Translator

```python
from src.translation.sakura_translator import SakuraTranslator

# Create translator from config
translator = SakuraTranslator.create_from_config(config)

# Translate Japanese to Chinese
result = translator.translate_text("この小説はとても面白いです。")
print(f"Chinese: {result.translated_text}")
```

## Configuration Options

### Complete Configuration File

Create `config_sakura.json`:

```json
{
  "sakura": {
    "enabled": true,
    "model_name": "SakuraLLM/Sakura-1B8-Qwen2.5-v1.0-GGUF",
    "model_file": "sakura-1b8-qwen2.5-v1.0-q4_k_m.gguf",
    "device": "auto",
    "context_length": 8192,
    "max_new_tokens": 512,
    "temperature": 0.1,
    "top_p": 0.95,
    "repetition_penalty": 1.1,
    "batch_size": 1,
    "torch_dtype": "float16",
    "force_gpu": true,
    "use_chat_template": true
  }
}
```

### Key Parameters

- **`enabled`**: Enable/disable SakuraLLM (default: `false`)
- **`model_name`**: HuggingFace model repository
- **`model_file`**: Specific GGUF file to use
- **`device`**: `"auto"`, `"cuda"`, `"mps"`, or `"cpu"`
- **`temperature`**: Lower = more consistent (0.1 recommended)
- **`force_gpu`**: Require GPU acceleration (recommended: `true`)
- **`use_chat_template`**: Use ChatML format for better results

## Hardware Requirements

### GPU Acceleration (Recommended)

SakuraLLM performs best with GPU acceleration:

| GPU Type | Recommended Model | Expected Performance |
|----------|-------------------|---------------------|
| RTX 3060/4060 (8GB) | `sakura-1b8-v1.0` | Fast inference |
| RTX 3070/4070 (12GB) | `sakura-7b-v1.0` | Excellent quality |
| RTX 3080/4080 (16GB) | `sakura-14b-v1.0` | Superior quality |
| RTX 4090 (24GB+) | `sakura-32b-v1.0` | Maximum quality |
| Apple M1/M2/M3/M4 | `sakura-1b8-v1.0` | MPS acceleration |

### CPU Fallback

While possible, CPU inference is **significantly slower**:
- Only recommended for testing or very small batches
- Set `"force_gpu": false` in configuration
- Expect 10-50x slower performance

## Usage Examples

### Basic Translation

```python
from src.utils.config import Config
from src.translation.sakura_translator import SakuraTranslator

# Setup
config = Config()
config.set("sakura.enabled", True)
config.set_sakura_model("sakura-7b-v1.0")

# Create translator
translator = SakuraTranslator.create_from_config(config)

# Translate
japanese_text = "「おはようございます」と彼女は微笑みながら言った。"
result = translator.translate_text(japanese_text)

print(f"Japanese: {result.original_text}")
print(f"Chinese: {result.translated_text}")
print(f"Confidence: {result.confidence}")
```

### Batch Translation

```python
japanese_texts = [
    "この小説はとても面白いです。",
    "彼は学校に行きました。", 
    "桜の花が美しく咲いています。"
]

results = translator.translate_batch(japanese_texts)

for result in results:
    print(f"🇯🇵 {result.original_text}")
    print(f"🇨🇳 {result.translated_text}")
    print()
```

### Model Information

```python
# Get model details
info = translator.get_model_info()
print(f"Model: {info['model_name']}")
print(f"Parameters: {info.get('parameters_human', 'Unknown')}")
print(f"Memory Usage: {info.get('memory_usage_gb', 'Unknown')}")
print(f"Device: {info['device']}")

# Get recommendations
recommendations = SakuraTranslator.get_recommended_models()
for category, rec in recommendations.items():
    print(f"{category}: {rec['model_key']} ({rec['vram_required']})")
```

## Integration with Subtitle Generation

### Complete Workflow

```python
from src.utils.config import Config
from src.asr.whisper_asr import WhisperASR
from src.translation.sakura_translator import SakuraTranslator

# 1. Setup configuration
config = Config()
config.set("sakura.enabled", True)
config.set_sakura_model("sakura-7b-v1.0")

# 2. ASR: Audio → Japanese text
asr = WhisperASR(config=config)
japanese_segments = asr.transcribe_file("anime_episode.wav")

# 3. Translation: Japanese → Chinese
translator = SakuraTranslator.create_from_config(config)

chinese_segments = []
for segment in japanese_segments:
    result = translator.translate_text(segment.text)
    chinese_segments.append(result.translated_text)

# 4. Generate subtitles
print("Japanese subtitles with Chinese translation ready!")
```

## Performance Optimization

### GPU Memory Management

```python
# Monitor GPU memory
import torch

if torch.cuda.is_available():
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    print(f"GPU Usage: {torch.cuda.memory_allocated() / 1e9:.1f}GB")

# Clear cache when needed
torch.cuda.empty_cache()  # NVIDIA
torch.mps.empty_cache()   # Apple Silicon
```

### Batch Size Tuning

```python
# Start with batch_size=1 for large models
config.set("sakura.batch_size", 1)

# Increase gradually based on VRAM
# More VRAM = larger batch size = faster processing
```

### Temperature Settings

```python
# For consistent translation (recommended)
config.set("sakura.temperature", 0.1)

# For more creative translation
config.set("sakura.temperature", 0.3)

# For deterministic results
config.set("sakura.temperature", 0.0)
```

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```python
# Reduce batch size or use smaller model
config.set("sakura.batch_size", 1)
config.set_sakura_model("sakura-1b8-v1.0")
```

**2. Model Download Failed**
```python
# Check internet connection and HuggingFace access
# Models are downloaded to ~/.cache/huggingface/
# Ensure sufficient disk space (models are 2-20GB)
```

**3. MPS Issues on Apple Silicon**
```python
# Ensure PyTorch 2.0+ with MPS support
# Some operations may fall back to CPU automatically
```

**4. Translation Quality Issues**
```python
# Use larger model for better quality
config.set_sakura_model("sakura-14b-v1.0")

# Enable chat template
config.set("sakura.use_chat_template", True)

# Adjust temperature
config.set("sakura.temperature", 0.1)
```

## Testing Your Setup

Run the demo script to test your SakuraLLM setup:

```bash
python demo_sakura.py
```

Or run specific tests:

```bash
# Test SakuraLLM integration
pytest tests/test_sakura.py -v

# Test with your GPU
./run_tests.py --type gpu --gpu -v
```

## Model Downloads

SakuraLLM models are automatically downloaded from HuggingFace:
- **Location**: `~/.cache/huggingface/hub/`
- **Size**: 2-20GB depending on model
- **Format**: GGUF (optimized for inference)
- **First run**: May take 10-30 minutes to download

## Advanced Configuration

### Custom Model Files

```python
# Use specific quantization
config.set("sakura.model_file", "sakura-7b-qwen2.5-v1.0-q8_0.gguf")  # Higher quality
config.set("sakura.model_file", "sakura-7b-qwen2.5-v1.0-q4_k_s.gguf")  # Smaller size
```

### Generation Parameters

```python
sakura_config = {
    "max_new_tokens": 512,      # Maximum translation length
    "temperature": 0.1,         # Consistency vs creativity
    "top_p": 0.95,             # Nucleus sampling threshold  
    "repetition_penalty": 1.1,  # Prevent repetitive output
    "use_cache": True           # Enable KV caching
}

for key, value in sakura_config.items():
    config.set(f"sakura.{key}", value)
```

## Comparison with Other Models

| Aspect | SakuraLLM | Helsinki-NLP | Generic LLM |
|--------|-----------|--------------|-------------|
| **Quality** | Excellent | Good | Variable |
| **Speed** | Medium | Fast | Slow |
| **VRAM** | 4-32GB | <2GB | 8-80GB |
| **Specialization** | Japanese→Chinese | Multiple pairs | General |
| **Context** | Light novels | General | General |

---

**🌸 SakuraLLM provides the highest quality Japanese to Chinese translation for anime, manga, and light novel content!**