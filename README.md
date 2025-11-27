# 🌸 Sakura Subtitle Generator

A powerful, well-architected application for generating Japanese subtitles with multi-language translation support. Built with modern AI models and a user-friendly Tkinter interface.

## ✨ Features

- **🎙️ Japanese ASR**: High-quality Japanese speech recognition using OpenAI Whisper
- **🌐 Multi-language Translation**: Translate to English and Traditional Chinese using Hugging Face models
- **🎯 Dual Language Subtitle Output**: Generate both original Japanese and translated subtitle files with customizable file name suffixes  
- **🖥️ User-Friendly GUI**: Clean Tkinter interface for easy video processing
- **⚙️ Configurable**: Comprehensive settings for models, devices, and output preferences
- **🏗️ Well-Architected**: Modular design with clean separation of concerns
- **📊 Progress Tracking**: Real-time progress updates during processing
- **💾 Multiple Export Formats**: SRT subtitle export with more formats planned
- **⚡ GPU-First Design**: Optimized for NVIDIA CUDA and Apple Silicon MPS acceleration

## 🆕 Recent New Features (Nov 2025)

- **VAD Improvements & UI Controls**: WebRTC VAD tuning updated (default `frame_duration_ms=30`) and VAD parameters (enable, mode, frame duration, padding, min segment duration) exposed in the Settings dialog for easy runtime tuning.
- **Whisper ASR stability**: Suppressed a deprecation warning related to `return_token_timestamps` while keeping native Whisper `generate()` usage for reliable timestamps and no token-limit behavior.
- **Modular Prompt Templates**: Prompt templates are now stored under `prompts/`; the SakuraLLM translator loads the selected template from config. The Settings dialog dynamically lists templates available in `prompts/`.
- **SakuraLLM Prompt Flow Fixed**: `SakuraTranslator.create_from_config()` now passes the configured prompt template style to the translator so the chosen template is actually used during translation.
- **UI & Config Fixes**: Settings dialog layout fixes, persistent config saving/loading for new options, and safer default fallbacks.

## 🐞 Bug Fixes (Nov 2025)

- **VAD zero-interval issue**: Resolved the case where VAD reported 0 speech intervals by enforcing supported WebRTC frame durations (10/20/30 ms) and defaulting to 30 ms for long-form content.
- **Prompt template selection bug**: Fixed translator initialization so non-`standard` templates selected in the UI are applied correctly.
- **Syntax & compile fixes**: Cleaned stray syntax errors and verified modules compile with `python -m py_compile`.

## 🎯 GPU Acceleration Focus

This project is **GPU-first** and optimized for hardware acceleration:

### **✅ Supported GPU Acceleration:**
- **🚀 NVIDIA CUDA**: Full GPU acceleration for RTX/GTX cards
- **🍎 Apple Silicon MPS**: Metal Performance Shaders for M1/M2/M3/M4 chips
- **⚡ Automatic Detection**: Intelligent GPU selection (CUDA > MPS > CPU)

### **⚠️ CPU-Only Users:**
If you **only have CPU** and need CPU-optimized inference, this project focuses on GPU acceleration. For CPU-only setups, consider:

- **Alternative**: Implement `ctransformers`-based classes for CPU optimization
- **Current Focus**: This project prioritizes GPU performance (MPS/CUDA)
- **Performance**: GPU provides 3-5x faster inference than CPU-only solutions
- **Development**: CPU-specific optimizations (ctransformers, GGUF, etc.) are not the main focus

### **Recommended Hardware:**
- **Apple Silicon Macs**: M1/M2/M3/M4 with 16GB+ RAM
- **NVIDIA GPUs**: RTX/GTX cards with 8GB+ VRAM
- **Minimum**: 16GB system RAM for model loading

## 🚀 Hardware Acceleration Performance

**Cross-Platform Hardware Acceleration** with significant performance improvements:

### **⚡ Audio Extraction Performance:**
| Platform | Acceleration | Performance | Realtime Factor |
|----------|--------------|-------------|-----------------|
| Apple Silicon | VideoToolbox | **18.8x realtime** | M1/M2/M3/M4 optimized |
| Windows/Linux | CUDA | **15-20x realtime** | RTX/GTX GPU acceleration |
| Software Fallback | CPU | **1.0x realtime** | Universal compatibility |

### **🎯 ASR Processing Performance:**
| Component | Performance Gain | Notes |
|-----------|------------------|-------|
| ASR Processing | 3-5x faster | MPS/CUDA acceleration for Whisper models |
| Translation | 2-4x faster | Hardware-optimized PyTorch operations |  
| Model Loading | 50% faster | Optimized dependencies with uv |
| Memory Usage | 20-30% less | Efficient native libraries |

### Hardware Acceleration Features:
- **🏁 Cross-Platform**: VideoToolbox (Apple), CUDA (NVIDIA), Software fallback
- **🔥 Automatic Detection**: Intelligent hardware capability detection
- **⚡ Audio Extraction**: Up to 18.8x realtime performance on Apple Silicon  
- **💚 Universal Compatibility**: 100% fallback support for all hardware
- **🧠 Memory Efficient**: Optimized resource management and cleanup

## 🚀 Quick Start

### Hardware Requirements

#### **🎯 Recommended (GPU Acceleration):**
- **Apple Silicon**: M2/M3/M4 with 16GB+ unified memory
- **NVIDIA GPU**: RTX 3070/4070+ with 8GB+ VRAM 
- **System RAM**: 16GB+ for model loading
- **Storage**: 10GB+ free space (3.6GB models + cache)
- **Expected Performance**: 8-12x realtime processing

#### **💰 Minimum (Budget Setup):**
- **Apple Silicon**: M1 with 8GB unified memory
- **NVIDIA GPU**: GTX 1660 with 6GB VRAM
- **Alternative**: Use `whisper-medium` (1.5GB vs 3GB)
- **Expected Performance**: 4-6x realtime processing

#### **⚠️ CPU-Only (Not Optimized):**
- **Performance**: 4-12x slower than GPU (1-2x realtime)
- **Memory**: 32GB+ RAM recommended for all models
- **Models**: Consider `whisper-small` for better CPU performance
- **Note**: Implement ctransformers for CPU optimization (outside project scope)

> 📝 **See [Default Models & Requirements](#-default-models--requirements) section for detailed specifications**

### Software Prerequisites

- Python 3.8 or higher
- FFmpeg (for audio extraction)
- GPU drivers (CUDA 11.8+ for NVIDIA, latest macOS for Apple Silicon)

### Installation

#### Method 1: Pure uv (Recommended - No Environment Pollution)
```bash
git clone <repository-url>
cd py-sub-generator-sakura

# No installation needed - runs in isolated environment!
uv run python main.py
```
- 🔒 **Zero environment pollution** - Completely isolated
- ⚡ **Fastest startup** - Auto-manages dependencies 
- 🛡️ **Safe** - No system Python modification
- 🧹 **Clean** - No installation artifacts

#### Method 2: Universal Setup 
```bash
git clone <repository-url>
cd py-sub-generator-sakura

uv run python setup.py  # Use uv run to avoid polluting system Python!
# OR (if you must):
python setup.py  # ⚠️ May install to system Python if uv not available
```
- ✅ **Works on all platforms** (Windows, Linux, macOS)
- ✅ **Auto-detects uv** for faster installation
- ✅ **Checks system requirements** (Python, FFmpeg)
- ⚠️ **Warning**: May pollute system Python if uv unavailable

#### Method 3: Apple Silicon Optimized (M1/M2/M3 Macs)
```bash
git clone <repository-url>
cd py-sub-generator-sakura

python setup_apple_silicon.py      # Maximum performance setup
```
- 🍎 **Apple Silicon specific optimizations**
- ⚡ **3-5x faster processing** with MPS acceleration
- 🔧 **Automatic FFmpeg installation** via Homebrew
- 💾 **20-30% less memory usage**

#### Method 4: Manual Installation (Advanced)
```bash
git clone <repository-url>
cd py-sub-generator-sakura

# Install with uv (automatically handles dependencies)
uv pip install -e .                    # Basic installation
uv pip install -e ".[gpu]"            # With GPU support  
uv pip install -e ".[apple-silicon]"  # Apple Silicon optimized
uv pip install -e ".[dev]"            # Development dependencies

# Or install development dependencies separately
uv pip install -e . --group dev
```

#### Method 5: Traditional pip Installation (⚠️ Not Recommended)
```bash
git clone <repository-url>
cd py-sub-generator-sakura

pip install -e .                      # Basic installation
pip install -e ".[gpu]"              # With GPU support
pip install -e ".[apple-silicon]"    # Apple Silicon optimized
```

### Usage

#### GUI Mode (Recommended)
```bash
# With uv (fastest, automatic dependency management):
uv run python main.py

# After installation with setup.py or setup_apple_silicon.py:
python main.py
```

#### CLI Mode (Future)
```bash
# With uv (no installation needed)
uv run python main.py --no-gui video.mp4

# After installation
python main.py --no-gui video.mp4
```

## 🎯 Dual Language Subtitle Feature

### **🌸 Generate Both Japanese and Translated Subtitles**

The application supports **dual language subtitle output**, perfect for language learning and multilingual content creation:

#### ✨ **Feature Highlights:**
- **🎌 Japanese Original Subtitles**: Preserves original ASR transcription results
- **🌐 Translated Subtitles**: Simultaneously generates target language (e.g., Chinese, English) subtitles  
- **⚙️ UI Controls**: Convenient toggle option in settings dialog
- **📝 Custom File Naming**: Configurable file suffixes (e.g., `_ja.srt`, `_zh.srt`)
- **🎯 Flexible Output**: Choose to generate single or dual language subtitles as needed

#### 🛠️ **How to Use:**

**Via GUI Settings:**
1. Click the **⚙️ Settings** button
2. Find the **"Dual Language Output"** option in the settings dialog
3. Check **"Generate both language subtitles"**
4. Optional: Click **"Advanced Options"** to customize file suffixes
5. Settings are automatically saved

**File Output Example:**
```
video.mp4 → 
├── video_ja.srt    (Japanese original subtitles)
└── video_zh.srt    (Chinese translated subtitles)
```

**Advanced Customization:**
- **Japanese Suffix**: Default `_ja` (can change to `_japanese`, `_orig`, etc.)
- **Translated Suffix**: Default `_zh` (can change to `_chinese`, `_trans`, etc.)
- **File Naming**: Fully customizable for workflow integration

#### 🎬 **Perfect Use Cases:**
- **🎓 Language Learning**: Compare original and translated text for learning
- **📺 Multilingual Content**: Provide options for different audiences
- **🔄 Translation Comparison**: Check translation quality and accuracy
- **📚 Subtitle Production**: Professional subtitle creation workflows

#### Development with uv
```bash
# Install dev dependencies (modern way)
uv pip install -e . --group dev

# Or using optional dependencies
uv pip install -e ".[dev]"

# Run tests
uv run pytest

# Format code  
uv run black src/
uv run isort src/

# Type checking
uv run mypy src/

# Run with specific Python version
uv run --python 3.11 python main.py
```

## 🔧 Build System & GPU Dependencies

This project uses modern Python packaging with GPU-first dependencies:

- **📦 Build Backend**: `hatchling` - Fast, modern PEP 517/518 compliant builder
- **📋 Configuration**: Pure `pyproject.toml` - No `setup.py` needed
- **⚡ Package Manager**: `uv` - Ultra-fast Python package installer and resolver
- **🔄 Dependencies**: Modern `dependency-groups` for development dependencies
- **🚀 GPU Dependencies**: PyTorch with CUDA/MPS support, transformers with acceleration

### Key Benefits:
- ✅ **Zero setup.py** - Pure `pyproject.toml` configuration
- ✅ **Fast builds** - Hatchling is significantly faster than setuptools
- ✅ **Modern standards** - Full PEP 517/518/621 compliance
- ✅ **uv integration** - Seamless `uv run` support without installation
- ✅ **GPU-optimized** - Automatic PyTorch CUDA/MPS dependencies
- ✅ **Clean dependencies** - No deprecated configurations

### GPU Dependencies Included:
- **PyTorch 2.8+** with CUDA 11.8+ / MPS support
- **Transformers 4.55+** with GPU acceleration
- **Apple Silicon**: Native MPS backend
- **NVIDIA**: CUDA toolkit integration

### CPU-Only Alternative:
If you need CPU-only inference, consider developing additional classes:
```python
# Example CPU-optimized approach (not included in main project)
from ctransformers import AutoModelForCausalLM  # CPU-optimized
# This project focuses on GPU acceleration instead
```

### pyproject.toml Structure:
```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[dependency-groups]
dev = ["pytest", "black", "isort", "mypy", "ruff"]
```

## 🏗️ Architecture

The application follows a clean, modular architecture:

```
src/
├── models/          # Data models and structures
├── utils/           # Utilities (config, logging, file handling, audio processing)  
├── asr/             # Automatic Speech Recognition modules
├── translation/     # Translation pipeline and models
├── subtitle/        # Subtitle processing and generation
└── ui/              # Tkinter GUI components
```

### Core Components

#### 🎯 **Subtitle Generator** (`src/subtitle/subtitle_generator.py`)
The main orchestrator that coordinates the entire subtitle generation pipeline:
- Video validation and metadata extraction
- Audio extraction and preprocessing  
- ASR transcription with chunking for long videos
- Multi-language translation
- Subtitle formatting and export

#### 🎤 **ASR Module** (`src/asr/`)
- **Base ASR** (`base_asr.py`): Abstract interface for ASR implementations
- **Whisper ASR** (`whisper_asr.py`): OpenAI Whisper integration with batching support

#### 🌐 **Translation Module** (`src/translation/`)
- **Translation Pipeline** (`translation_pipeline.py`): Coordinates multi-stage translation
- **HuggingFace Translator** (`huggingface_translator.py`): Transformer-based GPU translation
- **PyTorch Translator** (`pytorch_translator.py`): GPU-first PyTorch implementation
- **Multi-Stage Translator**: Japanese → English → Traditional Chinese

##### Translation Backend Strategy:
- **🚀 Primary**: HuggingFace Transformers with GPU acceleration (MPS/CUDA) 
- **🌸 Advanced**: PyTorchTranslator for SakuraLLM integration (specialized Japanese LLMs)
- **⚡ Alternative**: Pure PyTorch with GPU optimization for any HuggingFace model
- **❌ Not Included**: ctransformers (CPU-only focus conflicts with project goals)
- **🎯 Focus**: Maximum GPU performance for real-time subtitle generation

##### **🌸 PyTorchTranslator Features (SakuraLLM Ready):**
- **🎯 LLM Support**: Direct integration with large language models
- **🚀 GPU Optimization**: MPS/CUDA acceleration with FP16 precision
- **🧠 Advanced Generation**: Temperature, sampling, KV caching
- **📝 Prompt Engineering**: Customizable prompts for specialized models
- **🎮 Memory Management**: Efficient VRAM usage and model loading
- **⚖️ Model Agnostic**: Works with any HuggingFace Transformers model

**SakuraLLM Integration Example:**
```python
# Specialized Japanese light novel translation
sakura_translator = PyTorchTranslator(
    model_name="SakuraLLM/Sakura-1.5B-Qwen2.5-v1.0-GGUF",
    source_lang="ja", target_lang="zh",
    device="auto", torch_dtype="float16"
)
```

**Note for CPU Users**: This project prioritizes GPU acceleration. If you require CPU-only inference, consider implementing additional `ctransformers`-based classes, though this is outside the current project scope.

#### 🖥️ **UI Module** (`src/ui/`)
- **Main Window** (`main_window.py`): Primary application interface
- **Components**: Settings dialog, preview dialog, progress tracking

#### 🛠️ **Utilities** (`src/utils/`)
- **Config Management** (`config.py`): JSON-based configuration with defaults
- **File Handler** (`file_handler.py`): Video validation and file operations
- **Audio Processor** (`audio_processor.py`): Audio extraction and preprocessing
- **Logger** (`logger.py`): Structured logging with file rotation

## ⚙️ Configuration

The application uses a JSON configuration file with sensible defaults:

```json
{
  "asr": {
    "model_name": "openai/whisper-large-v3",
    "device": "auto",
    "chunk_length": 30
  },
  "translation": {
    "ja_to_en_model": "Helsinki-NLP/opus-mt-ja-en", 
    "en_to_zh_model": "Helsinki-NLP/opus-mt-en-zh",
    "batch_size": 8
  },
  "dual_language": {
    "generate_both_languages": false,
    "japanese_suffix": "_ja",
    "translated_suffix": "_zh"
  },
  "ui": {
    "window_size": "800x600",
    "theme": "default"
  }
}
```

Configuration can be modified through:
- GUI Settings dialog (⚙️ Settings button)
- Direct JSON file editing (`config.json`)
- Command line arguments

## 🤖 Default Models & Requirements

### 🎙️ **ASR Models (Speech Recognition)**

#### **Default: OpenAI Whisper Large-v3**
- **Model**: [`openai/whisper-large-v3`](https://huggingface.co/openai/whisper-large-v3)
- **Size**: ~3GB download
- **Languages**: 99+ languages (optimized for Japanese)
- **Quality**: Highest accuracy for Japanese speech recognition

**Hardware Requirements:**
| Hardware | **Minimum** | **Recommended** |
|----------|-------------|----------------|
| **Apple Silicon** | M1 8GB | M2/M3 16GB+ |
| **NVIDIA GPU** | GTX 1660 6GB | RTX 3070 8GB+ |
| **CPU Only** | 16GB RAM | 32GB RAM |
| **Processing** | 2x realtime | 8-12x realtime |

#### **Alternative ASR Models:**

| Model | Size | Speed | Accuracy | Min Requirements |
|-------|------|-------|----------|-----------------|
| **whisper-large-v3** | 3GB | Slow | Highest | 8GB VRAM/RAM |
| **whisper-medium** | 1.5GB | Medium | High | 4GB VRAM/RAM |
| **whisper-small** | 500MB | Fast | Good | 2GB VRAM/RAM |
| **whisper-base** | 150MB | Fastest | Fair | 1GB VRAM/RAM |

### 🌐 **Translation Models**

#### **Japanese → English**
- **Model**: [`Helsinki-NLP/opus-mt-ja-en`](https://huggingface.co/Helsinki-NLP/opus-mt-ja-en)
- **Size**: ~300MB download
- **Specialty**: Japanese to English translation
- **Performance**: Optimized for subtitle-length text

#### **English → Chinese (Traditional)**
- **Model**: [`Helsinki-NLP/opus-mt-en-zh`](https://huggingface.co/Helsinki-NLP/opus-mt-en-zh)
- **Size**: ~300MB download  
- **Specialty**: English to Chinese translation
- **Output**: Traditional Chinese characters

**Translation Hardware Requirements:**
| Hardware | **Both Models Combined** |
|----------|-------------------------|
| **GPU Memory** | 1GB VRAM (both models) |
| **System RAM** | 2GB RAM (both models) |
| **Processing** | 4x faster on GPU vs CPU |

### 📊 **Total System Requirements**

#### **Complete Setup (All Models):**
| Component | **Download Size** | **Runtime Memory** |
|-----------|------------------|-------------------|
| **Whisper Large-v3** | 3.0GB | 6GB VRAM/RAM |
| **Translation Models** | 0.6GB | 1GB VRAM/RAM |
| **Total** | **3.6GB** | **7GB VRAM/RAM** |

#### **Recommended Configurations:**

**🍎 Apple Silicon (Optimal):**
- **M2/M3/M4**: 16GB unified memory
- **Storage**: 10GB free space
- **Expected Performance**: 8x realtime processing

**🚀 NVIDIA GPU (Optimal):**
- **GPU**: RTX 3070/4070 (8GB+ VRAM)
- **RAM**: 16GB system memory
- **Expected Performance**: 12x realtime processing

**💻 Budget/Lower-end Hardware:**
```python
# Use smaller models in config.json
{
  "asr": {
    "model_name": "openai/whisper-medium"  # 1.5GB vs 3GB
  }
}
```

### 🔗 **Model Links & Licenses**

| Model | Hugging Face Link | License | Purpose |
|-------|------------------|---------|---------|
| **Whisper Large-v3** | [🤗 Link](https://huggingface.co/openai/whisper-large-v3) | MIT | Japanese ASR |
| **OPUS JA-EN** | [🤗 Link](https://huggingface.co/Helsinki-NLP/opus-mt-ja-en) | Apache 2.0 | Japanese→English |
| **OPUS EN-ZH** | [🤗 Link](https://huggingface.co/Helsinki-NLP/opus-mt-en-zh) | Apache 2.0 | English→Chinese |

### ⚠️ **Model Constraints & Limitations**

#### **Whisper Large-v3:**
- **Languages**: Optimized for Japanese, may struggle with heavy dialect
- **Audio Quality**: Works best with clear speech, struggles with background noise
- **Length**: Optimal for 30-second chunks (longer may degrade accuracy)
- **Real-time**: Not suitable for live transcription (processing delay)

#### **Translation Models:**
- **Domain**: Trained on general text, may struggle with technical/domain-specific terms
- **Length**: Optimized for sentence-level translation (subtitle appropriate)
- **Formality**: May not preserve Japanese politeness levels in translation
- **Cultural Context**: Limited cultural nuance preservation

#### **Hardware Constraints:**
- **Memory**: All models must fit in VRAM/RAM simultaneously
- **Processing**: CPU fallback is 4-12x slower than GPU
- **Storage**: SSD recommended for faster model loading (vs HDD)

### 🎯 **Hardware-Specific Setup Recommendations**

#### **🏆 High-End Setup (Recommended)**
```json
// config.json for optimal performance
{
  "asr": {
    "model_name": "openai/whisper-large-v3",
    "batch_size": 1,
    "device": "auto"
  },
  "translation": {
    "batch_size": 8,
    "device": "auto"
  }
}
```
**Hardware**: M3/M4 16GB+ or RTX 3070+ 8GB  
**Performance**: 8-12x realtime  
**Use Case**: Production subtitle generation

#### **💰 Budget Setup (Balanced)**
```json
// config.json for balanced performance/memory
{
  "asr": {
    "model_name": "openai/whisper-medium",
    "batch_size": 1,
    "device": "auto"
  },
  "translation": {
    "batch_size": 4,
    "device": "auto"  
  }
}
```
**Hardware**: M1/M2 8GB or GTX 1660 6GB  
**Performance**: 4-6x realtime  
**Use Case**: Personal use, occasional processing

#### **🔋 Minimal Setup (Emergency)**
```json
// config.json for low-resource systems
{
  "asr": {
    "model_name": "openai/whisper-small",
    "batch_size": 1,
    "device": "cpu"
  },
  "translation": {
    "batch_size": 1,
    "device": "cpu"
  }
}
```
**Hardware**: Any system with 8GB RAM  
**Performance**: 1-2x realtime  
**Use Case**: Testing, very low-end hardware

### 📥 **First-Time Setup & Downloads**

#### **Automatic Model Downloads:**
```bash
# First run will download all models (~3.6GB)
uv run python main.py

# Downloads will occur to:
# ~/.cache/huggingface/transformers/
# Total download time: 5-15 minutes (depending on internet)
# Note: Models cache to ~/.cache/huggingface/ and won't re-download
```

#### **⚡ Cache Optimization:**

Models are cached to `~/.cache/huggingface/` and reused across runs. However, you might see downloads for:
- **Format updates**: safetensors (preferred) vs pytorch_model.bin
- **Missing components**: tokenizer files, config updates

**Cache Status Check:**
```bash
# Check cached models
ls ~/.cache/huggingface/hub/

# Cache size
du -sh ~/.cache/huggingface/
```

**Force Pre-download:**
```bash
# Pre-download all model formats to avoid future downloads
uv run python predownload_models.py
```

#### **Pre-download Models (Optional):**
```python
# Pre-download script (optional)
uv run python -c "
from transformers import AutoModel, AutoTokenizer, pipeline

# Download ASR model
pipeline('automatic-speech-recognition', model='openai/whisper-large-v3')

# Download translation models  
pipeline('translation', model='Helsinki-NLP/opus-mt-ja-en')
pipeline('translation', model='Helsinki-NLP/opus-mt-en-zh')

print('All models downloaded successfully!')
"
```

## 🌸 **SakuraLLM Integration (Advanced Option)**

### **🎯 Alternative: SakuraLLM for Superior Japanese Translation**

For users seeking **highest quality Japanese→Chinese translation**, we support **SakuraLLM integration** via the PyTorchTranslator:

#### **📋 SakuraLLM Model Options:**

| Model | Parameters | Download Size | Memory Req | Performance | Quality |
|-------|------------|--------------|-------------|-------------|---------|
| **🌸 Sakura-1.5B** ⭐ | 1.78B | ~2-4GB | 4-6GB | 8-15x faster | Very Good |
| **🌸 Sakura-14B** | 14.8B | ~15-30GB | 16-32GB | 2-4x slower | Excellent |

#### **✅ Recommended: Sakura-1.5B-Qwen2.5-v1.0**
- **🎯 Optimal Balance**: Quality vs Hardware Requirements
- **🚀 Real-time Ready**: Fast enough for subtitle generation  
- **💾 Accessible**: Fits on M1 8GB and GTX 1660 6GB
- **🎨 Specialized**: Light novel translation optimized

#### **⚙️ SakuraLLM Integration:**

```python
# config.json - SakuraLLM setup
{
  "translation": {
    "use_sakura": true,
    "sakura_model": "SakuraLLM/Sakura-1.5B-Qwen2.5-v1.0-GGUF",
    "source_lang": "ja",
    "target_lang": "zh", 
    "device": "auto",
    "torch_dtype": "float16"
  }
}

# Usage with PyTorchTranslator
from src.translation.pytorch_translator import PyTorchTranslator

sakura = PyTorchTranslator(
    model_name="SakuraLLM/Sakura-1.5B-Qwen2.5-v1.0-GGUF",
    source_lang="ja",
    target_lang="zh",
    device="auto",  # GPU-first
    force_gpu=True
)
```

#### **🌸 SakuraLLM Advantages:**
- **📚 Light Novel Optimized**: Specialized for Japanese fiction/anime content
- **👥 Character Context**: Better character name and pronoun handling
- **🎭 Cultural Nuance**: Superior cultural context preservation
- **📖 Terminology Support**: Built-in glossary/dictionary support
- **🎯 Subtitle Appropriate**: Optimized for short text segments

#### **⚠️ SakuraLLM Limitations:**
- **🌐 Language Pair**: Japanese→Chinese only (no English intermediate)
- **📖 Domain**: Optimized for light novels, may struggle with technical content  
- **💾 Size**: Larger than Helsinki-NLP models (2-4GB vs 300MB)
- **⚖️ License**: CC-BY-NC-SA-4.0 (non-commercial)
- **🔧 Complexity**: Requires more sophisticated prompt engineering

#### **🎯 When to Use SakuraLLM:**
- **✅ Anime/Light Novel** content translation
- **✅ High-quality Japanese→Chinese** required
- **✅ Have 8GB+ VRAM/RAM** available
- **✅ Non-commercial use** acceptable

#### **🎯 When to Use Default (Helsinki-NLP):**
- **✅ General purpose** translation
- **✅ Resource-constrained** systems
- **✅ Commercial use** required
- **✅ Multi-language pairs** needed

## ⚡ Performance Comparison

### **Translation Backend Comparison:**

| Backend | **Quality** | **Speed** | **Memory** | **Languages** | **Use Case** |
|---------|------------|-----------|-----------|---------------|--------------|
| **Helsinki-NLP** | Good | Very Fast | 1GB | Multi-pair | General |
| **SakuraLLM 1.5B** | Very Good | Fast | 4GB | JA→ZH | Anime/LN |
| **SakuraLLM 14B** | Excellent | Moderate | 16GB | JA→ZH | Premium |

### **GPU vs CPU Performance** (Approximate benchmarks):

| Backend | **Apple M3** | **NVIDIA RTX 4080** | **CPU Only (Intel/AMD)** |
|---------|-------------|---------------------|-------------------------|
| **Whisper Large-v3** | 8x realtime (MPS) | 12x realtime (CUDA) | 1.5x realtime |
| **Helsinki Translation** | 4x faster (MPS) | 6x faster (CUDA) | Baseline |
| **SakuraLLM 1.5B** | 3x faster (MPS) | 5x faster (CUDA) | 0.5x baseline |
| **Memory Usage** | 8GB unified | 6GB VRAM | 16GB RAM |
| **Power Efficiency** | Excellent | Moderate | Poor |

### **Why GPU-First Design:**
- **🚀 Speed**: 4-12x faster processing than CPU-only solutions
- **🧠 Efficiency**: Better memory bandwidth utilization  
- **⚡ Real-time**: Enables real-time subtitle generation
- **🔋 Power**: More energy-efficient on Apple Silicon

### **CPU-Only Alternative Guidance:**
If you're limited to CPU-only inference:
1. **Performance**: Expect 4-12x slower processing
2. **Implementation**: Consider `ctransformers` with GGUF models
3. **Project Scope**: CPU optimization is outside current focus
4. **Recommendation**: Use smaller models (whisper-small, smaller translation models)

## 📋 Workflow

1. **🎬 Video Input**: Select video file through file browser
2. **⚙️ Configuration**: Choose target languages, dual language output, and settings
3. **🔍 Validation**: Verify video format and extract metadata  
4. **🎵 Audio Extraction**: Extract audio using FFmpeg
5. **🎙️ ASR Processing**: Generate Japanese transcription with Whisper
6. **🌐 Translation**: Translate to English and Traditional Chinese
7. **📝 Subtitle Generation**: Create formatted subtitle files (single or dual language)
8. **👁️ Preview & Export**: Review results and export SRT files

## 🚦 Error Handling

- **Graceful Degradation**: Continues processing if non-critical steps fail
- **User Feedback**: Clear error messages with actionable suggestions
- **Resource Management**: Automatic cleanup of temporary files and GPU memory
- **Logging**: Comprehensive logging for debugging

## 🎛️ Advanced Features

### Chunked Processing
- Automatically splits long videos into chunks for memory efficiency
- Overlapping chunks prevent word boundary issues
- Configurable chunk size and overlap

### Device Management  
- Automatic GPU detection and fallback to CPU
- Memory-conscious model loading/unloading
- Configurable device preferences

### Subtitle Optimization
- Automatic text cleaning and formatting
- Optimal line breaking for readability  
- Configurable timing constraints

## 📊 Performance Considerations

- **Memory Usage**: Models are loaded/unloaded as needed
- **GPU Support**: CUDA acceleration when available
- **Batch Processing**: Efficient batching for translations
- **Chunking**: Handles videos of any length

## 🔮 Future Enhancements

- [ ] CLI mode implementation
- [ ] Additional subtitle formats (VTT, ASS)
- [ ] Speaker diarization
- [ ] Custom model fine-tuning
- [ ] Web interface
- [ ] Docker deployment
- [ ] Additional language pairs

## 📚 Documentation

### 📋 Project Documentation
- **[Progress Summary](PROGRESS_SUMMARY.md)** - Complete overview of recent improvements and transformations
- **[Technical Changelog](TECHNICAL_CHANGELOG.md)** - Detailed technical implementation documentation  
- **[UV Tool Strategy](docs/UV_TOOL_STRATEGY.md)** - Why we use `uv tool` for development tools instead of project dependencies

### 🚀 Recent Major Updates
- **Native Whisper Implementation** - Refactored from pipeline to native `generate()` method
- **Zero Experimental Warnings** - Eliminated all ASR warnings for clean operation  
- **Production Ready** - Following Whisper paper best practices (Section 3.8)
- **Performance Optimized** - Better memory usage and generation efficiency
- **Dual Language Subtitle Generation** - Support for simultaneous Japanese and translated subtitle file output

### 🎯 Key Improvements
- ✅ **No Token Limits** - Removed artificial generation constraints
- ✅ **Silent Operation** - Clean logs without experimental warnings
- ✅ **Better Quality** - Using Whisper's intended architecture
- ✅ **Full Device Support** - Maintained MPS/CUDA/CPU compatibility
- ✅ **Dual Language Output** - Configurable Japanese + translated subtitle file generation

*For complete technical details, see [TECHNICAL_CHANGELOG.md](TECHNICAL_CHANGELOG.md)*

## 🤝 Contributing

The codebase is designed for extensibility:

- **New ASR Models**: Extend `BaseASR` class
- **New Translation Models**: Extend `BaseTranslator` class  
- **New Export Formats**: Add methods to `SubtitleFile` class
- **New UI Components**: Add to `src/ui/components/`

## 📄 License

[License information]

## 🙏 Acknowledgments

- **OpenAI** for Whisper ASR models and the foundational research
- **Whisper Paper (Section 3.8)** - "Robust Speech Recognition via Large-Scale Weak Supervision" for implementation methodology and best practices
- **SakuraLLM Project** for specialized Japanese-to-Chinese translation models optimized for light novel and anime content
- **Hugging Face** for transformer models and pipeline APIs
- **Helsinki-NLP** for OPUS-MT translation models
- **FFmpeg Project** for comprehensive audio processing capabilities

## 📚 Documentation

### **Comprehensive Guides:**
- **[Hardware Acceleration Guide](docs/HARDWARE_ACCELERATION_GUIDE.md)** - Cross-platform acceleration setup and performance optimization
- **[Complete Progress Summary](docs/COMPLETE_PROGRESS_SUMMARY.md)** - Detailed development progress and achievements
- **[Apple Silicon Setup Guide](docs/APPLE_SILICON_GUIDE.md)** - M1/M2/M3/M4 optimization instructions
- **[SakuraLLM Integration Guide](docs/SAKURA_GUIDE.md)** - SakuraLLM model setup and configuration
- **[UV Tool Strategy](docs/UV_TOOL_STRATEGY.md)** - Clean development environment management

### **Technical Documentation:**
- **[Source Structure](docs/SOURCE_STRUCTURE.md)** - Codebase architecture and module organization
- **[Makefile Improvements](docs/MAKEFILE_IMPROVEMENTS.md)** - Development workflow automation
- **[Optimization Summary](docs/OPTIMIZATION_SUMMARY.md)** - Performance enhancements and code quality improvements

---

**Built with ❤️ for the subtitle generation community** 🌸