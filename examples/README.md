# 🌸 Sakura Subtitle Generator - Examples

This directory contains example scripts demonstrating various features of the Sakura Subtitle Generator.

## 📋 Available Examples

### 🚀 Core Demos
- **`demo_sakura_translation.py`** - Complete SakuraLLM pipeline (Japanese → Chinese)
- **`demo_sakura_14b_test.py`** - Compare 7B vs 14B SakuraLLM models  
- **`demo_three_languages.py`** - Full pipeline with Helsinki-NLP models
- **`demo_three_languages_quick.py`** - Quick demo showing all outputs

### 🛠️ Utilities
- **`download_sakura_models.py`** - Download SakuraLLM GGUF models
- **`sakura_llm_example.py`** - Basic SakuraLLM usage example

## 🎯 Quick Start

### Test SakuraLLM Pipeline
```bash
# Download model first
uv run python examples/download_sakura_models.py

# Run complete pipeline
uv run python examples/demo_sakura_translation.py
```

### Compare Model Performance
```bash
# Compare 7B vs 14B models
uv run python examples/demo_sakura_14b_test.py
```

### Test Helsinki-NLP Pipeline
```bash
# Three-language pipeline
uv run python examples/demo_three_languages.py
```

## 📊 Model Requirements

- **SakuraLLM 7B**: ~4.3GB VRAM, good quality
- **SakuraLLM 14B**: ~8.5GB VRAM, best quality  
- **Helsinki-NLP**: ~1GB VRAM, basic quality

## 🎌 Features Demonstrated

- ✅ Japanese ASR with kotoba-tech/kotoba-whisper-v2.1
- ✅ SakuraLLM translation (Japanese → Chinese) 
- ✅ Helsinki-NLP translation (Japanese → English → Chinese)
- ✅ Traditional Chinese conversion with OpenCC
- ✅ Apple Silicon Metal acceleration
- ✅ GGUF model support with llama-cpp-python
