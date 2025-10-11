# 🏗️ Source Code Structure

## 📁 Directory Organization

```
src/
├── asr/                    # Automatic Speech Recognition
│   ├── base_asr.py        # Abstract ASR base class
│   ├── whisper_asr.py     # Whisper implementation
│   └── __init__.py
├── translation/            # Translation engines
│   ├── base_translator.py           # Abstract translator base
│   ├── huggingface_translator.py    # Helsinki-NLP models
│   ├── sakura_translator_llama_cpp.py # SakuraLLM with llama-cpp
│   ├── translation_pipeline.py      # Pipeline orchestration
│   └── __init__.py
├── models/                 # Data models
│   ├── subtitle_data.py   # Subtitle and translation models
│   ├── video_data.py      # Audio/video data models
│   └── __init__.py
├── utils/                  # Utilities
│   ├── audio_processor.py # Audio file handling
│   ├── chinese_converter.py # OpenCC integration
│   ├── config.py          # Configuration management
│   ├── file_handler.py    # File I/O utilities
│   ├── logger.py          # Logging setup
│   └── __init__.py
├── subtitle/               # Subtitle processing
│   ├── subtitle_generator.py # SRT generation
│   ├── subtitle_processor.py # Subtitle manipulation
│   └── __init__.py
├── ui/                     # User interface (GUI)
│   ├── components/        # UI components
│   ├── main_window.py     # Main GUI window
│   └── __init__.py
└── __init__.py
```

## 🎯 Key Components

### ASR Module (`src/asr/`)
- **Purpose**: Convert audio to Japanese text
- **Implementation**: Whisper-based with kotoba-tech models
- **Features**: Apple Silicon optimization, batch processing

### Translation Module (`src/translation/`)
- **Purpose**: Translate Japanese text to Chinese
- **Implementations**: 
  - SakuraLLM (best quality, GGUF models)
  - Helsinki-NLP (basic quality, transformer models)
- **Features**: Batch translation, progress callbacks

### Models Module (`src/models/`)
- **Purpose**: Data structures and type definitions
- **Classes**: SubtitleSegment, TranslationResult, AudioData
- **Features**: Type safety, serialization support

### Utils Module (`src/utils/`)
- **Purpose**: Cross-cutting utilities
- **Features**: Config management, file handling, Chinese conversion

## 🔧 Architecture Principles

1. **Modular Design**: Each component has clear responsibilities
2. **Abstract Interfaces**: Easy to add new ASR/translation engines
3. **Type Safety**: Comprehensive data models with validation
4. **Configuration Driven**: JSON-based configuration system
5. **Testing Focused**: Comprehensive test coverage
