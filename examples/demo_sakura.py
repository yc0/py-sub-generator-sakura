#!/usr/bin/env uv run python
"""
🌸 SakuraLLM Example Script

This script demonstrates how to use SakuraLLM for high-quality Japanese to Chinese translation.
Run with: uv run python demo_sakura.py
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import logging

from src.translation.sakura_translator import SakuraTranslator
from src.utils.config import Config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demo_sakura_config():
    """Demonstrate SakuraLLM configuration options."""
    print("🌸 SakuraLLM Configuration Demo")
    print("=" * 50)

    # Create config
    config = Config()

    # Show available SakuraLLM models
    print("Available SakuraLLM Models:")
    models = config.get_available_sakura_models()
    for key, info in models.items():
        print(f"  {key}:")
        print(f"    Description: {info.get('description')}")
        print(f"    VRAM Required: {info.get('vram_required')}")
        print(f"    Model: {info.get('model_name')}")
        print()

    # Show current configuration
    print("Current SakuraLLM Configuration:")
    sakura_config = config.get_sakura_config()
    for key, value in sakura_config.items():
        if key != "available_models":  # Skip the large nested dict
            print(f"  {key}: {value}")
    print()

    # Show if SakuraLLM is enabled
    print(f"SakuraLLM Enabled: {config.is_sakura_enabled()}")
    print()


def demo_sakura_translation():
    """Demonstrate SakuraLLM translation."""
    print("🌸 SakuraLLM Translation Demo")
    print("=" * 50)

    # Sample Japanese texts (light novel style)
    japanese_texts = [
        "こんにちは、元気ですか？",
        "今日はとても美しい桜が咲いています。",
        "彼女は微笑みながら私に手を振った。",
        "この小説はとても面白いストーリーです。",
        "「ありがとうございます」と彼女は言いました。"
    ]

    try:
        # Create config and enable SakuraLLM
        config = Config()
        config.set("sakura.enabled", True)

        print("Available SakuraLLM models:")
        recommendations = SakuraTranslator.get_recommended_models()
        for category, info in recommendations.items():
            print(f"  {category}: {info['model_key']} ({info['vram_required']})")
        print()

        # Select model based on available VRAM (use smallest for demo)
        model_key = "sakura-1b8-v1.0"  # Smallest model for demo

        print(f"Using model: {model_key}")
        print("Note: This is a demo - model may not actually load without proper setup")
        print()

        # Create SakuraTranslator (may fail if model not available)
        try:
            translator = SakuraTranslator.create_from_config(config, model_key=model_key)
            print("✅ SakuraTranslator created successfully!")

            # Show model info
            model_info = translator.get_model_info()
            print("Model Information:")
            for key, value in model_info.items():
                print(f"  {key}: {value}")
            print()

            # Try translation (this will likely fail without actual model)
            print("Attempting translations...")
            for i, text in enumerate(japanese_texts[:2]):  # Just try first 2
                print(f"🇯🇵 Japanese: {text}")
                try:
                    result = translator.translate_text(text)
                    print(f"🇨🇳 Chinese: {result.translated_text}")
                    print(f"   Confidence: {result.confidence}")
                    print()
                except Exception as e:
                    print(f"   ❌ Translation failed: {e}")
                    print()

        except Exception as e:
            print(f"❌ Could not create SakuraTranslator: {e}")
            print("This is expected if SakuraLLM models are not downloaded")
            print()

    except Exception as e:
        print(f"❌ Demo failed: {e}")


def demo_model_selection():
    """Demonstrate model selection based on hardware."""
    print("🌸 SakuraLLM Model Selection Demo")
    print("=" * 50)

    config = Config()

    # Show model recommendations
    recommendations = SakuraTranslator.get_recommended_models()

    print("SakuraLLM Model Recommendations:")
    print()
    for category, info in recommendations.items():
        print(f"{category.replace('_', ' ').title()}:")
        print(f"  Model: {info['model_key']}")
        print(f"  VRAM: {info['vram_required']}")
        print(f"  Description: {info['description']}")
        print(f"  Performance: {info['performance']}")
        print()

    # Show how to set a specific model
    print("Setting different models:")
    for model_key in ["sakura-1b8-v1.0", "sakura-7b-v1.0"]:
        success = config.set_sakura_model(model_key)
        if success:
            current_model = config.get("sakura.model_name")
            current_file = config.get("sakura.model_file")
            print(f"✅ Set to {model_key}:")
            print(f"   Model: {current_model}")
            print(f"   File: {current_file}")
        else:
            print(f"❌ Failed to set {model_key}")
        print()


def main():
    """Main demo function."""
    print("🌸 SakuraLLM Integration Demo")
    print("=" * 60)
    print()

    try:
        # Demo 1: Configuration
        demo_sakura_config()

        # Demo 2: Model selection
        demo_model_selection()

        # Demo 3: Translation (may fail without models)
        demo_sakura_translation()

        print("🌸 Demo completed!")
        print()
        print("To use SakuraLLM:")
        print("1. Enable in config: config.set('sakura.enabled', True)")
        print("2. Select model: config.set_sakura_model('sakura-1b8-v1.0')")
        print("3. Create translator: SakuraTranslator.create_from_config(config)")
        print("4. Translate: translator.translate_text('こんにちは')")

    except Exception as e:
        logger.error(f"Demo failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
