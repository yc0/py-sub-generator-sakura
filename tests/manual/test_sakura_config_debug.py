#!/usr/bin/env python3
"""
Manual test/debugging tool for SakuraLLM configuration and model loading
This is intended for interactive debugging and manual verification.
"""
import logging
from pathlib import Path

from src.utils.config import Config
from src.translation.sakura_translator_llama_cpp import SakuraTranslator

# Set up logging
logging.basicConfig(level=logging.INFO)


def test_sakura_config():
    """Test SakuraLLM configuration and loading."""
    
    print("🧪 Testing SakuraLLM Configuration")
    print("=" * 50)
    
    # Load config
    config = Config()
    
    print("📋 Configuration Check:")
    print(f"   Sakura enabled: {config.is_sakura_enabled()}")
    
    sakura_config = config.get_sakura_config()
    print(f"   Model name: {sakura_config.get('model_name')}")
    print(f"   Model file: {sakura_config.get('model_file')}")
    print(f"   Device: {sakura_config.get('device')}")
    
    try:
        print("\n🌸 Initializing SakuraLLM Translator...")
        translator = SakuraTranslator.create_from_config(config)
        print("   ✅ Translator initialized")
        
        # Try to load model
        print("\n📦 Loading SakuraLLM model...")
        success = translator.load_model()
        
        if success:
            print("   ✅ Model loaded successfully!")
            
            # Test translation
            print("\n🔤 Testing translation...")
            test_text = "こんにちは"
            result = translator.translate_text(test_text)
            
            print(f"   Original: {test_text}")
            print(f"   Translation: {result.translated_text}")
            print(f"   Confidence: {result.confidence}")
            
        else:
            print("   ❌ Model loading failed")
            print("   💡 This is expected if GGUF model files are not downloaded")
            
    except Exception as e:
        print(f"   💥 Error: {e}")
        print(f"   Traceback: {e.__class__.__name__}: {e}")
        
        if "not enabled" in str(e):
            print("\n💡 To enable SakuraLLM:")
            print('   1. Edit config.json')
            print('   2. Set "sakura.enabled": true')
            print('   3. Ensure model files are downloaded')


def test_available_models():
    """Test available SakuraLLM model configurations."""
    
    print("\n🎯 Available SakuraLLM Models:")
    print("=" * 50)
    
    config = Config()
    models = config.get_available_sakura_models()
    
    if not models:
        print("   ❌ No SakuraLLM models configured")
        return
    
    for model_key, model_info in models.items():
        print(f"   📦 {model_key}:")
        print(f"      Model: {model_info.get('model_name')}")
        print(f"      File:  {model_info.get('model_file')}")
        print(f"      Size:  {model_info.get('size', 'Unknown')}")
        print()


def test_model_recommendations():
    """Test model recommendations."""
    
    print("\n💡 Model Recommendations:")
    print("=" * 50)
    
    try:
        recommendations = SakuraTranslator.get_recommended_models()
        
        for category, info in recommendations.items():
            print(f"   🏷️  {category.replace('_', ' ').title()}:")
            print(f"      Model: {info.get('model_name')}")
            print(f"      VRAM:  {info.get('vram_required')}")
            print(f"      Perf:  {info.get('performance')}")
            print(f"      Desc:  {info.get('description')}")
            print()
            
    except Exception as e:
        print(f"   ❌ Error getting recommendations: {e}")


def main():
    """Main function for manual testing."""
    
    print("🌸 SakuraLLM Configuration and Model Testing Tool")
    print("=" * 60)
    print("This tool helps debug SakuraLLM configuration and model loading issues.")
    print("Run this interactively to diagnose problems.")
    print()
    
    # Test basic configuration
    test_sakura_config()
    
    # Test available models  
    test_available_models()
    
    # Test recommendations
    test_model_recommendations()
    
    print("\n🏁 Testing completed")
    print("\n💡 Tips:")
    print("   - If models fail to load, check if GGUF files are downloaded")
    print("   - Enable SakuraLLM in config.json: 'sakura.enabled': true") 
    print("   - Check model file paths in configuration")


if __name__ == "__main__":
    main()