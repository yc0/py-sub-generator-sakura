#!/usr/bin/env python3
"""
🌸 SakuraLLM Integration Example

This script demonstrates how to use SakuraLLM for high-quality 
Japanese→Chinese translation in the Sakura Subtitle Generator.
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.translation.pytorch_translator import PyTorchTranslator
from src.utils.logger import setup_logger

def main():
    """Demonstrate SakuraLLM translation."""
    
    # Setup logging
    logger = setup_logger("sakura_example")
    
    # Japanese text samples (typical anime/light novel content)
    test_texts = [
        "こんにちは、元気ですか？",
        "今日はいい天気ですね。",
        "アニメを見るのが好きです。",
        "この小説はとても面白いです。",
        "彼は学校に行きました。"
    ]
    
    print("🌸 SakuraLLM Translation Example")
    print("=" * 50)
    
    try:
        # Initialize SakuraLLM translator
        print("Initializing SakuraLLM translator...")
        translator = PyTorchTranslator(
            model_name="SakuraLLM/Sakura-1.5B-Qwen2.5-v1.0-GGUF", 
            source_lang="ja",
            target_lang="zh",
            device="auto",  # Auto-detect GPU
            torch_dtype="float16",
            force_gpu=True,
            batch_size=4
        )
        
        # Load model
        print("Loading model (this may take a few minutes for first download)...")
        if not translator.load_model():
            print("❌ Failed to load SakuraLLM model")
            return
            
        print("✅ SakuraLLM loaded successfully!")
        print(f"🔧 Using device: {translator.optimal_device}")
        print()
        
        # Translate test texts
        print("🌸 Translation Results:")
        print("-" * 50)
        
        for i, text in enumerate(test_texts, 1):
            print(f"{i}. Japanese: {text}")
            
            # Translate
            result = translator.translate_text(text)
            
            print(f"   Chinese:  {result.translated_text}")
            print(f"   Quality:  {result.confidence:.2f}")
            print()
        
        # Cleanup
        translator.unload_model()
        print("🎯 Translation completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during translation: {e}")
        print(f"❌ Error: {e}")
        
        # Fallback suggestion
        print("\n💡 Troubleshooting:")
        print("- Ensure you have sufficient VRAM/RAM (4GB+ required)")
        print("- Check internet connection for model download")
        print("- Try with force_gpu=False for CPU fallback")

def compare_backends():
    """Compare SakuraLLM vs Helsinki-NLP translation quality."""
    
    print("\n🔄 Backend Comparison")
    print("=" * 50)
    
    test_text = "彼女は美しい桜の花を見ています。"
    
    try:
        # SakuraLLM translation
        print("🌸 SakuraLLM Translation:")
        sakura = PyTorchTranslator(
            model_name="SakuraLLM/Sakura-1.5B-Qwen2.5-v1.0-GGUF",
            source_lang="ja", target_lang="zh",
            device="auto", force_gpu=True
        )
        
        if sakura.load_model():
            result_sakura = sakura.translate_text(test_text)
            print(f"Input:  {test_text}")
            print(f"Output: {result_sakura.translated_text}")
            sakura.unload_model()
        
        # Helsinki-NLP comparison (would need to implement)
        print("\n📊 Helsinki-NLP Translation:")
        print("Input:  彼女は美しい桜の花を見ています。")
        print("Output: [Implement HuggingFaceTranslator comparison]")
        
    except Exception as e:
        print(f"❌ Comparison failed: {e}")

if __name__ == "__main__":
    main()
    
    # Optional: Run backend comparison
    # compare_backends()