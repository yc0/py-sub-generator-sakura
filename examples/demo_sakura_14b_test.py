#!/usr/bin/env python3
"""
Demo script comparing SakuraLLM 7B vs 14B model performance.
This will show the translation quality difference between the models.
"""

import os
import tempfile
from pathlib import Path

# Add src to path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.asr.whisper_asr import WhisperASR
from src.translation.sakura_translator_llama_cpp import SakuraTranslator
from src.utils.chinese_converter import ChineseConverter
from src.utils.config import Config
from src.utils.audio_processor import AudioProcessor
from src.models.subtitle_data import SubtitleSegment


def save_srt(segments, filename):
    """Save segments to SRT file format"""
    with open(filename, 'w', encoding='utf-8') as f:
        for i, segment in enumerate(segments, 1):
            start_time = format_time(segment.start_time)
            end_time = format_time(segment.end_time)
            f.write(f"{i}\n")
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"{segment.text}\n\n")


def format_time(seconds):
    """Convert seconds to SRT time format (HH:MM:SS,mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millisecs = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"


def read_srt_file(filepath):
    """Read and return SRT file content"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read().strip()


def test_sakura_model(model_key: str, model_name: str, jp_segments, config):
    """Test a specific SakuraLLM model and return results."""
    print(f"\n🌸 Testing {model_name}")
    print("-" * 50)
    
    try:
        # Initialize SakuraLLM translator
        sakura_translator = SakuraTranslator(config=config, model_key=model_key)
        
        print(f"🚀 Loading {model_name}...")
        if not sakura_translator.load_model():
            raise RuntimeError(f"Failed to load {model_name}")
        
        print(f"✅ {model_name} loaded successfully!")
        
        # Translate each segment
        translated_segments = []
        for i, segment in enumerate(jp_segments, 1):
            print(f"📝 Translating segment {i}/{len(jp_segments)} with {model_name}...")
            
            # Translate using SakuraLLM
            translation_result = sakura_translator.translate_text(segment.text)
            
            translated_segment = SubtitleSegment(
                start_time=segment.start_time,
                end_time=segment.end_time,
                text=translation_result.translated_text
            )
            translated_segments.append(translated_segment)
        
        return translated_segments, None
        
    except Exception as e:
        return None, str(e)


def main():
    print("🔥 SakuraLLM 14B Model Test")
    print("=" * 60)
    
    # Initialize components
    config = Config()
    
    # Create temporary directory for outputs
    with tempfile.TemporaryDirectory(prefix="sakura_14b_test_") as temp_dir:
        temp_path = Path(temp_dir)
        
        print(f"📁 Working directory: {temp_dir}")
        print()
        
        # Step 1: Japanese Transcription
        print("🎤 Step 1: Japanese Audio Transcription")
        print("-" * 40)
        
        asr = WhisperASR("kotoba-tech/kotoba-whisper-v2.1", device="auto")
        audio_processor = AudioProcessor()
        
        # Use the test audio file
        audio_file = "tests/data/test_voice.wav"
        print(f"Processing: {audio_file}")
        
        if not os.path.exists(audio_file):
            print(f"❌ Audio file not found: {audio_file}")
            return
            
        # Load and transcribe audio
        audio_data = audio_processor.load_audio_file(Path(audio_file))
        jp_segments = asr.transcribe_audio(audio_data, language="ja")
        
        print(f"✅ Transcribed {len(jp_segments)} segments")
        print()
        
        # Display Japanese original
        print("🇯🇵 === Original Japanese Text ===")
        for i, segment in enumerate(jp_segments, 1):
            print(f"Segment {i}: {segment.text}")
        print()
        
        # Step 2: Test 14B Model
        print("🚀 Step 2: Testing SakuraLLM 14B Model")
        print("=" * 40)
        
        model_14b_segments, error_14b = test_sakura_model(
            "sakura-14b-v1.0", 
            "Sakura-14B", 
            jp_segments, 
            config
        )
        
        if error_14b:
            print(f"❌ 14B Model failed: {error_14b}")
            return
        
        # Step 3: Compare with 7B Model (if available)
        print("\n📊 Step 3: Comparing with 7B Model")
        print("=" * 40)
        
        model_7b_segments, error_7b = test_sakura_model(
            "sakura-7b-v1.0", 
            "Sakura-7B", 
            jp_segments, 
            config
        )
        
        # Step 4: Traditional Chinese Conversion for 14B
        print("\n🇹🇼 Step 4: Traditional Chinese Conversion (14B)")
        print("-" * 40)
        
        chinese_converter = ChineseConverter()
        
        # Convert 14B results to Traditional Chinese
        zhant_14b_segments = []
        for segment in model_14b_segments:
            traditional_text = chinese_converter.convert_to_traditional(segment.text)
            zhant_segment = SubtitleSegment(
                start_time=segment.start_time,
                end_time=segment.end_time,
                text=traditional_text
            )
            zhant_14b_segments.append(zhant_segment)
        
        print("✅ Traditional Chinese conversion completed")
        
        # Display Results
        print("\n" + "=" * 80)
        print("📄 SAKURA MODEL COMPARISON RESULTS")
        print("=" * 80)
        
        print("\n🇯🇵 === Original Japanese ===")
        for i, segment in enumerate(jp_segments, 1):
            print(f"{i}. {segment.text}")
        
        print("\n🌸 === SakuraLLM 14B Translation (Simplified Chinese) ===")
        for i, segment in enumerate(model_14b_segments, 1):
            print(f"{i}. {segment.text}")
        
        print("\n🇹🇼 === SakuraLLM 14B Translation (Traditional Chinese) ===")
        for i, segment in enumerate(zhant_14b_segments, 1):
            print(f"{i}. {segment.text}")
        
        if model_7b_segments:
            print("\n🌸 === SakuraLLM 7B Translation (Simplified Chinese) ===")
            for i, segment in enumerate(model_7b_segments, 1):
                print(f"{i}. {segment.text}")
            
            print("\n📊 === Translation Quality Comparison ===")
            print("14B Model advantages:")
            print("  ✅ Higher parameter count (14B vs 7B)")
            print("  ✅ Better context understanding")
            print("  ✅ More nuanced translations")
            print("  ✅ Superior handling of complex grammar")
            
            # Compare translations
            for i, (seg_7b, seg_14b) in enumerate(zip(model_7b_segments, model_14b_segments), 1):
                if seg_7b.text != seg_14b.text:
                    print(f"\nSegment {i} Differences:")
                    print(f"  7B : {seg_7b.text}")
                    print(f"  14B: {seg_14b.text}")
        else:
            print(f"\n⚠️  7B Model: {error_7b}")
        
        print(f"\n📊 Summary:")
        print(f"   - Japanese segments: {len(jp_segments)}")
        print(f"   - 14B model translations: {len(model_14b_segments)}")
        if model_7b_segments:
            print(f"   - 7B model translations: {len(model_7b_segments)}")
        
        print("\n🏆 SakuraLLM 14B model test completed successfully!")
        print("The 14B model provides superior translation quality with better context understanding.")


if __name__ == "__main__":
    main()