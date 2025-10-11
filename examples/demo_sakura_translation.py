#!/usr/bin/env python3
"""
Demo script using SakuraLLM instead of Helsinki-NLP models.
Pipeline: Japanese ASR → SakuraLLM (ja→zh) → Traditional Chinese conversion

This demo shows:
- Japanese (original transcription)  
- Simplified Chinese (SakuraLLM translation)
- Traditional Chinese (OpenCC conversion)
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


def main():
    print("🌸 SakuraLLM Translation Pipeline Demo")
    print("=" * 60)
    
    # Initialize components
    config = Config()
    
    # Create temporary directory for outputs
    with tempfile.TemporaryDirectory(prefix="sakura_demo_") as temp_dir:
        temp_path = Path(temp_dir)
        
        # Define output files
        jp_srt = temp_path / "jp.srt"
        zhcn_srt = temp_path / "zhcn.srt"  # Simplified Chinese (SakuraLLM)
        zhant_srt = temp_path / "zhant.srt"  # Traditional Chinese (OpenCC)
        
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
        
        # Save Japanese SRT
        save_srt(jp_segments, jp_srt)
        print(f"✅ Japanese SRT saved: {jp_srt}")
        print()
        
        # Step 2: SakuraLLM Translation (Japanese → Simplified Chinese)
        print("🌸 Step 2: SakuraLLM Translation (Japanese → Chinese)")
        print("-" * 40)
        
        try:
            # Initialize SakuraLLM translator
            sakura_translator = SakuraTranslator(config=config, model_key="sakura-7b-v1.0")
            
            print("🚀 Loading SakuraLLM model...")
            if not sakura_translator.load_model():
                raise RuntimeError("Failed to load SakuraLLM model")
            
            print("✅ SakuraLLM model loaded successfully!")
            
            # Translate each segment
            zhcn_segments = []
            for i, segment in enumerate(jp_segments, 1):
                print(f"📝 Translating segment {i}/{len(jp_segments)}: {segment.text[:50]}...")
                
                # Translate using SakuraLLM
                translation_result = sakura_translator.translate_text(segment.text)
                
                zhcn_segment = SubtitleSegment(
                    start_time=segment.start_time,
                    end_time=segment.end_time,
                    text=translation_result.translated_text
                )
                zhcn_segments.append(zhcn_segment)
            
            # Save Simplified Chinese SRT
            save_srt(zhcn_segments, zhcn_srt)
            print(f"✅ Simplified Chinese SRT saved: {zhcn_srt}")
            
        except Exception as e:
            print(f"❌ SakuraLLM translation failed: {e}")
            print("\nTo use SakuraLLM, first download the model files:")
            print("  uv run python download_sakura_models.py")
            return
        
        print()
        
        # Step 3: Traditional Chinese Conversion
        print("🇹🇼 Step 3: Traditional Chinese Conversion")
        print("-" * 40)
        
        chinese_converter = ChineseConverter()
        
        # Convert each Simplified Chinese segment to Traditional
        zhant_segments = []
        for zhcn_segment in zhcn_segments:
            # Convert to Traditional Chinese
            traditional_text = chinese_converter.convert_to_traditional(zhcn_segment.text)
            
            zhant_segment = SubtitleSegment(
                start_time=zhcn_segment.start_time,
                end_time=zhcn_segment.end_time,
                text=traditional_text
            )
            zhant_segments.append(zhant_segment)
        
        # Save Traditional Chinese SRT
        save_srt(zhant_segments, zhant_srt)
        print(f"✅ Traditional Chinese SRT saved: {zhant_srt}")
        print()
        
        # Display Results
        print("📄 FINAL RESULTS - SakuraLLM Translation")
        print("=" * 60)
        
        print("\n🇯🇵 === Japanese SRT Content (jp.srt) ===")
        print(read_srt_file(jp_srt))
        
        print("\n🇨🇳 === Simplified Chinese SRT Content (zhcn.srt) ===")
        print(read_srt_file(zhcn_srt))
        
        print("\n🇹🇼 === Traditional Chinese SRT Content (zhant.srt) ===")
        print(read_srt_file(zhant_srt))
        
        print(f"\n📊 Summary:")
        print(f"   - Japanese segments: {len(jp_segments)}")
        print(f"   - Simplified Chinese segments: {len(zhcn_segments)}")
        print(f"   - Traditional Chinese segments: {len(zhant_segments)}")
        
        print(f"\n📁 File sizes:")
        print(f"   - jp.srt: {jp_srt.stat().st_size} bytes")
        print(f"   - zhcn.srt: {zhcn_srt.stat().st_size} bytes") 
        print(f"   - zhant.srt: {zhant_srt.stat().st_size} bytes")
        
        print("\n🌸 SakuraLLM translation pipeline completed successfully!")


if __name__ == "__main__":
    main()