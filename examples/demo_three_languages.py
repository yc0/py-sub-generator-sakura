#!/usr/bin/env python3
"""
Demo script showing complete pipeline with three language outputs:
- Japanese (original transcription)
- English (Helsinki-NLP translation)        print(f"📊 Summary:")
        print(f"   - Japanese segments: {len(jp_segments)}")
        print(f"   - English segments: {len(en_segments)}")
        print(f"   - Traditional Chinese segments: {len(zhant_segments)}")raditional Chinese (Helsinki-NLP + OpenCC conversion)
"""

import os
import tempfile
from pathlib import Path

# Add src to path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.asr.whisper_asr import WhisperASR
from src.translation.huggingface_translator import HuggingFaceTranslator
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
    print("🎌 Complete Three-Language Subtitle Pipeline Demo")
    print("=" * 60)
    
    # Initialize components
    config = Config()
    
    # Create temporary directory for outputs
    with tempfile.TemporaryDirectory(prefix="three_lang_demo_") as temp_dir:
        temp_path = Path(temp_dir)
        
        # Define output files
        jp_srt = temp_path / "jp.srt"
        en_srt = temp_path / "en.srt"
        zhant_srt = temp_path / "zhant.srt"
        
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
        
        # Step 2: English Translation
        print("🇺🇸 Step 2: Japanese → English Translation")
        print("-" * 40)
        
        ja_en_translator = HuggingFaceTranslator(
            model_name="Helsinki-NLP/opus-mt-ja-en",
            source_lang="ja",
            target_lang="en", 
            device="auto"
        )
        
        # Translate each segment
        en_segments = []
        for segment in jp_segments:
            en_result = ja_en_translator.translate_text(segment.text)
            en_segment = SubtitleSegment(
                start_time=segment.start_time,
                end_time=segment.end_time,
                text=en_result.translated_text
            )
            en_segments.append(en_segment)
        
        # Save English SRT
        save_srt(en_segments, en_srt)
        print(f"✅ English SRT saved: {en_srt}")
        print()
        
        # Step 3: Traditional Chinese Translation
        print("🇹🇼 Step 3: English → Traditional Chinese Translation")
        print("-" * 40)
        
        en_zh_translator = HuggingFaceTranslator(
            model_name="Helsinki-NLP/opus-mt-en-zh",
            source_lang="en",
            target_lang="zh",
            device="auto"
        )
        chinese_converter = ChineseConverter()
        
        # Translate each English segment to Simplified Chinese, then convert to Traditional
        zhant_segments = []
        for en_segment in en_segments:
            # Translate to Simplified Chinese
            zh_result = en_zh_translator.translate_text(en_segment.text)
            
            # Convert to Traditional Chinese
            traditional_text = chinese_converter.to_traditional(zh_result.translated_text)
            
            zhant_segment = SubtitleSegment(
                start_time=en_segment.start_time,
                end_time=en_segment.end_time,
                text=traditional_text
            )
            zhant_segments.append(zhant_segment)
        
        # Save Traditional Chinese SRT
        save_srt(zhant_segments, zhant_srt)
        print(f"✅ Traditional Chinese SRT saved: {zhant_srt}")
        print()
        
        # Display Results
        print("📄 FINAL RESULTS - Three Language Subtitles")
        print("=" * 60)
        
        print("\n🇯🇵 === Japanese SRT Content (jp.srt) ===")
        print(read_srt_file(jp_srt))
        
        print("\n🇺🇸 === English SRT Content (en.srt) ===")
        print(read_srt_file(en_srt))
        
        print("\n🇹🇼 === Traditional Chinese SRT Content (zhant.srt) ===")
        print(read_srt_file(zhant_srt))
        
        print(f"\n📊 Summary:")
        print(f"   - Japanese segments: {len(jp_result.segments)}")
        print(f"   - English segments: {len(en_segments)}")
        print(f"   - Traditional Chinese segments: {len(zhant_segments)}")
        
        print(f"\n📁 File sizes:")
        print(f"   - jp.srt: {jp_srt.stat().st_size} bytes")
        print(f"   - en.srt: {en_srt.stat().st_size} bytes") 
        print(f"   - zhant.srt: {zhant_srt.stat().st_size} bytes")
        
        print("\n🎉 Three-language subtitle generation completed successfully!")


if __name__ == "__main__":
    main()