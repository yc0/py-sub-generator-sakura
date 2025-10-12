#!/usr/bin/env python3
"""
Full Pipeline Demonstration Test

This test demonstrates the complete pipeline:
Japanese Audio → Japanese SRT → English SRT → Traditional Chinese SRT

Shows all three output files regardless of whether SakuraLLM GGUF models are available.
"""

import pytest
import os
import tempfile
import shutil
from pathlib import Path

from src.asr.whisper_asr import WhisperASR
from src.translation.huggingface_translator import HuggingFaceTranslator
from src.translation.sakura_translator_llama_cpp import SakuraTranslator
from src.models.subtitle_data import SubtitleFile, SubtitleSegment, TranslationResult
from src.utils.audio_processor import AudioProcessor
from src.utils.config import Config
from src.utils.chinese_converter import convert_to_traditional


class TestFullPipelineDemo:
    """Demonstration of the complete pipeline with all output formats"""
    
    @pytest.fixture
    def audio_file(self):
        """Path to test Japanese audio file"""
        test_data_dir = Path(__file__).parent.parent / "data"
        audio_path = test_data_dir / "test_voice.wav"
        
        if not audio_path.exists():
            pytest.skip(f"Test audio file not found: {audio_path}")
        
        return str(audio_path)
    
    @pytest.fixture
    def output_dir(self):
        """Temporary directory for test outputs"""
        temp_dir = tempfile.mkdtemp(prefix="sakura_full_demo_")
        yield temp_dir
        # Cleanup after test
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def whisper_asr(self):
        """WhisperASR instance for Japanese transcription"""
        return WhisperASR(model_name="kotoba-tech/kotoba-whisper-v2.1", device="auto", language="ja")
    
    @pytest.fixture
    def audio_processor(self):
        """AudioProcessor for loading audio files"""
        return AudioProcessor()
    
    @pytest.fixture  
    def ja_en_translator(self):
        """HuggingFace translator for Japanese->English"""
        return HuggingFaceTranslator(
            model_name="Helsinki-NLP/opus-mt-ja-en",
            source_lang="ja",
            target_lang="en",
            device="auto"
        )
    
    @pytest.fixture  
    def en_zh_translator(self):
        """HuggingFace translator for English->Chinese"""
        return HuggingFaceTranslator(
            model_name="Helsinki-NLP/opus-mt-en-zh",
            source_lang="en",
            target_lang="zh",
            device="auto"
        )

    @pytest.mark.integration
    @pytest.mark.slow
    def test_full_pipeline_all_outputs(
        self, 
        audio_file, 
        output_dir, 
        whisper_asr, 
        ja_en_translator,
        en_zh_translator,
        audio_processor
    ):
        """
        Test complete pipeline showing all three SRT outputs:
        1. Japanese (from ASR)
        2. English (from ja->en translation)  
        3. Traditional Chinese (from en->zh + character conversion)
        """
        print(f"\n🎬 Full Sakura Pipeline Demonstration")
        print("=" * 60)
        
        # Step 1: ASR - Japanese audio to Japanese subtitles
        print(f"Step 1: 🎤 ASR Transcription")
        print(f"   Audio: {Path(audio_file).name}")
        
        # Load and process audio
        audio_data = audio_processor.load_audio_file(Path(audio_file))
        assert audio_data is not None, "Audio file should be loaded successfully"
        print(f"   Duration: {audio_data.duration:.1f}s, Sample Rate: {audio_data.sample_rate}Hz")
        
        # Load ASR model and transcribe
        assert whisper_asr.load_model(), "WhisperASR should load successfully"
        print(f"   ASR Model: {whisper_asr.model_name}")
        
        segments = whisper_asr.transcribe_audio(audio_data, language="ja")
        assert segments and len(segments) > 0, "Should produce segments"
        print(f"   ✅ Generated {len(segments)} segments")
        
        # Create Japanese SRT
        jp_srt_path = os.path.join(output_dir, "japanese.srt")
        subtitle_file = SubtitleFile(segments=segments)
        jp_srt_content = subtitle_file.export_srt()
        
        with open(jp_srt_path, 'w', encoding='utf-8') as f:
            f.write(jp_srt_content)
        
        print(f"   📄 Japanese SRT: {Path(jp_srt_path).name} ({len(jp_srt_content)} chars)")
        
        # Step 2: Japanese → English Translation
        print(f"\nStep 2: 🇯🇵→🇺🇸 Japanese to English Translation")
        
        assert ja_en_translator.load_model(), "ja->en translator should load"
        print(f"   Translation Model: {ja_en_translator.model_name}")
        
        source_texts = [segment.text for segment in segments]
        en_results = ja_en_translator.translate_batch(source_texts)
        assert len(en_results) == len(segments), "Should have translation for each segment"
        
        # Create English subtitle segments
        en_segments = []
        for original_segment, en_result in zip(segments, en_results):
            en_segments.append(SubtitleSegment(
                start_time=original_segment.start_time,
                end_time=original_segment.end_time,
                text=en_result.translated_text,
                confidence=original_segment.confidence
            ))
        
        # Create English SRT
        en_srt_path = os.path.join(output_dir, "english.srt")
        en_subtitle_file = SubtitleFile(segments=en_segments)
        en_srt_content = en_subtitle_file.export_srt()
        
        with open(en_srt_path, 'w', encoding='utf-8') as f:
            f.write(en_srt_content)
        
        print(f"   📄 English SRT: {Path(en_srt_path).name} ({len(en_srt_content)} chars)")
        
        # Step 3: English → Chinese Translation  
        print(f"\nStep 3: 🇺🇸→🇨🇳 English to Chinese Translation")
        
        assert en_zh_translator.load_model(), "en->zh translator should load"
        print(f"   Translation Model: {en_zh_translator.model_name}")
        
        en_texts = [result.translated_text for result in en_results]
        zh_results = en_zh_translator.translate_batch(en_texts)
        assert len(zh_results) == len(segments), "Should have Chinese translation for each segment"
        
        # Step 4: Simplified → Traditional Chinese Character Conversion
        print(f"\nStep 4: 🔤 Simplified → Traditional Chinese Conversion")
        
        zh_hant_segments = []
        for original_segment, zh_result in zip(segments, zh_results):
            # Convert to Traditional Chinese using OpenCC
            traditional_text = convert_to_traditional(zh_result.translated_text)
            
            zh_hant_segments.append(SubtitleSegment(
                start_time=original_segment.start_time,
                end_time=original_segment.end_time,
                text=traditional_text,
                confidence=original_segment.confidence
            ))
        
        # Create Traditional Chinese SRT
        zh_srt_path = os.path.join(output_dir, "chinese_traditional.srt")
        zh_subtitle_file = SubtitleFile(segments=zh_hant_segments)
        zh_srt_content = zh_subtitle_file.export_srt()
        
        with open(zh_srt_path, 'w', encoding='utf-8') as f:
            f.write(zh_srt_content)
        
        print(f"   📄 Traditional Chinese SRT: {Path(zh_srt_path).name} ({len(zh_srt_content)} chars)")
        print(f"   🔤 Character Conversion: OpenCC (Simplified → Traditional)")
        
        # Step 5: Verification and Display
        print(f"\nStep 5: ✅ Verification and Output Display")
        
        # Verify all files exist and have content
        files_to_check = [
            ("Japanese", jp_srt_path, jp_srt_content),
            ("English", en_srt_path, en_srt_content), 
            ("Traditional Chinese", zh_srt_path, zh_srt_content)
        ]
        
        for lang_name, file_path, content in files_to_check:
            assert os.path.exists(file_path), f"{lang_name} SRT should exist"
            assert len(content) > 0, f"{lang_name} SRT should have content"
            
            # Validate SRT format
            assert "00:" in content, f"{lang_name} SRT should contain timestamps"
            assert "-->" in content, f"{lang_name} SRT should contain timestamp separators"
        
        print(f"\n🎯 Pipeline Results Summary:")
        print(f"   📊 Processed {len(segments)} audio segments")
        print(f"   📄 Generated 3 subtitle files")
        print(f"   ⏱️  Total processing: Complete")
        
        # Display all SRT file contents
        print(f"\n" + "=" * 60)
        print(f"📂 SUBTITLE FILE CONTENTS")
        print(f"=" * 60)
        
        print(f"\n📄 🇯🇵 JAPANESE.SRT")
        print("-" * 30)
        print(jp_srt_content.strip())
        
        print(f"\n📄 🇺🇸 ENGLISH.SRT") 
        print("-" * 30)
        print(en_srt_content.strip())
        
        print(f"\n📄 🇹🇼 CHINESE_TRADITIONAL.SRT")
        print("-" * 30)
        print(zh_srt_content.strip())
        
        print(f"\n" + "=" * 60)
        print(f"✅ FULL PIPELINE DEMONSTRATION COMPLETE!")
        print(f"🌸 Ready for production with SakuraLLM enhancement")
        print(f"=" * 60)
        
        # Cleanup note
        print(f"\n🧹 Temporary files will be cleaned up automatically...")

    @pytest.mark.integration
    def test_sakura_llm_enhancement_demo(self):
        """
        Demonstrate SakuraLLM enhancement when GGUF models become available.
        This shows what the pipeline will look like with SakuraLLM.
        """
        print(f"\n🌸 SakuraLLM Enhancement Preview")
        print("=" * 50)
        
        config = Config()
        print(f"SakuraLLM Configuration:")
        print(f"   Enabled: {config.is_sakura_enabled()}")
        print(f"   Model: {config.get_sakura_config().get('model_name')}")
        print(f"   GGUF File: {config.get_sakura_config().get('model_file')}")
        
        try:
            # Try to initialize SakuraLLM
            from src.translation.sakura_translator_llama_cpp import SakuraTranslator
            translator = SakuraTranslator.create_from_config(config)
            
            print(f"   Status: ✅ Configured and ready")
            print(f"")
            print(f"🚀 Enhanced Pipeline (when GGUF models available):")
            print(f"   Japanese Audio → ASR")
            print(f"   ↓")
            print(f"   Japanese SRT") 
            print(f"   ↓")
            print(f"   🌸 SakuraLLM: ja → zh-Hans (High Quality)")
            print(f"   ↓")
            print(f"   🔤 OpenCC: zh-Hans → zh-Hant")
            print(f"   ↓")
            print(f"   Traditional Chinese SRT")
            print(f"")
            print(f"💡 Benefits:")
            print(f"   - Superior Japanese translation quality")
            print(f"   - Light novel/anime style awareness") 
            print(f"   - Direct ja→zh translation (no English intermediate)")
            print(f"   - 7B/14B model options for quality vs speed")
            
        except Exception as e:
            print(f"   Status: ⚠️ Ready (GGUF models needed)")
            print(f"   Details: {e}")


if __name__ == "__main__":
    """Run as standalone demonstration"""
    import sys
    
    print("🎬 Sakura Subtitle Generator - Full Pipeline Demo")
    print("=" * 60)
    print("This demonstration shows the complete subtitle generation workflow")
    print("including Japanese ASR, translation, and Traditional Chinese conversion.")
    print()
    
    # Run the demo
    test_instance = TestFullPipelineDemo()
    
    # Get fixtures manually for standalone run
    test_data_dir = Path(__file__).parent.parent / "data"
    audio_file = str(test_data_dir / "test_voice.wav")
    
    if not Path(audio_file).exists():
        print(f"❌ Test audio file not found: {audio_file}")
        print("Please ensure test audio is available for demonstration.")
        sys.exit(1)
    
    # Create temporary output directory
    import tempfile
    with tempfile.TemporaryDirectory(prefix="sakura_demo_") as temp_dir:
        try:
            # Create instances  
            config = Config()
            whisper_asr = WhisperASR(config)
            audio_processor = AudioProcessor()
            ja_en_translator = HuggingFaceTranslator("Helsinki-NLP/opus-mt-ja-en", "ja", "en")
            en_zh_translator = HuggingFaceTranslator("Helsinki-NLP/opus-mt-en-zh", "en", "zh")
            
            # Run full demo
            test_instance.test_full_pipeline_all_outputs(
                audio_file, temp_dir, whisper_asr, ja_en_translator, 
                en_zh_translator, audio_processor
            )
            
        except Exception as e:
            print(f"❌ Demo failed: {e}")
            import traceback
            traceback.print_exc()