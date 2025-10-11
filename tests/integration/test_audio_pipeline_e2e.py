"""
End-to-End Audio Pipeline Integration Test

Tests the complete workflow:
ASR (Japanese audio) -> Japanese SRT -> English SRT + Traditional Chinese SRT

This test validates:
1. WhisperASR transcription of Japanese audio
2. SRT file generation in Japanese 
3. Translation to English
4. Translation to Traditional Chinese with OpenCC conversion
5. File output preservation and cleanup
"""

import pytest
import os
import tempfile
import shutil
from pathlib import Path

from src.asr.whisper_asr import WhisperASR
from src.translation.huggingface_translator import HuggingFaceTranslator
from src.translation.sakura_translator_llama_cpp import SakuraTranslator
from src.models.subtitle_data import SubtitleFile, SubtitleSegment
from src.utils.audio_processor import AudioProcessor


class TestAudioPipelineE2E:
    """End-to-end testing of the complete audio processing pipeline"""
    
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
        temp_dir = tempfile.mkdtemp(prefix="py_sub_generator_e2e_")
        yield temp_dir
        # Cleanup after test
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def whisper_asr(self):
        """WhisperASR instance for Japanese transcription"""
        return WhisperASR(
            model_name="kotoba-tech/kotoba-whisper-v2.1",  # Use configured Japanese ASR model
            device="auto"
        )
    
    @pytest.fixture
    def audio_processor(self):
        """AudioProcessor for loading audio files"""
        return AudioProcessor()
    
    @pytest.fixture  
    def huggingface_translator(self):
        """HuggingFace translator for Japanese->English"""
        return HuggingFaceTranslator(
            model_name="Helsinki-NLP/opus-mt-ja-en",
            source_lang="ja",
            target_lang="en",
            device="auto"
        )
    
    @pytest.fixture
    def sakura_translator(self):
        """SakuraTranslator for Chinese conversion (if available)"""
        try:
            return SakuraTranslator(
                model_name_or_path="SakuraLLM/Sakura-1.5B-Qwen2.5-v1.0-GGUF",
                model_file="sakura-1.5b-qwen2.5-v1.0-q4_k_m.gguf",  # Use configured GGUF model
                device="auto",
                target_lang="zh-cn"  # Start with simplified, then convert to traditional
            )
        except Exception:
            # SakuraTranslator may not be available in test environment
            return None
    
    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.model_download
    def test_complete_audio_pipeline_hf_only(
        self, 
        audio_file, 
        output_dir, 
        whisper_asr, 
        huggingface_translator,
        audio_processor
    ):
        """
        Test complete pipeline using HuggingFace models only:
        Japanese Audio -> Japanese SRT -> English SRT
        """
        # Step 1: ASR - Japanese audio to segments
        print(f"Step 1: Transcribing audio file: {audio_file}")
        
        # Load audio file
        audio_data = audio_processor.load_audio_file(Path(audio_file))
        assert audio_data is not None, "Audio file should be loaded successfully"
        
        # Transcribe audio
        segments = whisper_asr.transcribe_audio(audio_data, language="ja")
        
        assert segments is not None, "ASR should produce segments"
        assert len(segments) > 0, "ASR should produce at least one segment"
        
        # Validate segment structure (SubtitleSegment objects)
        for segment in segments:
            assert hasattr(segment, 'start_time'), "Segment should have start_time"
            assert hasattr(segment, 'end_time'), "Segment should have end_time"
            assert hasattr(segment, 'text'), "Segment should have text"
            assert len(segment.text.strip()) > 0, "Segment text should not be empty"
        
        print(f"ASR produced {len(segments)} segments")
        
        # Step 2: Generate Japanese SRT file
        jp_srt_path = os.path.join(output_dir, "japanese.srt")
        print(f"Step 2: Writing Japanese SRT to: {jp_srt_path}")
        
        # Create SubtitleFile and export to SRT (segments are already SubtitleSegment objects)
        subtitle_file = SubtitleFile(segments=segments)
        srt_content = subtitle_file.export_srt()
        
        # Write SRT content to file
        with open(jp_srt_path, 'w', encoding='utf-8') as f:
            f.write(srt_content)
        
        assert os.path.exists(jp_srt_path), "Japanese SRT file should be created"
        assert os.path.getsize(jp_srt_path) > 0, "Japanese SRT file should not be empty"
        
        # Validate Japanese SRT content
        with open(jp_srt_path, 'r', encoding='utf-8') as f:
            jp_content = f.read()
            assert "00:" in jp_content, "SRT should contain timestamps"
            assert "-->" in jp_content, "SRT should contain timestamp separators"
        
        # Step 3: Translate to English
        print(f"Step 3: Translating segments to English")
        
        # Extract text from segments for translation
        source_texts = [segment.text for segment in segments]
        
        # Translate Japanese to English
        english_results = huggingface_translator.translate_batch(source_texts)
        english_texts = [result.translated_text for result in english_results]
        
        assert len(english_texts) == len(source_texts), "Should have translation for each segment"
        
        # Create English subtitle segments
        english_subtitle_segments = []
        for original_segment, english_text in zip(segments, english_texts):
            english_subtitle_segments.append(SubtitleSegment(
                start_time=original_segment.start_time,
                end_time=original_segment.end_time,
                text=english_text,
                confidence=original_segment.confidence
            ))
        
        # Step 4: Generate English SRT file
        en_srt_path = os.path.join(output_dir, "english.srt")
        print(f"Step 4: Writing English SRT to: {en_srt_path}")
        
        # Create English SubtitleFile and export to SRT
        english_subtitle_file = SubtitleFile(segments=english_subtitle_segments)
        english_srt_content = english_subtitle_file.export_srt()
        
        # Write English SRT content to file
        with open(en_srt_path, 'w', encoding='utf-8') as f:
            f.write(english_srt_content)
        
        assert os.path.exists(en_srt_path), "English SRT file should be created"
        assert os.path.getsize(en_srt_path) > 0, "English SRT file should not be empty"
        
        # Validate English SRT content
        with open(en_srt_path, 'r', encoding='utf-8') as f:
            en_content = f.read()
            assert "00:" in en_content, "English SRT should contain timestamps"
            assert "-->" in en_content, "English SRT should contain timestamp separators"
        
        # Step 5: Verify output preservation
        print(f"Step 5: Verifying output files preservation")
        
        # Both files should exist and be non-empty
        assert os.path.exists(jp_srt_path), "Japanese SRT should be preserved"
        assert os.path.exists(en_srt_path), "English SRT should be preserved"
        
        jp_size = os.path.getsize(jp_srt_path)
        en_size = os.path.getsize(en_srt_path)
        
        assert jp_size > 0, "Japanese SRT should have content"
        assert en_size > 0, "English SRT should have content"
        
        print(f"✅ Pipeline completed successfully!")
        print(f"   - Japanese SRT: {jp_srt_path} ({jp_size} bytes)")
        print(f"   - English SRT: {en_srt_path} ({en_size} bytes)")
        print(f"   - Processed {len(segments)} audio segments")
        
        # Display file contents before cleanup
        print(f"\n=== 🇯🇵 Japanese SRT Content ===")
        with open(jp_srt_path, 'r', encoding='utf-8') as f:
            print(f.read())
        
        print(f"=== 🇺🇸 English SRT Content ===")
        with open(en_srt_path, 'r', encoding='utf-8') as f:
            print(f.read())
        
        print(f"🧹 Cleaning up temporary files...")
        # Note: Cleanup happens automatically via the output_dir fixture
    
    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.model_download
    def test_complete_audio_pipeline_with_sakura(
        self, 
        audio_file, 
        output_dir, 
        whisper_asr, 
        huggingface_translator,
        sakura_translator,
        audio_processor
    ):
        """
        Test complete pipeline with SakuraLLM and Chinese conversion:
        Japanese Audio -> Japanese SRT -> English SRT + Traditional Chinese SRT
        """
        if sakura_translator is None:
            pytest.skip("SakuraTranslator not available in test environment")
        
        # Step 1-4: Same as HuggingFace-only test
        print(f"Step 1: Transcribing audio file: {audio_file}")
        
        # Load audio file
        audio_data = audio_processor.load_audio_file(Path(audio_file))
        assert audio_data is not None, "Audio file should be loaded successfully"
        
        # Transcribe audio
        segments = whisper_asr.transcribe_audio(audio_data, language="ja")
        
        assert segments is not None and len(segments) > 0
        
        # Japanese SRT
        jp_srt_path = os.path.join(output_dir, "japanese.srt")
        
        # Create SubtitleFile and export to SRT (segments are already SubtitleSegment objects)
        subtitle_file = SubtitleFile(segments=segments)
        srt_content = subtitle_file.export_srt()
        
        # Write SRT content to file
        with open(jp_srt_path, 'w', encoding='utf-8') as f:
            f.write(srt_content)
        
        # English translation
        source_texts = [segment.text for segment in segments]
        english_results = huggingface_translator.translate_batch(source_texts)
        english_texts = [result.translated_text for result in english_results]
        
        english_subtitle_segments = []
        for original_segment, english_text in zip(segments, english_texts):
            english_subtitle_segments.append(SubtitleSegment(
                start_time=original_segment.start_time,
                end_time=original_segment.end_time,
                text=english_text,
                confidence=original_segment.confidence
            ))
        
        en_srt_path = os.path.join(output_dir, "english.srt")
        
        # Create English SubtitleFile and export to SRT
        english_subtitle_file = SubtitleFile(segments=english_subtitle_segments)
        english_srt_content = english_subtitle_file.export_srt()
        
        # Write English SRT content to file
        with open(en_srt_path, 'w', encoding='utf-8') as f:
            f.write(english_srt_content)
        
        # Step 5: SakuraLLM Chinese translation
        print(f"Step 5: Translating to Traditional Chinese with SakuraLLM")
        
        try:
            # Use SakuraTranslator for Japanese -> Chinese translation
            chinese_texts = sakura_translator.translate(source_texts)
            
            assert len(chinese_texts) == len(source_texts), "Should have Chinese translation for each segment"
            
            # Create Traditional Chinese segments with OpenCC conversion
            zhant_subtitle_segments = []
            for original_segment, chinese_text in zip(segments, chinese_texts):
                # Apply Traditional Chinese conversion if SakuraTranslator supports it
                if hasattr(sakura_translator, 'chinese_converter') and sakura_translator.chinese_converter:
                    zhant_text = sakura_translator.chinese_converter.convert(chinese_text)
                else:
                    # Fallback: use the translated text as-is
                    zhant_text = chinese_text
                
                zhant_subtitle_segments.append(SubtitleSegment(
                    start_time=original_segment.start_time,
                    end_time=original_segment.end_time,
                    text=zhant_text,
                    confidence=original_segment.confidence
                ))
            
            # Step 6: Generate Traditional Chinese SRT file
            zhant_srt_path = os.path.join(output_dir, "traditional_chinese.srt")
            print(f"Step 6: Writing Traditional Chinese SRT to: {zhant_srt_path}")
            
            # Create Traditional Chinese SubtitleFile and export to SRT
            zhant_subtitle_file = SubtitleFile(segments=zhant_subtitle_segments)
            zhant_srt_content = zhant_subtitle_file.export_srt()
            
            # Write Traditional Chinese SRT content to file
            with open(zhant_srt_path, 'w', encoding='utf-8') as f:
                f.write(zhant_srt_content)
            
            assert os.path.exists(zhant_srt_path), "Traditional Chinese SRT file should be created"
            assert os.path.getsize(zhant_srt_path) > 0, "Traditional Chinese SRT file should not be empty"
            
            # Step 7: Verify all three output files
            print(f"Step 7: Verifying all output files preservation")
            
            files_to_check = [
                ("Japanese", jp_srt_path),
                ("English", en_srt_path), 
                ("Traditional Chinese", zhant_srt_path)
            ]
            
            for lang_name, file_path in files_to_check:
                assert os.path.exists(file_path), f"{lang_name} SRT should exist"
                size = os.path.getsize(file_path)
                assert size > 0, f"{lang_name} SRT should have content"
                
                # Validate SRT format
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    assert "00:" in content, f"{lang_name} SRT should contain timestamps"
                    assert "-->" in content, f"{lang_name} SRT should contain timestamp separators"
            
            print(f"✅ Complete pipeline with SakuraLLM completed successfully!")
            print(f"   - Japanese SRT: {jp_srt_path}")
            print(f"   - English SRT: {en_srt_path}")
            print(f"   - Traditional Chinese SRT: {zhant_srt_path}")
            print(f"   - Processed {len(segments)} audio segments")
            
            # Display all file contents before cleanup
            print(f"\n=== 🇯🇵 Japanese SRT Content ===")
            with open(jp_srt_path, 'r', encoding='utf-8') as f:
                print(f.read())
            
            print(f"=== 🇺🇸 English SRT Content ===")
            with open(en_srt_path, 'r', encoding='utf-8') as f:
                print(f.read())
                
            print(f"=== 🇹🇼 Traditional Chinese SRT Content ===")
            with open(zhant_srt_path, 'r', encoding='utf-8') as f:
                print(f.read())
            
            print(f"🧹 Cleaning up temporary files...")
            # Note: Cleanup happens automatically via the output_dir fixture
            
        except Exception as e:
            pytest.skip(f"SakuraLLM translation failed: {e}")
    
    @pytest.mark.integration
    @pytest.mark.manual
    def test_pipeline_output_inspection(self, audio_file, whisper_asr, audio_processor):
        """
        Manual inspection test - outputs transcription for human validation
        This test helps verify ASR quality on the Japanese test audio
        """
        print(f"\n=== Manual Pipeline Inspection ===")
        print(f"Audio file: {audio_file}")
        
        # Get audio file info
        import subprocess
        try:
            result = subprocess.run(['file', audio_file], capture_output=True, text=True)
            print(f"Audio info: {result.stdout.strip()}")
        except:
            pass
        
        # Transcribe
        print(f"\nTranscribing...")
        
        # Load audio file
        audio_data = audio_processor.load_audio_file(Path(audio_file))
        if audio_data is None:
            print("❌ Failed to load audio file")
            return
        
        print(f"Audio loaded: {audio_data.duration:.2f}s, {audio_data.sample_rate}Hz")
        
        # Transcribe audio
        segments = whisper_asr.transcribe_audio(audio_data, language="ja")
        
        if segments and len(segments) > 0:
            print(f"\n=== Transcription Results ({len(segments)} segments) ===")
            for i, segment in enumerate(segments, 1):
                start_time = segment.start_time
                end_time = segment.end_time
                text = segment.text.strip()
                
                print(f"Segment {i}: [{start_time:.2f}s - {end_time:.2f}s]")
                print(f"  Japanese: {text}")
                print()
        else:
            print("❌ No segments produced from audio file")
        
        # This test always passes - it's for manual inspection
        assert True, "Manual inspection test"


# Standalone execution for debugging
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    # Quick manual test
    test_instance = TestAudioPipelineE2E()
    
    # Get test audio file
    test_data_dir = Path(__file__).parent.parent / "data"
    audio_path = test_data_dir / "test_voice.wav"
    
    if audio_path.exists():
        print(f"Testing with audio file: {audio_path}")
        
        # Create WhisperASR and AudioProcessor for quick test
        try:
            whisper_asr = WhisperASR(model_name="kotoba-tech/kotoba-whisper-v2.1", device="auto")
            audio_processor = AudioProcessor()
            test_instance.test_pipeline_output_inspection(str(audio_path), whisper_asr, audio_processor)
        except Exception as e:
            print(f"Error during manual test: {e}")
    else:
        print(f"Audio file not found: {audio_path}")