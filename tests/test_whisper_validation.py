#!/usr/bin/env python3
"""
Test case to validate that test_voice.wav can be properly whispered
"""

import sys
import json
from pathlib import Path

# Add parent directory to path to import src modules
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_whisper_audio_validation():
    """Comprehensive test to validate audio can be transcribed by Whisper"""
    
    print("🧪 Whisper Audio Validation Test Suite")
    print("=" * 50)
    
    test_results = {
        "audio_loading": False,
        "model_loading": False,
        "transcription": False,
        "segments_generated": False,
        "japanese_text": False,
        "timestamps": False,
        "file_size": False
    }
    
    try:
        # Test 1: Audio File Validation
        print("\n📁 Test 1: Audio File Validation")
        from src.utils.audio_processor import AudioProcessor
        
        # Get the test audio file path
        test_audio_path = Path(__file__).parent / "data" / "test_voice.wav"
        print(f"   🎵 Loading: {test_audio_path}")
        
        processor = AudioProcessor()
        audio_data = processor.load_audio_file(test_audio_path)
        
        if not audio_data:
            print(f"   ❌ Failed to load {test_audio_path}")
            return False
        
        print(f"   ✅ Audio loaded successfully")
        print(f"   📊 Duration: {audio_data.duration:.2f}s")
        print(f"   📊 Sample rate: {audio_data.sample_rate} Hz")
        print(f"   📊 Samples: {len(audio_data.audio_array):,}")
        
        # Check if it's exactly 20 seconds
        if abs(audio_data.duration - 20.0) < 0.1:
            print("   ✅ Duration is exactly 20 seconds")
            test_results["file_size"] = True
        else:
            print(f"   ⚠️  Duration is {audio_data.duration:.2f}s (expected ~20s)")
        
        test_results["audio_loading"] = True
        
        # Test 2: Model Loading
        print("\n🤖 Test 2: Whisper Model Loading")
        
        # Load config from project root
        config_path = Path(__file__).parent.parent / "config.json"
        with open(config_path, "r") as f:
            config = json.load(f)
        
        model_name = config["asr"]["model_name"]
        print(f"   🎯 Testing model: {model_name}")
        
        from src.asr.whisper_asr import WhisperASR
        
        asr = WhisperASR(
            model_name=model_name,
            device="auto",
            return_timestamps=True,
        )
        
        if asr.load_model():
            print("   ✅ Model loaded successfully")
            test_results["model_loading"] = True
        else:
            print("   ❌ Model failed to load")
            return False
        
        # Test 3: Transcription
        print("\n🗣️  Test 3: Audio Transcription")
        
        segments = asr.transcribe_audio(audio_data, language="ja")
        
        if segments:
            print(f"   ✅ Transcription successful")
            print(f"   📊 Generated {len(segments)} segments")
            test_results["transcription"] = True
            test_results["segments_generated"] = True
            
            # Test 4: Content Validation
            print("\n📝 Test 4: Content Quality Validation")
            
            for i, segment in enumerate(segments):
                print(f"   Segment {i+1}: {segment.start_time:.2f}s-{segment.end_time:.2f}s")
                print(f"   Text: '{segment.text[:100]}{'...' if len(segment.text) > 100 else ''}'")
                
                # Check if contains Japanese characters
                japanese_chars = any(ord(char) > 127 for char in segment.text)
                if japanese_chars:
                    print("   ✅ Contains Japanese characters")
                    test_results["japanese_text"] = True
                else:
                    print("   ⚠️  No Japanese characters detected")
                
                # Check timestamps
                if segment.start_time >= 0 and segment.end_time > segment.start_time:
                    print("   ✅ Valid timestamps")
                    test_results["timestamps"] = True
                else:
                    print("   ❌ Invalid timestamps")
            
        else:
            print("   ❌ No segments generated")
            return False
        
        # Test 5: Performance Validation
        print("\n⚡ Test 5: Performance Validation")
        
        import time
        start_time = time.time()
        
        # Quick second transcription to test performance
        quick_segments = asr.transcribe_audio(audio_data, language="ja")
        
        elapsed = time.time() - start_time
        print(f"   ⏱️  Transcription time: {elapsed:.2f}s")
        print(f"   📊 Audio ratio: {audio_data.duration / elapsed:.1f}x real-time")
        
        if elapsed < 60:  # Should be much faster than 60s for 20s audio
            print("   ✅ Performance acceptable")
        else:
            print("   ⚠️  Performance slower than expected")
        
        # Cleanup
        asr.unload_model()
        
        # Final Results
        print("\n📋 Test Summary")
        print("=" * 30)
        
        passed_tests = sum(test_results.values())
        total_tests = len(test_results)
        
        for test_name, passed in test_results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"   {test_name:<20}: {status}")
        
        print(f"\n🎯 Overall: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("🎉 ALL TESTS PASSED! Audio is ready for Whisper transcription.")
            return True
        elif passed_tests >= 5:  # Most important tests
            print("✅ CORE TESTS PASSED! Audio should work with minor issues.")
            return True
        else:
            print("❌ CRITICAL TESTS FAILED! Audio needs attention.")
            return False
            
    except Exception as e:
        print(f"\n💥 Test suite crashed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_whisper_validation_pytest():
    """Pytest-compatible wrapper for the validation test"""
    result = test_whisper_audio_validation()
    assert result, "Whisper audio validation failed"

if __name__ == "__main__":
    success = test_whisper_audio_validation()
    sys.exit(0 if success else 1)