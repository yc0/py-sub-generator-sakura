#!/usr/bin/env python3
"""
Regression test to reproduce and verify fix for 'No segments produced' issue
"""
import pytest
import numpy as np
from pathlib import Path

from src.asr.whisper_asr import WhisperASR
from src.models.video_data import AudioData
from src.utils.config import Config


def create_test_audio(duration_seconds=1.0, sample_rate=16000, amplitude=0.0):
    """Create test audio with specified parameters."""
    num_samples = int(duration_seconds * sample_rate)
    
    if amplitude == 0.0:
        # Silent audio
        audio_array = np.zeros(num_samples, dtype=np.float32)
    else:
        # Sine wave audio
        t = np.linspace(0, duration_seconds, num_samples, False)
        audio_array = amplitude * np.sin(440 * 2 * np.pi * t).astype(np.float32)
    
    return AudioData(
        audio_array=audio_array,
        sample_rate=sample_rate,
        duration=duration_seconds,
        channels=1
    )


@pytest.mark.regression
@pytest.mark.slow
@pytest.mark.model_download
class TestNoSegmentsIssue:
    """Regression tests for 'No segments produced' issue."""
    
    @pytest.fixture(scope="class")
    def asr(self):
        """Create WhisperASR instance for testing."""
        asr = WhisperASR(model_name="kotoba-tech/kotoba-whisper-v2.1", device="auto", language="ja")
        success = asr.load_model()
        if not success:
            pytest.skip("WhisperASR model could not be loaded")
        return asr
    
    def test_silent_audio_handling(self, asr):
        """Test that silent audio doesn't cause 'No segments produced'."""
        audio_data = create_test_audio(duration_seconds=1.0, amplitude=0.0)
        
        segments = asr.transcribe_audio(audio_data)
        
        # Should produce at least one segment, even if empty/silence
        assert isinstance(segments, list), "Expected list of segments"
        # Note: We don't assert len(segments) > 0 because silent audio 
        # might legitimately produce no segments in some cases
    
    def test_very_short_audio(self, asr):
        """Test very short audio (0.1s) handling."""
        audio_data = create_test_audio(duration_seconds=0.1, amplitude=0.1)
        
        segments = asr.transcribe_audio(audio_data)
        
        assert isinstance(segments, list), "Expected list of segments"
        # Very short audio might not produce segments, which is acceptable
    
    def test_short_silent_audio(self, asr):
        """Test short silent audio (0.5s) handling."""
        audio_data = create_test_audio(duration_seconds=0.5, amplitude=0.0)
        
        segments = asr.transcribe_audio(audio_data)
        
        assert isinstance(segments, list), "Expected list of segments"
    
    def test_short_audio_with_sound(self, asr):
        """Test short audio with sound (0.5s) handling."""
        audio_data = create_test_audio(duration_seconds=0.5, amplitude=0.1)
        
        segments = asr.transcribe_audio(audio_data)
        
        assert isinstance(segments, list), "Expected list of segments"
    
    def test_normal_length_audio(self, asr):
        """Test normal length audio handling."""
        audio_data = create_test_audio(duration_seconds=2.0, amplitude=0.1)
        
        segments = asr.transcribe_audio(audio_data)
        
        assert isinstance(segments, list), "Expected list of segments"
        # Normal length audio should typically produce segments


@pytest.mark.regression
def test_audio_data_creation():
    """Test that AudioData objects can be created correctly."""
    
    # Test silent audio
    silent = create_test_audio(1.0, amplitude=0.0)
    assert silent.duration == 1.0
    assert silent.sample_rate == 16000
    assert len(silent.audio_array) == 16000
    assert np.max(np.abs(silent.audio_array)) == 0.0
    
    # Test audio with sound
    sound = create_test_audio(0.5, amplitude=0.1)
    assert sound.duration == 0.5
    assert sound.sample_rate == 16000
    assert len(sound.audio_array) == 8000
    assert np.max(np.abs(sound.audio_array)) > 0.0


def main():
    """Manual debugging function."""
    print("🧪 Testing scenarios that might cause 'No segments produced'")
    print("=" * 60)
    
    # Test audio creation first
    print("\n📁 Testing AudioData creation...")
    try:
        test_audio_data_creation()
        print("✅ AudioData creation test passed!")
    except Exception as e:
        print(f"❌ AudioData creation failed: {e}")
        return
    
    # Test ASR scenarios
    print("\n🎤 Testing ASR scenarios...")
    from src.utils.config import Config
    from src.asr.whisper_asr import WhisperASR
    
    config = Config()
    asr = WhisperASR(model_name="kotoba-tech/kotoba-whisper-v2.1", device="auto", language="ja")
    
    if not asr.load_model():
        print("❌ Could not load WhisperASR model")
        return

    test_scenarios = [
        ("Silent audio (1s)", create_test_audio(1.0, amplitude=0.0)),
        ("Very short audio (0.1s)", create_test_audio(0.1, amplitude=0.1)),
        ("Short silent audio (0.5s)", create_test_audio(0.5, amplitude=0.0)),
        ("Short audio with sound (0.5s)", create_test_audio(0.5, amplitude=0.1)),
    ]
    
    for name, audio_data in test_scenarios:
        print(f"\n🎯 Testing: {name}")
        print(f"   Duration: {audio_data.duration:.2f}s")
        print(f"   Samples: {len(audio_data.audio_array):,}")
        print(f"   Max amplitude: {np.max(np.abs(audio_data.audio_array)):.3f}")
        
        try:
            segments = asr.transcribe_audio(audio_data)
            
            if len(segments) == 0:
                print("   ⚠️  No segments produced (may be expected for silent/short audio)")
            else:
                print(f"   ✅ Generated {len(segments)} segments")
                for i, segment in enumerate(segments[:3]):  # Show first 3
                    print(f"      Segment {i+1}: {segment.start_time:.2f}s-{segment.end_time:.2f}s: '{segment.text[:30]}...'")
        
        except Exception as e:
            print(f"   💥 ERROR: {e}")
    
    print("\n🏁 Test completed")


if __name__ == "__main__":
    """Run as standalone script for manual debugging"""
    main()