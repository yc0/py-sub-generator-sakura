#!/usr/bin/env python3
"""
Hardware Acceleration Tests for AudioProcessor
Tests the cross-platform hardware acceleration detection and audio extraction using pytest framework.
"""

import logging
import time
import pytest
from pathlib import Path

from src.utils.audio_processor import AudioProcessor
from src.models.video_data import VideoFile


@pytest.fixture
def audio_processor():
    """AudioProcessor fixture for testing."""
    return AudioProcessor()


@pytest.fixture
def test_audio_file():
    """Test audio file fixture."""
    test_audio = Path("tests/data/test_voice.wav")
    if not test_audio.exists():
        pytest.skip(f"Test audio file not found: {test_audio}")
    return test_audio


@pytest.fixture
def video_file_obj(test_audio_file):
    """VideoFile object fixture for testing."""
    return VideoFile(
        file_path=test_audio_file,
        filename=test_audio_file.name,
        file_size=test_audio_file.stat().st_size
    )


class TestHardwareAcceleration:
    """Test hardware acceleration functionality."""

    def test_hardware_acceleration_detection(self, audio_processor):
        """Test hardware acceleration detection."""
        # Test that hardware acceleration detection runs without error
        # The actual acceleration available depends on hardware
        hwaccel = audio_processor.hwaccel
        
        # Should be either None (software), or a string (hardware acceleration type)
        assert hwaccel is None or isinstance(hwaccel, str)
        
        # Log the detected acceleration for debugging
        print(f"🎯 Detected Hardware Acceleration: {hwaccel or 'Software (No HW acceleration)'}")

    @pytest.mark.slow
    def test_audio_extraction_performance(self, audio_processor, video_file_obj):
        """Test audio extraction performance with timing."""
        print(f"📁 Test File: {video_file_obj.filename} ({video_file_obj.file_size} bytes)")
        
        # Time the extraction
        start_time = time.time()
        audio_data = audio_processor.extract_audio_from_video(video_file_obj)
        extraction_time = time.time() - start_time
        
        # Verify extraction succeeded
        assert audio_data is not None, "Audio extraction should succeed"
        assert audio_data.duration > 0, "Audio should have positive duration"
        assert audio_data.sample_rate > 0, "Audio should have valid sample rate"
        assert len(audio_data.audio_array) > 0, "Audio array should not be empty"
        
        # Log performance metrics
        performance_ratio = audio_data.duration / extraction_time
        print(f"✅ Audio Extraction Successful!")
        print(f"   ⏱️  Extraction Time: {extraction_time:.3f} seconds")
        print(f"   🎵 Audio Duration: {audio_data.duration:.2f} seconds")
        print(f"   📊 Sample Rate: {audio_data.sample_rate:,} Hz")
        print(f"   📈 Audio Shape: {audio_data.audio_array.shape}")
        print(f"   🔊 Channels: {audio_data.channels}")
        print(f"   ⚡ Performance: {performance_ratio:.1f}x realtime")
        
        # Performance should be at least 0.5x realtime (not slower than 2x audio duration)
        assert performance_ratio >= 0.5, f"Extraction too slow: {performance_ratio:.1f}x realtime"

    def test_audio_preprocessing(self, audio_processor, video_file_obj):
        """Test audio preprocessing functionality."""
        # First extract audio
        audio_data = audio_processor.extract_audio_from_video(video_file_obj)
        assert audio_data is not None
        
        # Test preprocessing
        start_time = time.time()
        processed_audio = audio_processor.preprocess_audio_for_asr(audio_data)
        preprocess_time = time.time() - start_time
        
        # Verify preprocessing results
        assert processed_audio is not None, "Preprocessing should return audio data"
        assert len(processed_audio.shape) >= 1, "Processed audio should have valid shape"
        
        print(f"🔧 Audio Preprocessing:")
        print(f"   ✅ Preprocessing Time: {preprocess_time:.3f} seconds")
        print(f"   📊 Processed Shape: {processed_audio.shape}")
        print(f"   📏 Data Type: {processed_audio.dtype}")

    def test_audio_chunking(self, audio_processor, video_file_obj):
        """Test audio chunking functionality."""
        # First extract audio
        audio_data = audio_processor.extract_audio_from_video(video_file_obj)
        assert audio_data is not None
        
        # Test chunking
        chunks = audio_processor.split_audio_chunks(audio_data, chunk_duration=5.0, overlap=1.0)
        
        # Verify chunking results
        assert isinstance(chunks, list), "Chunks should be a list"
        assert len(chunks) > 0, "Should produce at least one chunk"
        
        print(f"✂️  Audio Chunking:")
        print(f"   ✅ Created {len(chunks)} chunks")
        if chunks:
            print(f"   📊 First chunk duration: {chunks[0].duration:.2f}s")
            print(f"   📊 Last chunk duration: {chunks[-1].duration:.2f}s")


@pytest.mark.integration
class TestCrossPlatformCompatibility:
    """Test cross-platform hardware acceleration compatibility."""

    def test_ffmpeg_availability(self):
        """Test that FFmpeg is available and has hardware acceleration info."""
        import subprocess
        import platform
        
        print(f"🌍 Cross-Platform Compatibility Test")
        print(f"🖥️  Platform: {platform.system()} {platform.machine()}")
        
        try:
            # Check FFmpeg version
            result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
            assert result.returncode == 0, "FFmpeg should be available"
            
            version_line = result.stdout.split('\n')[0]
            print(f"🎬 FFmpeg: {version_line}")
            
            # Check hardware accelerators
            result = subprocess.run(['ffmpeg', '-hwaccels'], capture_output=True, text=True)
            if result.returncode == 0:
                hwaccels = [line.strip() for line in result.stdout.split('\n') 
                           if line.strip() and 'Hardware' not in line]
                print(f"🚀 Available Hardware Accelerators: {', '.join(hwaccels) if hwaccels else 'None'}")
            
        except FileNotFoundError:
            pytest.fail("FFmpeg not found. Please install FFmpeg.")

    def test_hardware_acceleration_graceful_fallback(self, audio_processor, video_file_obj):
        """Test that hardware acceleration fails gracefully when not available."""
        # This test ensures that even if hardware acceleration is not available,
        # the software fallback works correctly
        
        audio_data = audio_processor.extract_audio_from_video(video_file_obj)
        
        # Should always succeed regardless of hardware acceleration availability
        assert audio_data is not None, "Audio extraction should work with software fallback"
        assert audio_data.duration > 0, "Extracted audio should have valid duration"


if __name__ == "__main__":
    """Run tests standalone for debugging."""
    pytest.main([__file__, "-v", "-s"])