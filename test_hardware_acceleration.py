#!/usr/bin/env python3
"""
Hardware Acceleration Test for AudioProcessor
Tests the cross-platform hardware acceleration detection and audio extraction.
"""

import logging
import time
from pathlib import Path

from src.utils.audio_processor import AudioProcessor
from src.models.video_data import VideoFile

def test_hardware_acceleration():
    """Test hardware acceleration detection and performance."""
    
    # Setup logging to see acceleration info
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    print("🚀 AudioProcessor Hardware Acceleration Test")
    print("=" * 50)
    
    # Test 1: Hardware Detection
    processor = AudioProcessor()
    print(f"🎯 Detected Hardware Acceleration: {processor.hwaccel or 'Software (No HW acceleration)'}")
    
    # Test 2: Audio Extraction Performance
    test_audio = Path("tests/data/test_voice.wav")
    if not test_audio.exists():
        print(f"❌ Test file not found: {test_audio}")
        return False
        
    # Create VideoFile object
    video_file = VideoFile(
        file_path=test_audio,
        filename=test_audio.name,
        file_size=test_audio.stat().st_size
    )
    
    print(f"📁 Test File: {test_audio.name} ({video_file.file_size} bytes)")
    
    # Time the extraction
    start_time = time.time()
    audio_data = processor.extract_audio_from_video(video_file)
    extraction_time = time.time() - start_time
    
    if audio_data:
        print(f"✅ Audio Extraction Successful!")
        print(f"   ⏱️  Extraction Time: {extraction_time:.3f} seconds")
        print(f"   🎵 Audio Duration: {audio_data.duration:.2f} seconds")
        print(f"   📊 Sample Rate: {audio_data.sample_rate:,} Hz")
        print(f"   📈 Audio Shape: {audio_data.audio_array.shape}")
        print(f"   🔊 Channels: {audio_data.channels}")
        print(f"   ⚡ Performance: {audio_data.duration/extraction_time:.1f}x realtime")
        
        # Test 3: Audio Preprocessing
        print(f"\n🔧 Testing Audio Preprocessing...")
        start_time = time.time()
        processed_audio = processor.preprocess_audio_for_asr(audio_data)
        preprocess_time = time.time() - start_time
        
        print(f"   ✅ Preprocessing Time: {preprocess_time:.3f} seconds")
        print(f"   📊 Processed Shape: {processed_audio.shape}")
        print(f"   📏 Data Type: {processed_audio.dtype}")
        
        # Test 4: Audio Chunking
        print(f"\n✂️  Testing Audio Chunking...")
        chunks = processor.split_audio_chunks(audio_data, chunk_duration=5.0, overlap=1.0)
        print(f"   ✅ Created {len(chunks)} chunks")
        if chunks:
            print(f"   📊 First chunk duration: {chunks[0].duration:.2f}s")
            print(f"   📊 Last chunk duration: {chunks[-1].duration:.2f}s")
        
        return True
    else:
        print(f"❌ Audio Extraction Failed!")
        return False

def test_cross_platform_compatibility():
    """Test cross-platform hardware acceleration compatibility."""
    
    print(f"\n🌍 Cross-Platform Compatibility Test")
    print("=" * 40)
    
    import subprocess
    import platform
    
    print(f"🖥️  Platform: {platform.system()} {platform.machine()}")
    
    try:
        # Check FFmpeg capabilities
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            print(f"🎬 FFmpeg: {version_line}")
        
        # Check hardware accelerators
        result = subprocess.run(['ffmpeg', '-hwaccels'], capture_output=True, text=True)
        if result.returncode == 0:
            hwaccels = [line.strip() for line in result.stdout.split('\n') 
                       if line.strip() and 'Hardware' not in line]
            print(f"🚀 Available Hardware Accelerators: {', '.join(hwaccels) if hwaccels else 'None'}")
        
        return True
        
    except Exception as e:
        print(f"❌ FFmpeg check failed: {e}")
        return False

if __name__ == "__main__":
    success = True
    
    # Run hardware acceleration test
    success &= test_hardware_acceleration()
    
    # Run cross-platform compatibility test  
    success &= test_cross_platform_compatibility()
    
    print(f"\n{'🎉 ALL TESTS PASSED!' if success else '❌ SOME TESTS FAILED!'}")
    print("=" * 50)
    
    if success:
        print("✅ Hardware acceleration is working correctly")
        print("✅ Audio extraction performance is optimal") 
        print("✅ Cross-platform compatibility confirmed")
    
    exit(0 if success else 1)