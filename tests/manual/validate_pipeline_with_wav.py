from src.subtitle.subtitle_generator import SubtitleGenerator
from src.utils.config import Config
from src.models.video_data import VideoFile, AudioData
import soundfile as sf
from pathlib import Path

# Use output.wav as a 'video' file for testing
wav_path = Path('output.wav')

# Load config
from pathlib import Path
config = Config(config_file=Path('config.json'))
gen = SubtitleGenerator(config)


# Get file size for VideoFile
import os
file_size = os.path.getsize(wav_path)
video_file = VideoFile(
    file_path=wav_path,
    filename=wav_path.name,
    file_size=file_size,
    duration=None  # Will be filled by get_video_metadata if needed
)

# Patch file_handler to skip video metadata extraction for wav
class DummyFileHandler:
    def create_video_file_object(self, path):
        return video_file
    def get_video_metadata(self, vf):
        # Fill duration from wav file
        data, sr = sf.read(str(vf.file_path))
        vf.duration = len(data) / sr
        return vf
    def cleanup_temp_files(self, temp_files):
        pass

gen.file_handler = DummyFileHandler()

# Run the pipeline
subtitle_file = gen.process_video_file(wav_path, target_languages=["ja"])

if subtitle_file:
    print(f"Segments: {len(subtitle_file.segments)}")
    total = sum(seg.end_time - seg.start_time for seg in subtitle_file.segments)
    print(f"Total duration: {total:.2f} seconds ({int(total//60)} min {int(total%60)} sec)")
    for i, seg in enumerate(subtitle_file.segments, 1):
        print(f"[{i:3}] {seg.start_time:06.3f} → {seg.end_time:06.3f} | {seg.text[:40]}")
else:
    print("Subtitle generation failed.")
