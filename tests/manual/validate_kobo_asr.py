from src.asr.whisper_asr import WhisperASR
from src.models.video_data import AudioData
import soundfile as sf

# Load your audio
file_path = 'output.wav'
data, sr = sf.read(file_path)
audio_data = AudioData(audio_array=data, sample_rate=sr, duration=len(data)/sr, channels=1, bit_depth=16)

# Use kobo model
asr = WhisperASR(model_name="kotoba-tech/kotoba-whisper-v2.1")
segments = asr.transcribe_batch([audio_data])

total = sum(seg.end_time - seg.start_time for seg in segments)
print(f'Segments: {len(segments)}, Total duration: {total:.2f} seconds ({int(total//60)} min {int(total%60)} sec)')
for i, seg in enumerate(segments, 1):
    print(f'[{i:3}] {seg.start_time:06.3f} → {seg.end_time:06.3f} | {seg.text[:40]}')
