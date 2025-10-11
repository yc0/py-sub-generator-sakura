# Test Data

This directory contains test audio files for validating the Whisper ASR functionality.

## Files

### `test_voice.wav`
- **Duration**: 20 seconds
- **Sample Rate**: 16kHz (optimal for Whisper)
- **Content**: Japanese speech sample
- **Purpose**: Validate Whisper transcription accuracy and performance
- **File Size**: ~625KB

## Usage

The test audio file is used by:
- `test_whisper_validation.py` - Comprehensive Whisper validation test suite

## Running Tests

### Direct execution:
```bash
uv run python tests/test_whisper_validation.py
```

### With pytest:
```bash
uv run pytest tests/test_whisper_validation.py -v
```

## Test Coverage

The validation test covers:
1. **Audio File Loading** - Verifies file can be loaded and has correct properties
2. **Model Loading** - Ensures Whisper model initializes correctly
3. **Transcription** - Tests actual speech-to-text conversion
4. **Content Quality** - Validates Japanese text output
5. **Timestamp Generation** - Checks timing accuracy
6. **Performance Benchmarking** - Measures transcription speed
7. **File Size Validation** - Confirms expected duration

Expected results: 7/7 tests should pass for a healthy Whisper setup.