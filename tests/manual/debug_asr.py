#!/usr/bin/env python3
"""
Diagnostic script to troubleshoot the "No segments produced" issue.
This helps identify where in the ASR pipeline the problem occurs.
"""

import logging
import sys
from pathlib import Path

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.asr.whisper_asr import WhisperASR
from src.utils.audio_processor import AudioProcessor
from src.utils.config import Config


def diagnose_asr_issue():
    """Diagnose the ASR issue with detailed logging."""
    
    print("🔍 ASR Diagnostic Tool - Investigating 'No segments produced' issue")
    print("=" * 70)
    
    # Initialize components
    try:
        config = Config()
        
        # Initialize ASR properly with model name from config
        asr_config = config.get_asr_config()
        asr = WhisperASR(
            model_name=asr_config.get("model_name", "openai/whisper-large-v3"),
            device=asr_config.get("device", "auto"),
            batch_size=asr_config.get("batch_size", 1),
            return_timestamps=asr_config.get("return_timestamps", True),
            chunk_length_s=asr_config.get("chunk_length", 30),
        )
        
        audio_processor = AudioProcessor()
        
        print(f"✅ Components initialized")
        print(f"   ASR Model: {asr.model_name}")
        print(f"   Return Timestamps: {asr.return_timestamps}")
        print(f"   Generation Config: {asr.generation_kwargs}")
        
        # Load model with detailed logging
        print(f"\n🤖 Loading ASR model...")
        if not asr.load_model():
            print("❌ Failed to load ASR model")
            return False
        
        print(f"✅ ASR model loaded successfully")
        print(f"   Device: {asr.torch_device}")
        print(f"   Model type: {type(asr.model)}")
        print(f"   Tokenizer type: {type(asr.tokenizer)}")
        
        # Load test audio
        audio_file = Path("tests/data/test_voice.wav")
        if not audio_file.exists():
            print(f"❌ Test audio file not found: {audio_file}")
            return False
        
        print(f"\n🎵 Loading test audio: {audio_file}")
        audio_data = audio_processor.load_audio_file(audio_file)
        
        if audio_data is None:
            print("❌ Failed to load audio data")
            return False
        
        print(f"✅ Audio loaded successfully")
        print(f"   Duration: {audio_data.duration:.2f}s")
        print(f"   Sample Rate: {audio_data.sample_rate}")
        print(f"   Shape: {audio_data.audio_array.shape}")
        print(f"   Audio Range: [{audio_data.audio_array.min():.4f}, {audio_data.audio_array.max():.4f}]")
        
        # Test preprocessing
        print(f"\n🔧 Testing audio preprocessing...")
        try:
            features = asr._preprocess_audio(audio_data)
            print(f"✅ Preprocessing successful")
            print(f"   Features shape: {features.shape}")
            print(f"   Features device: {features.device}")
            print(f"   Features dtype: {features.dtype}")
            print(f"   Features range: [{features.min():.4f}, {features.max():.4f}]")
        except Exception as e:
            print(f"❌ Preprocessing failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Test token generation
        print(f"\n🎯 Testing token generation...")
        try:
            # Generate forced decoder IDs (same as WhisperASR.transcribe_audio)
            language = "ja"
            language_token = asr.tokenizer.convert_tokens_to_ids(f"<|{language}|>")
            task_token = asr.tokenizer.convert_tokens_to_ids("<|transcribe|>")
            forced_decoder_ids = [
                (1, language_token),  # Language token at position 1
                (2, task_token),      # Task token at position 2
            ]
            print(f"✅ Token generation successful")
            print(f"   Language token: {language_token}")
            print(f"   Task token: {task_token}")
            print(f"   Forced decoder IDs: {forced_decoder_ids}")
            
            # Test generation with detailed logging
            import torch
            with torch.no_grad():
                generation_config = asr.generation_config.copy()
                generation_config["forced_decoder_ids"] = forced_decoder_ids
                generation_config["max_new_tokens"] = 50  # Limit for testing
                
                print(f"   Generation config: {generation_config}")
                
                # Create attention mask
                batch_size, n_mels, seq_len = features.shape
                attention_mask = torch.ones(
                    (batch_size, seq_len), device=asr.torch_device, dtype=torch.long
                )
                
                print(f"   Input features shape: {features.shape}")
                print(f"   Attention mask shape: {attention_mask.shape}")
                
                # Generate tokens
                generated_ids = asr.model.generate(
                    input_features=features,
                    attention_mask=attention_mask,
                    **generation_config,
                )
                
                print(f"✅ Token generation successful")
                print(f"   Generated IDs shape: {generated_ids.shape}")
                print(f"   Generated IDs: {generated_ids}")
                
                # Test decoding
                print(f"\n📝 Testing token decoding...")
                skip_tokens = 3
                new_tokens = generated_ids[0, skip_tokens:]
                print(f"   Tokens to decode (after skipping {skip_tokens}): {new_tokens}")
                
                # Test raw decoding
                raw_decoded = asr.tokenizer.decode(new_tokens, skip_special_tokens=False)
                print(f"   Raw decoded (with special tokens): '{raw_decoded}'")
                
                clean_decoded = asr.tokenizer.decode(new_tokens, skip_special_tokens=True)
                print(f"   Clean decoded (without special tokens): '{clean_decoded}'")
                
        except Exception as e:
            print(f"❌ Token generation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Test full transcription
        print(f"\n🎤 Testing full transcription...")
        try:
            segments = asr.transcribe_audio(audio_data, language="ja")
            print(f"📊 Transcription result: {len(segments)} segments")
            
            if len(segments) == 0:
                print(f"❌ No segments produced (this is the issue we're investigating)")
            else:
                print(f"✅ Segments produced successfully:")
                for i, segment in enumerate(segments):
                    print(f"   Segment {i+1}: [{segment.start_time:.2f}s-{segment.end_time:.2f}s] '{segment.text}'")
            
        except Exception as e:
            print(f"❌ Full transcription failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        return len(segments) > 0
        
    except Exception as e:
        print(f"❌ Diagnostic failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = diagnose_asr_issue()
    print(f"\n{'🎉 Diagnostic completed successfully!' if success else '💥 Diagnostic revealed issues!'}")