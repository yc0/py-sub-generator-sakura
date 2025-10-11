#!/usr/bin/env python3
"""
Download SakuraLLM GGUF model files from Hugging Face.
"""

import os
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download

def download_sakura_model(model_name: str, model_file: str, target_dir: Path = None):
    """Download SakuraLLM GGUF model file."""
    
    if target_dir is None:
        target_dir = Path.home() / ".cache" / "sakura"
    
    target_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 Downloading SakuraLLM model: {model_name}")
    print(f"📁 Target directory: {target_dir}")
    print(f"📦 Model file: {model_file}")
    
    try:
        # Download the model file
        local_file = hf_hub_download(
            repo_id=model_name,
            filename=model_file,
            cache_dir=target_dir / "hub",
            local_dir=target_dir / model_name.split("/")[-1],
            local_dir_use_symlinks=False
        )
        
        print(f"✅ Downloaded: {local_file}")
        return Path(local_file)
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return None

def main():
    """Download SakuraLLM models based on config.json"""
    
    print("🌸 SakuraLLM Model Downloader")
    print("=" * 50)
    
    # Available models from config
    models = {
        "7b": {
            "model_name": "SakuraLLM/Sakura-7B-Qwen2.5-v1.0-GGUF",
            "model_file": "sakura-7b-qwen2.5-v1.0-iq4xs.gguf",
            "size": "~4.3GB"
        },
        "14b": {
            "model_name": "SakuraLLM/Sakura-14B-Qwen2.5-v1.0-GGUF", 
            "model_file": "sakura-14b-qwen2.5-v1.0-iq4xs.gguf",
            "size": "~8.5GB"
        }
    }
    
    print("Available models:")
    for key, info in models.items():
        print(f"  {key}: {info['model_name']} ({info['size']})")
    
    # Ask user which model to download
    choice = input("\nWhich model to download? (7b/14b/both): ").lower().strip()
    
    if choice in ["7b", "both"]:
        print(f"\n📥 Downloading 7B model...")
        download_sakura_model(
            models["7b"]["model_name"],
            models["7b"]["model_file"]
        )
    
    if choice in ["14b", "both"]:
        print(f"\n📥 Downloading 14B model...")
        download_sakura_model(
            models["14b"]["model_name"], 
            models["14b"]["model_file"]
        )
    
    if choice not in ["7b", "14b", "both"]:
        print("❌ Invalid choice. Please select 7b, 14b, or both.")
        return
    
    print("\n🎉 Download complete!")
    print("\nYou can now run the SakuraLLM demo with:")
    print("  uv run python demo_sakura_translation.py")

if __name__ == "__main__":
    main()