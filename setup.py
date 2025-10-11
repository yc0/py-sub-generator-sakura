#!/usr/bin/env python3
"""Setup script for Sakura Subtitle Generator."""

import sys
import subprocess
import platform
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required.")
        print(f"Current version: {sys.version}")
        sys.exit(1)
    else:
        print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")


def check_ffmpeg():
    """Check if FFmpeg is installed."""
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ FFmpeg found")
            return True
    except FileNotFoundError:
        pass
    
    print("⚠️  FFmpeg not found - required for audio extraction")
    print("Please install FFmpeg:")
    
    system = platform.system().lower()
    if system == "darwin":  # macOS
        print("  brew install ffmpeg")
    elif system == "linux":
        print("  sudo apt install ffmpeg  # Ubuntu/Debian")
        print("  sudo yum install ffmpeg  # RHEL/CentOS")
    elif system == "windows":
        print("  Download from: https://ffmpeg.org/download.html")
    
    return False


def check_uv():
    """Check if uv is installed."""
    try:
        subprocess.run(['uv', '--version'], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def install_dependencies():
    """Install Python dependencies using uv or pip."""
    print("📦 Installing Python dependencies...")
    
    # Check if uv is available
    use_uv = check_uv()
    
    if use_uv:
        print("🚀 Using uv for fast installation...")
        installer = ['uv', 'pip', 'install']
        sync_cmd = ['uv', 'pip', 'sync', 'requirements.txt']
    else:
        print("📦 Using pip for installation...")
        installer = [sys.executable, '-m', 'pip', 'install']
        sync_cmd = None
    
    try:
        if use_uv and sync_cmd:
            # Use uv sync for faster installation
            try:
                subprocess.check_call(sync_cmd)
                print("✅ Dependencies synced with uv")
            except subprocess.CalledProcessError:
                # Fallback to regular install
                subprocess.check_call(installer + ['-r', 'requirements.txt'])
        else:
            # Use pip install
            subprocess.check_call(installer + ['-r', 'requirements.txt'])
        
        print("✅ Dependencies installed successfully")
        
        # Check for CUDA and offer GPU support
        try:
            import torch
            if torch.cuda.is_available():
                response = input("🚀 CUDA detected! Install GPU support? (y/N): ")
                if response.lower().startswith('y'):
                    gpu_deps = ['torch>=2.0.0,<3.0.0', 'torchvision>=0.15.0', 'torchaudio>=2.0.0']
                    if use_uv:
                        subprocess.check_call(['uv', 'pip', 'install'] + gpu_deps)
                    else:
                        subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + gpu_deps)
                    print("✅ GPU support installed")
        except ImportError:
            pass
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False


def create_directories():
    """Create necessary directories."""
    dirs = ["outputs", "temp", "logs"]
    
    for dir_name in dirs:
        dir_path = Path(dir_name)
        dir_path.mkdir(exist_ok=True)
        print(f"📁 Created directory: {dir_name}")


def check_apple_silicon():
    """Check if running on Apple Silicon and suggest optimized setup."""
    if platform.machine() == "arm64" and platform.system() == "Darwin":
        print("🍎 Apple Silicon detected!")
        print("💡 For optimal performance, consider using:")
        print("   python setup_apple_silicon.py")
        print("   This includes MPS acceleration and ARM64 optimizations.")
        print()
        return True
    return False


def main():
    """Main setup function."""
    print("🌸 Sakura Subtitle Generator - Setup")
    print("=" * 40)
    
    # Check for Apple Silicon and suggest optimized setup
    check_apple_silicon()
    
    # Check system requirements
    check_python_version()
    ffmpeg_ok = check_ffmpeg()
    
    # Check if uv is available
    has_uv = check_uv()
    if has_uv:
        print("⚡ uv detected - using for fast installation")
    else:
        print("💡 For faster installs, consider installing uv:")
        print("   curl -LsSf https://astral.sh/uv/install.sh | sh")
        print("   # or")
        print("   pip install uv")
    
    # Install dependencies
    if not install_dependencies():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    print("\n🎉 Setup completed!")
    
    if not ffmpeg_ok:
        print("\n⚠️  Please install FFmpeg before running the application")
    
    print("\n🚀 To start the application, run:")
    print("   python main.py")
    
    if has_uv:
        print("\n⚡ With uv, you can also run:")
        print("   uv run python main.py")
        print("   uv pip install -e .")
    
    print("\n📖 For more information, see README.md")


if __name__ == "__main__":
    main()