#!/usr/bin/env python3
"""Installation script for Sakura Subtitle Generator."""

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
    
    try:
        if use_uv:
            print("🚀 Using uv for fast installation...")
            print("💡 Installing in isolated uv environment (no system pollution)")
            # Install project in editable mode with uv
            subprocess.check_call(['uv', 'pip', 'install', '-e', '.'])
            print("✅ Dependencies installed with uv")
        else:
            print("⚠️  WARNING: uv not found - falling back to pip")
            print("� This will install packages in your current Python environment")
            response = input("Continue with pip installation? (y/N): ")
            if not response.lower().startswith('y'):
                print("❌ Installation cancelled. Install uv with: pip install uv")
                return False
            # Install project in editable mode with pip
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-e', '.'])
            print("✅ Dependencies installed with pip")
        
        print("✅ Dependencies installed successfully")
        
        # Check for GPU support and offer additional packages
        try:
            import torch
            if torch.cuda.is_available():
                response = input("🚀 CUDA detected! Install GPU optimized packages? (y/N): ")
                if response.lower().startswith('y'):
                    if use_uv:
                        subprocess.check_call(['uv', 'pip', 'install', '-e', '.[gpu]'])
                    else:
                        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-e', '.[gpu]'])
                    print("✅ GPU optimized packages installed")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                response = input("🍎 MPS (Apple Silicon) detected! Install optimized packages? (y/N): ")
                if response.lower().startswith('y'):
                    if use_uv:
                        subprocess.check_call(['uv', 'pip', 'install', '-e', '.[apple-silicon]'])
                    else:
                        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-e', '.[apple-silicon]'])
                    print("✅ Apple Silicon optimized packages installed")
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
    print("🌸 Sakura Subtitle Generator - Installation")
    print("=" * 42)
    
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
    
    print("\n🎉 Installation completed!")
    
    if not ffmpeg_ok:
        print("\n⚠️  Please install FFmpeg before running the application")
    
    if has_uv:
        print("\n🚀 Recommended way to run (isolated environment):")
        print("   uv run python main.py")
        print("\n📦 Alternative (after installation):")
        print("   python main.py")
        print("\n💡 Tip: Use 'uv run' to avoid environment pollution!")
    else:
        print("\n🚀 To start the application, run:")
        print("   python main.py")
    
    print("\n📖 For more information, see README.md")


if __name__ == "__main__":
    main()