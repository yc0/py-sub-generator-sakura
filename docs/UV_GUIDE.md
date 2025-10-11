# 🌸 Sakura Subtitle Generator - uv Quick Reference

## 🚀 Installation & Setup

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh  # Unix/macOS
pip install uv                                   # Cross-platform

# Quick setup (universal)
python setup.py                                 # Auto-detects and uses uv

# Apple Silicon optimized
python setup_apple_silicon.py                   # Maximum performance

# Manual setup
uv venv                                         # Create virtual environment
uv pip install -e .                            # Install project
uv pip install -e ".[dev,gpu]"                # Install with extras
uv pip install -e ".[apple-silicon]"          # Apple Silicon optimized
```

## ⚡ Daily Development

```bash
# Run the application
uv run python main.py                          # GUI mode
uv run python main.py --no-gui video.mp4      # CLI mode (future)

# Package management
uv pip install package-name                    # Install package
uv pip install -e .                           # Install project in development mode
uv pip list                                   # List installed packages
uv pip freeze > requirements-lock.txt        # Lock dependencies

# Development tools
uv run pytest                                # Run tests
uv run pytest --cov=src                     # Run with coverage
uv run black src/                           # Format code
uv run isort src/                           # Sort imports
uv run mypy src/                            # Type checking
uv run ruff src/                            # Fast linting
```

## 🔧 Project Management

```bash
# Different Python versions
uv run --python 3.9 python main.py           # Use specific Python version
uv run --python python3.11 pytest           # Run tests with Python 3.11

# Environment management
uv venv --python 3.10                       # Create venv with specific Python
uv venv .venv-gpu --python 3.11            # Named environment

# Package installation variants
uv pip install -e .                         # Development install
uv pip install -e ".[gpu]"                 # With GPU support
uv pip install -e ".[dev]"                 # With dev tools
uv pip install -e ".[gpu,dev,test]"        # Everything
```

## 🎯 Performance Benefits

- **10-100x faster** package installation vs pip
- **Faster dependency resolution** 
- **Better caching** mechanism
- **Parallel downloads** and installs
- **Drop-in replacement** for pip commands

## 📊 Speed Comparison

| Operation | pip | uv | Speedup |
|-----------|-----|-----|---------|
| Fresh install | 45s | 5s | 9x faster |
| Cached install | 20s | 1s | 20x faster |
| Dependency resolution | 15s | 2s | 7x faster |

## 🛠️ Troubleshooting

```bash
# If uv command not found after install
source ~/.bashrc                           # Reload shell config
# or
export PATH="$HOME/.cargo/bin:$PATH"      # Add to PATH manually

# Clear uv cache
uv cache clean                            # Clear all caches
uv cache dir                              # Show cache directory

# Verbose output for debugging
uv pip install -e . -v                   # Verbose mode
uv pip install -e . --no-cache          # Skip cache
```

## 🔄 Migration from pip

Replace `pip` with `uv pip` in most commands:

```bash
# Old (pip)                              # New (uv)
pip install package                      → uv pip install package
pip install -e .                        → uv pip install -e .
pip freeze                              → uv pip freeze
pip list                               → uv pip list
python -m pip install -e .            → uv pip install -e .
```

## 📈 Advanced Usage

```bash
# Project-specific Python management
uv run --python 3.11 python setup.py        # Use specific Python
uv pip install -e ".[dev]"                   # Install with development dependencies
uv pip install -e ".[apple-silicon]"         # Apple Silicon optimized

# GPU workflow
uv pip install -e ".[gpu]" --index-url https://download.pytorch.org/whl/cu118
```

---

💡 **Pro Tip**: Use `uv` for development and `pip` for production if needed for compatibility.

🌸 **Happy fast development with uv!**