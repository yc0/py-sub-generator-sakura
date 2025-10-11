# 🌸 Sakura Subtitle Generator - Makefile
# Convenient commands for development and testing

.PHONY: help test test-fast test-unit test-gpu test-slow test-coverage install clean lint format

# Default target
help:
	@echo "🌸 Sakura Subtitle Generator - Available Commands"
	@echo "================================================="
	@echo ""
	@echo "Testing:"
	@echo "  test          Run fast tests (no GPU, no downloads)"
	@echo "  test-unit     Run unit tests only"
	@echo "  test-gpu      Run GPU tests (requires GPU)"
	@echo "  test-slow     Run slow tests (may download models)"
	@echo "  test-all      Run all tests"
	@echo "  test-coverage Generate coverage report"
	@echo ""
	@echo "Development:"
	@echo "  install       Install dependencies with uv"
	@echo "  install-dev   Install with dev dependencies + tools"
	@echo "  install-tools Install dev tools (ruff, black, etc) via uv tool"  
	@echo "  check-tools   Check installed development tools"
	@echo "  format        Format code with black and isort"
	@echo "  lint          Run linting with ruff and mypy"
	@echo "  clean         Clean cache and temporary files"
	@echo ""
	@echo "Examples:"
	@echo "  make test              # Quick tests"
	@echo "  make test-gpu          # Test GPU acceleration"
	@echo "  make test-coverage     # Generate coverage report"

# Testing commands
test:
	@echo "🧪 Running fast tests..."
	./run_tests.py --type fast

test-unit:
	@echo "🧪 Running unit tests..."
	./run_tests.py --type unit -v

test-gpu:
	@echo "🧪 Running GPU tests..."
	./run_tests.py --type gpu --gpu -v

test-slow:
	@echo "🧪 Running slow tests (may download models)..."
	./run_tests.py --type slow -v

test-all:
	@echo "🧪 Running all tests..."
	./run_tests.py --type all --gpu -v

test-coverage:
	@echo "🧪 Running tests with coverage..."
	./run_tests.py --type fast --coverage
	@echo "📊 Coverage report generated in htmlcov/"

# Installation commands
install:
	@echo "📦 Installing dependencies..."
	uv sync

install-dev:
	@echo "📦 Installing with dev dependencies..."
	uv sync --extra dev
	@echo "🔧 Installing development tools with uv tool..."
	@$(MAKE) install-tools

install-gpu:
	@echo "📦 Installing with GPU dependencies..."
	uv sync --extra gpu

install-apple:
	@echo "📦 Installing Apple Silicon optimized..."
	uv sync --extra apple-silicon

install-all:
	@echo "📦 Installing all dependencies..."
	uv sync --all-extras

# Development tools (managed separately via uv tool)
install-tools:
	@echo "🔧 Installing development tools with uv tool..."
	@echo "  📏 Installing ruff (linter & formatter)..."
	@uv tool install ruff
	@echo "  🎨 Installing black (code formatter)..."  
	@uv tool install black
	@echo "  📋 Installing isort (import sorter)..."
	@uv tool install isort
	@echo "  🔍 Installing mypy (type checker)..."
	@uv tool install mypy
	@echo "✅ Development tools installed globally with uv tool!"

update-tools:
	@echo "⬆️  Updating development tools..."
	@uv tool upgrade ruff || echo "ruff not installed, skipping"
	@uv tool upgrade black || echo "black not installed, skipping"  
	@uv tool upgrade isort || echo "isort not installed, skipping"
	@uv tool upgrade mypy || echo "mypy not installed, skipping"
	@echo "✅ Development tools updated!"

check-tools:
	@echo "🔍 Checking development tools..."
	@echo -n "  ruff: " && (uv tool run ruff --version 2>/dev/null || echo "❌ Not installed")
	@echo -n "  black: " && (uv tool run black --version 2>/dev/null || echo "❌ Not installed")
	@echo -n "  isort: " && (uv tool run isort --version 2>/dev/null || echo "❌ Not installed") 
	@echo -n "  mypy: " && (uv tool run mypy --version 2>/dev/null || echo "❌ Not installed")

# Development commands
format:
	@echo "🎨 Formatting code..."
	@uv tool run black src/ tests/ --line-length 88
	@uv tool run isort src/ tests/ --profile black

lint:
	@echo "🔍 Running linters..."
	@uv tool run ruff check src/ tests/
	@uv tool run mypy src/ --ignore-missing-imports

lint-fix:
	@echo "🔧 Auto-fixing lint issues..."
	@uv tool run ruff check src/ tests/ --fix
	@uv tool run black src/ tests/ --line-length 88
	@uv tool run isort src/ tests/ --profile black

# Cleanup commands
clean:
	@echo "🧹 Cleaning up..."
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache/ .coverage htmlcov/ .mypy_cache/
	rm -rf build/ dist/ *.egg-info/

clean-cache:
	@echo "🧹 Cleaning model cache..."
	@echo "⚠️  This will remove downloaded models (~3.6GB)"
	@read -p "Are you sure? [y/N]: " confirm && [ "$$confirm" = "y" ] || exit 1
	rm -rf ~/.cache/huggingface/hub/models--Helsinki-NLP--*
	rm -rf ~/.cache/huggingface/hub/models--openai--whisper-*

# Model management
download-models:
	@echo "📥 Pre-downloading models..."
	uv run python -c "from old_tests.predownload_models import predownload_models; predownload_models()"

benchmark:
	@echo "⚡ Running performance benchmarks..."
	./run_tests.py --type slow -k benchmark -v

# Docker support (if needed)
docker-build:
	@echo "🐳 Building Docker image..."
	docker build -t sakura-subtitle-generator .

docker-test:
	@echo "🐳 Running tests in Docker..."
	docker run --rm sakura-subtitle-generator make test

# Documentation
docs:
	@echo "📚 Generating documentation..."
	@echo "README.md and inline docs are the primary documentation"

# CI/CD helpers
ci-test:
	@echo "🤖 Running CI tests..."
	./run_tests.py --type no-download --coverage

check-setup:
	@echo "🔍 Checking project setup..."
	uv run python -c "import torch; print(f'PyTorch: {torch.__version__}')"
	uv run python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
	uv run python -c "import torch; print(f'MPS Available: {torch.backends.mps.is_available() if hasattr(torch.backends, \"mps\") else False}')"