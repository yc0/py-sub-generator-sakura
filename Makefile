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

# Development tools (project-scoped via uv tool run - no global installation!)
check-tools:
	@echo "� Checking development tools (project-scoped)..."
	@echo "  📏 ruff: Available via 'uv tool run ruff'"
	@echo "  🎨 black: Available via 'uv tool run black'"
	@echo "  📋 isort: Available via 'uv tool run isort'"
	@echo "  🔍 mypy: Available via 'uv tool run mypy'"
	@echo ""
	@echo "💡 Tools are used on-demand without global installation!"
	@echo "   This keeps your global environment clean while providing access to latest versions."

# Development commands
format:
	@echo "🎨 Formatting code..."
	@uv tool run black src/ tests/ --line-length 88
	@uv tool run isort src/ tests/ --profile black

# Code quality checks (project-scoped tools)
lint:
	@echo "🔍 Running comprehensive code quality checks..."
	@echo "  📏 Running ruff linter..."
	@uv tool run ruff@latest check . --config pyproject.toml
	@echo "  🎨 Running black formatter check..."
	@uv tool run black@latest --check --diff --config pyproject.toml .
	@echo "  📋 Running isort import sorting check..."  
	@uv tool run isort@latest --check-only --diff --settings-path pyproject.toml .
	@echo "  🔍 Running mypy type checking..."
	@uv tool run mypy@latest --config-file pyproject.toml src/
	@echo "✅ All code quality checks passed!"

# Code formatting and fixing (project-scoped tools)
format:
	@echo "🎨 Formatting code with project-scoped tools..."
	@echo "  📏 Auto-fixing with ruff..."
	@uv tool run ruff@latest check . --fix --config pyproject.toml
	@echo "  🎨 Formatting with black..."
	@uv tool run black@latest --config pyproject.toml .
	@echo "  📋 Sorting imports with isort..."
	@uv tool run isort@latest --settings-path pyproject.toml .
	@echo "✅ Code formatted successfully!"

# Type checking only (project-scoped tools)
typecheck:
	@echo "🔍 Running type checking with project-scoped mypy..."
	@uv tool run mypy@latest --config-file pyproject.toml src/
	@echo "✅ Type checking completed!"

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