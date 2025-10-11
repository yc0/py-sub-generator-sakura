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
	@echo "E2E Testing:"
	@echo "  test-e2e      Run SakuraLLM pipeline demo"
	@echo "  test-e2e-integration  Run integration tests"
	@echo "  test-e2e-all  Run all e2e verification tests"
	@echo ""
	@echo "Examples & Demos:"
	@echo "  demo-sakura   Run SakuraLLM translation demo"
	@echo "  demo-14b      Compare 7B vs 14B models"
	@echo "  demo-3lang    Three-language pipeline demo"
	@echo "  download-models  Download SakuraLLM models"
	@echo ""
	@echo "Examples:"
	@echo "  make test-e2e          # Quick e2e verification"
	@echo "  make demo-sakura       # Try SakuraLLM pipeline"
	@echo "  make test-coverage     # Generate coverage report"

# Testing commands (project-scoped)
test:
	@echo "🧪 Running fast tests..."
	uv run python tools/run_tests.py --type fast

test-unit:
	@echo "🧪 Running unit tests..."
	uv run python tools/run_tests.py --type unit -v

test-gpu:
	@echo "🧪 Running GPU tests..."
	uv run python tools/run_tests.py --type gpu --gpu -v

test-slow:
	@echo "🧪 Running slow tests (may download models)..."
	uv run python tools/run_tests.py --type slow -v

test-all:
	@echo "🧪 Running all tests..."
	uv run python tools/run_tests.py --type all --gpu -v

test-coverage:
	@echo "🧪 Running tests with coverage..."
	uv run python tools/run_tests.py --type fast --coverage
	@echo "📊 Coverage report generated in htmlcov/"

# E2E Testing - The commands I keep running to verify integrity
test-e2e:
	@echo "🔄 Running end-to-end SakuraLLM pipeline test..."
	uv run python examples/demo_sakura_translation.py

test-e2e-integration:
	@echo "🔄 Running integration test pipeline..."
	uv run python -m pytest tests/integration/test_audio_pipeline_e2e.py::TestAudioPipelineE2E::test_complete_audio_pipeline_hf_only -v

test-e2e-all:
	@echo "🔄 Running all e2e verification tests..."
	@echo "1️⃣ SakuraLLM Pipeline Test:"
	@$(MAKE) test-e2e
	@echo ""
	@echo "2️⃣ Integration Test:"
	@$(MAKE) test-e2e-integration
	@echo ""
	@echo "✅ All E2E tests completed!"

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

# Examples and Demos
demo-sakura:
	@echo "🌸 Running SakuraLLM translation demo..."
	uv run python examples/demo_sakura_translation.py

demo-14b:
	@echo "🔥 Running SakuraLLM 14B vs 7B comparison..."
	uv run python examples/demo_sakura_14b_test.py

demo-3lang:
	@echo "🌍 Running three-language pipeline demo..."
	uv run python examples/demo_three_languages.py

# Model management  
download-models:
	@echo "📥 Downloading SakuraLLM models..."
	uv run python examples/download_sakura_models.py

download-7b:
	@echo "📥 Downloading SakuraLLM 7B model..."
	echo "7b" | uv run python examples/download_sakura_models.py

download-14b:
	@echo "📥 Downloading SakuraLLM 14B model..."
	echo "14b" | uv run python examples/download_sakura_models.py

benchmark:
	@echo "⚡ Running performance benchmarks..."
	uv run python tools/run_tests.py --type slow -k benchmark -v

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

# Apple Silicon setup
setup-apple:
	@echo "🍎 Running Apple Silicon optimization setup..."
	uv run python tools/setup_apple_silicon.py

# CI/CD helpers
ci-test:
	@echo "🤖 Running CI tests..."
	uv run python tools/run_tests.py --type no-download --coverage

check-setup:
	@echo "🔍 Checking project setup..."
	uv run python -c "import torch; print(f'PyTorch: {torch.__version__}')"
	uv run python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
	uv run python -c "import torch; print(f'MPS Available: {torch.backends.mps.is_available() if hasattr(torch.backends, \"mps\") else False}')"