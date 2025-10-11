"""
🔧 Pytest Configuration and Fixtures

Common fixtures and configuration for all tests.
"""

import pytest
import sys
import os
from pathlib import Path
import torch
from unittest.mock import Mock

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture(scope="session")
def project_root():
    """Get the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def test_data_dir():
    """Get the test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def sample_japanese_texts():
    """Sample Japanese texts for testing translation."""
    return [
        "こんにちは、元気ですか？",
        "今日はいい天気ですね。",
        "アニメを見るのが好きです。",
        "この小説はとても面白いです。",
        "彼は学校に行きました。",
        "桜の花が美しく咲いています。",
    ]


@pytest.fixture(scope="session")  
def sample_english_texts():
    """Sample English texts for testing."""
    return [
        "Hello, how are you?",
        "The weather is nice today.",
        "I like watching anime.",
        "This novel is very interesting.",
        "He went to school.",
        "The cherry blossoms are blooming beautifully.",
    ]


@pytest.fixture(scope="session")
def gpu_available():
    """Check if GPU acceleration is available."""
    cuda_available = torch.cuda.is_available()
    mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    return {
        "cuda": cuda_available,
        "mps": mps_available,
        "any": cuda_available or mps_available
    }


@pytest.fixture(scope="session")
def optimal_device(gpu_available):
    """Get the optimal device for testing."""
    if gpu_available["cuda"]:
        return "cuda"
    elif gpu_available["mps"]:
        return "mps"
    else:
        return "cpu"


@pytest.fixture
def mock_progress_callback():
    """Mock progress callback for testing."""
    return Mock()


@pytest.fixture
def sample_translation_config():
    """Sample translation configuration."""
    return {
        "ja_to_en_model": "Helsinki-NLP/opus-mt-ja-en",
        "en_to_zh_model": "Helsinki-NLP/opus-mt-en-zh",
        "device": "auto",
        "batch_size": 4,
        "max_length": 512
    }


@pytest.fixture
def sample_asr_config():
    """Sample ASR configuration."""
    return {
        "model_name": "openai/whisper-large-v3",
        "device": "auto",
        "batch_size": 1,
        "language": "ja",
        "return_timestamps": True
    }


@pytest.fixture(scope="session")
def skip_if_no_gpu(gpu_available):
    """Skip test if no GPU is available."""
    if not gpu_available["any"]:
        pytest.skip("No GPU acceleration available")


@pytest.fixture(scope="session")
def skip_if_no_mps(gpu_available):
    """Skip test if MPS is not available."""
    if not gpu_available["mps"]:
        pytest.skip("Apple Silicon MPS not available")


@pytest.fixture(scope="session")
def skip_if_no_cuda(gpu_available):
    """Skip test if CUDA is not available."""
    if not gpu_available["cuda"]:
        pytest.skip("NVIDIA CUDA not available")


class MockTranslationResult:
    """Mock translation result for testing."""
    
    def __init__(self, original_text: str, translated_text: str, confidence: float = 0.9):
        self.original_text = original_text
        self.translated_text = translated_text
        self.confidence = confidence
        self.source_lang = "ja"
        self.target_lang = "en"


@pytest.fixture
def mock_translation_result():
    """Mock translation result."""
    return MockTranslationResult("こんにちは", "Hello")


# Test markers for different test categories
pytestmark = pytest.mark.filterwarnings("ignore:.*:UserWarning")


def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    # Add custom markers
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "gpu: requires GPU acceleration")  
    config.addinivalue_line("markers", "integration: integration test")
    config.addinivalue_line("markers", "unit: unit test")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add unit marker to all tests by default
        if not any(marker.name in ["integration", "slow", "gpu"] for marker in item.iter_markers()):
            item.add_marker(pytest.mark.unit)
        
        # Add gpu marker to GPU-related tests
        if "gpu" in item.name.lower() or "mps" in item.name.lower() or "cuda" in item.name.lower():
            item.add_marker(pytest.mark.gpu)
        
        # Add slow marker to model loading tests
        if "model" in item.name.lower() and "load" in item.name.lower():
            item.add_marker(pytest.mark.slow)


def pytest_runtest_setup(item):
    """Setup for individual test runs."""
    # Skip GPU tests if no GPU available
    if "gpu" in [marker.name for marker in item.iter_markers()]:
        if not (torch.cuda.is_available() or 
                (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())):
            pytest.skip("GPU acceleration not available")