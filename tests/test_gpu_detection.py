"""
🧪 GPU Detection and Acceleration Tests

Tests for GPU detection, MPS acceleration, and device selection.
"""

import platform
from unittest.mock import Mock, patch

import pytest
import torch

from src.translation.interface.base_translator import BaseTranslator
from src.translation.interface.pytorch_translator import PyTorchTranslator


class TestGPUDetection:
    """Test GPU detection and device selection."""

    def test_torch_installation(self):
        """Test that PyTorch is properly installed."""
        assert torch.__version__ is not None
        assert len(torch.__version__) > 0

    def test_mps_availability_on_apple_silicon(self):
        """Test MPS availability on Apple Silicon."""
        if platform.machine() == "arm64" and platform.system() == "Darwin":
            assert hasattr(torch.backends, "mps")
            assert torch.backends.mps.is_available()
            assert torch.backends.mps.is_built()
        else:
            pytest.skip("Not running on Apple Silicon")

    @pytest.mark.gpu
    def test_mps_basic_operation(self, skip_if_no_mps):
        """Test basic MPS operation."""
        device = torch.device("mps")
        x = torch.randn(100, 100, device=device)
        y = torch.randn(100, 100, device=device)
        z = torch.mm(x, y)

        assert z.device.type == "mps"
        assert z.shape == (100, 100)

    def test_base_translator_device_resolution(self, optimal_device):
        """Test BaseTranslator device resolution logic."""

        class MockTranslator(BaseTranslator):
            def load_model(self):
                return True

            def translate_text(self, text, progress_callback=None):
                return Mock()

            def translate_batch(self, texts, progress_callback=None):
                return []

            def unload_model(self):
                pass

        # Test auto detection
        translator = MockTranslator("dummy", "ja", "en", device="auto")
        assert translator.device in ["cuda", "mps", "cpu"]

        # Test explicit device
        translator_cpu = MockTranslator("dummy", "ja", "en", device="cpu")
        assert translator_cpu.device == "cpu"

    @pytest.mark.gpu
    def test_pytorch_translator_device_detection(self, optimal_device):
        """Test PyTorchTranslator optimal device detection."""
        translator = PyTorchTranslator(
            model_name="dummy", source_lang="ja", target_lang="en", device="auto"
        )

        assert translator.optimal_device in ["cuda", "mps", "cpu"]

        # On systems with GPU, should not be CPU unless forced
        gpu_available = torch.cuda.is_available() or (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )

        if gpu_available:
            assert translator.optimal_device != "cpu"

    def test_force_gpu_behavior(self):
        """Test force_gpu parameter behavior."""
        # Mock no GPU available
        with patch("torch.cuda.is_available", return_value=False), patch(
            "torch.backends.mps.is_available", return_value=False
        ):

            # Should raise error when force_gpu=True and no GPU
            with pytest.raises(RuntimeError, match="GPU acceleration required"):
                PyTorchTranslator(
                    model_name="dummy",
                    source_lang="ja",
                    target_lang="en",
                    device="auto",
                    force_gpu=True,
                )


class TestSystemInfo:
    """Test system information detection."""

    def test_platform_detection(self):
        """Test platform detection."""
        system = platform.system()
        machine = platform.machine()

        assert system in ["Darwin", "Linux", "Windows"]
        assert isinstance(machine, str)
        assert len(machine) > 0

    def test_python_version(self):
        """Test Python version compatibility."""
        version = platform.python_version_tuple()
        major, minor = int(version[0]), int(version[1])

        # Project requires Python 3.8+
        assert major == 3
        assert minor >= 8

    def test_virtual_environment(self):
        """Test virtual environment detection."""
        import sys

        # Should be running in virtual environment (uv)
        in_venv = hasattr(sys, "real_prefix") or (
            hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
        )

        assert in_venv, "Tests should run in virtual environment"


class TestHardwareOptimization:
    """Test hardware-specific optimizations."""

    @pytest.mark.gpu
    @pytest.mark.mps
    def test_mps_memory_efficiency(self, skip_if_no_mps):
        """Test MPS memory efficiency."""
        device = torch.device("mps")

        # Test different tensor sizes
        sizes = [(100, 100), (500, 500), (1000, 1000)]

        for size in sizes:
            x = torch.randn(*size, device=device, dtype=torch.float32)
            y = torch.randn(*size, device=device, dtype=torch.float32)

            # Should not raise memory errors for reasonable sizes
            z = torch.mm(x, y)
            assert z.device.type == "mps"

            # Cleanup
            del x, y, z

    @pytest.mark.gpu
    @pytest.mark.cuda
    def test_cuda_optimization(self, skip_if_no_cuda):
        """Test CUDA optimization features."""
        device = torch.device("cuda")

        # Test basic operations
        x = torch.randn(100, 100, device=device)
        y = torch.randn(100, 100, device=device)
        z = torch.mm(x, y)

        assert z.device.type == "cuda"

        # Test half precision
        x_half = x.half()
        assert x_half.dtype == torch.float16

    def test_cpu_fallback(self):
        """Test CPU fallback behavior."""
        device = torch.device("cpu")

        # Should always work regardless of hardware
        x = torch.randn(100, 100, device=device)
        y = torch.randn(100, 100, device=device)
        z = torch.mm(x, y)

        assert z.device.type == "cpu"
        assert z.shape == (100, 100)
