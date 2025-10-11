"""
🧪 Performance and Benchmark Tests

Tests for translation performance and benchmarking.
"""

import time
from unittest.mock import Mock

import pytest

from src.translation.huggingface_translator import HuggingFaceTranslator


class TestPerformanceBenchmarks:
    """Test translation performance benchmarks."""

    @pytest.mark.slow
    @pytest.mark.model_download
    def test_huggingface_translation_speed(self, sample_japanese_texts):
        """Test HuggingFace translator speed."""
        translator = HuggingFaceTranslator(
            model_name="Helsinki-NLP/opus-mt-ja-en",
            source_lang="ja",
            target_lang="en",
            device="cpu",
            batch_size=4,
        )

        success = translator.load_model()
        assert success

        # Warm up
        translator.translate_text(sample_japanese_texts[0])

        # Benchmark single translation
        start_time = time.time()
        result = translator.translate_text(sample_japanese_texts[0])
        single_time = time.time() - start_time

        assert single_time < 5.0  # Should complete in under 5 seconds on CPU
        assert result.translated_text != result.original_text

        # Benchmark batch translation
        start_time = time.time()
        results = translator.translate_batch(sample_japanese_texts)
        batch_time = time.time() - start_time

        assert len(results) == len(sample_japanese_texts)
        texts_per_second = len(sample_japanese_texts) / batch_time
        assert texts_per_second > 0.5  # At least 0.5 texts per second

        translator.unload_model()

        return {
            "single_time": single_time,
            "batch_time": batch_time,
            "texts_per_second": texts_per_second,
        }

    @pytest.mark.gpu
    @pytest.mark.slow
    @pytest.mark.model_download
    def test_gpu_vs_cpu_performance(self, sample_japanese_texts, skip_if_no_gpu):
        """Compare GPU vs CPU performance."""
        # CPU benchmark
        cpu_translator = HuggingFaceTranslator(
            model_name="Helsinki-NLP/opus-mt-ja-en",
            source_lang="ja",
            target_lang="en",
            device="cpu",
            batch_size=2,
        )

        cpu_translator.load_model()

        start_time = time.time()
        cpu_results = cpu_translator.translate_batch(sample_japanese_texts[:3])
        cpu_time = time.time() - start_time

        cpu_translator.unload_model()

        # GPU benchmark
        gpu_translator = HuggingFaceTranslator(
            model_name="Helsinki-NLP/opus-mt-ja-en",
            source_lang="ja",
            target_lang="en",
            device="auto",  # Should use GPU
            batch_size=2,
        )

        gpu_translator.load_model()

        start_time = time.time()
        gpu_results = gpu_translator.translate_batch(sample_japanese_texts[:3])
        gpu_time = time.time() - start_time

        gpu_translator.unload_model()

        # GPU should be faster or at least comparable
        speedup = cpu_time / gpu_time

        # On Apple Silicon MPS, expect at least some improvement
        # Note: First run might be slower due to MPS compilation
        assert speedup > 0.5  # GPU shouldn't be more than 2x slower

        # Both should produce same quality results
        assert len(cpu_results) == len(gpu_results)

        return {"cpu_time": cpu_time, "gpu_time": gpu_time, "speedup": speedup}

    def test_batch_size_scaling(self, sample_japanese_texts):
        """Test how batch size affects performance."""
        batch_sizes = [1, 2, 4, 8]
        results = {}

        for batch_size in batch_sizes:
            translator = HuggingFaceTranslator(
                model_name="Helsinki-NLP/opus-mt-ja-en",
                source_lang="ja",
                target_lang="en",
                device="cpu",
                batch_size=batch_size,
            )

            # Mock to avoid actual model loading
            translator.pipeline = Mock()
            translator.pipeline.return_value = [
                {"translation_text": f"Translation {i}"}
                for i in range(len(sample_japanese_texts))
            ]
            translator.is_loaded = True

            start_time = time.time()
            translator.translate_batch(sample_japanese_texts)
            batch_time = time.time() - start_time

            results[batch_size] = batch_time

        # Results should show some relationship with batch size
        assert len(results) == len(batch_sizes)
        assert all(t > 0 for t in results.values())

    def test_memory_usage_scaling(self, sample_japanese_texts):
        """Test memory usage with different input sizes."""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        translator = HuggingFaceTranslator(
            model_name="Helsinki-NLP/opus-mt-ja-en",
            source_lang="ja",
            target_lang="en",
            device="cpu",
        )

        # Mock to control memory usage
        translator.pipeline = Mock()
        translator.is_loaded = True

        # Test with increasing text lengths
        for multiplier in [1, 10, 100]:
            long_texts = [text * multiplier for text in sample_japanese_texts]
            translator.translate_batch(long_texts)

            current_memory = process.memory_info().rss
            memory_increase = current_memory - initial_memory

            # Memory usage should be reasonable (< 1GB increase for mocked tests)
            assert memory_increase < 1024 * 1024 * 1024


class TestLoadingPerformance:
    """Test model loading performance."""

    @pytest.mark.slow
    @pytest.mark.model_download
    def test_model_loading_time(self):
        """Test model loading performance."""
        translator = HuggingFaceTranslator(
            model_name="Helsinki-NLP/opus-mt-ja-en",
            source_lang="ja",
            target_lang="en",
            device="cpu",
        )

        # First load (may include download/cache)
        start_time = time.time()
        success = translator.load_model()
        load_time = time.time() - start_time

        assert success
        # Should load in reasonable time (depends on cache state)
        assert load_time < 120  # 2 minutes max including potential download

        translator.unload_model()

        # Second load (should use cache)
        start_time = time.time()
        success = translator.load_model()
        cached_load_time = time.time() - start_time

        assert success
        # Cached load should be much faster
        assert cached_load_time < 30  # 30 seconds max from cache

        translator.unload_model()

        return {"initial_load_time": load_time, "cached_load_time": cached_load_time}

    def test_concurrent_loading(self):
        """Test concurrent model loading behavior."""
        import queue
        import threading

        results_queue = queue.Queue()

        def load_translator(device):
            translator = HuggingFaceTranslator(
                model_name="Helsinki-NLP/opus-mt-ja-en",
                source_lang="ja",
                target_lang="en",
                device=device,
            )

            start_time = time.time()
            success = translator.load_model()
            load_time = time.time() - start_time

            results_queue.put(
                {"device": device, "success": success, "load_time": load_time}
            )

            if success:
                translator.unload_model()

        # Start concurrent loading
        threads = []
        for device in ["cpu", "cpu"]:  # Two CPU instances
            thread = threading.Thread(target=load_translator, args=(device,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join(timeout=60)  # 1 minute timeout

        # Check results
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())

        assert len(results) == 2
        assert all(r["success"] for r in results)


class TestScalabilityTests:
    """Test system scalability and limits."""

    def test_large_batch_handling(self):
        """Test handling of large batches."""
        # Generate large batch
        large_batch = ["こんにちは"] * 100

        translator = HuggingFaceTranslator(
            model_name="Helsinki-NLP/opus-mt-ja-en",
            source_lang="ja",
            target_lang="en",
            device="cpu",
            batch_size=8,
        )

        # Mock to avoid resource usage
        translator.pipeline = Mock()
        translator.pipeline.return_value = [
            {"translation_text": "Hello"} for _ in large_batch
        ]
        translator.is_loaded = True

        # Should handle large batches without crashing
        results = translator.translate_batch(large_batch)
        assert len(results) == 100

    def test_long_text_handling(self):
        """Test handling of very long texts."""
        # Create very long text
        long_text = "これは非常に長いテキストです。" * 100

        translator = HuggingFaceTranslator(
            model_name="Helsinki-NLP/opus-mt-ja-en",
            source_lang="ja",
            target_lang="en",
            device="cpu",
        )

        # Mock to avoid resource usage
        translator.pipeline = Mock()
        translator.pipeline.return_value = [
            {"translation_text": "This is a very long text."}
        ]
        translator.is_loaded = True

        # Should handle long text gracefully
        result = translator.translate_text(long_text)
        assert result.original_text == long_text
        assert len(result.translated_text) > 0

    @pytest.mark.gpu
    def test_gpu_memory_limits(self, skip_if_no_gpu):
        """Test GPU memory limit handling."""
        import torch

        if torch.cuda.is_available():
            # Get GPU memory info
            total_memory = torch.cuda.get_device_properties(0).total_memory
            current_memory = torch.cuda.memory_allocated()

            # Should have reasonable memory available
            available_memory = total_memory - current_memory
            assert available_memory > 1024 * 1024 * 1024  # At least 1GB

        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # MPS doesn't have direct memory queries, but should work
            device = torch.device("mps")

            # Test reasonable tensor creation
            x = torch.randn(1000, 1000, device=device)
            assert x.device.type == "mps"

            del x  # Cleanup
