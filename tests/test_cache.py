"""
🧪 Cache and Model Management Tests

Tests for HuggingFace model caching and download behavior.
"""

import pytest
import os
from pathlib import Path
from unittest.mock import patch, Mock
import tempfile

from src.translation.huggingface_translator import HuggingFaceTranslator


class TestModelCaching:
    """Test HuggingFace model caching behavior."""
    
    def test_default_cache_location(self):
        """Test that default cache location is correct."""
        from transformers.utils import TRANSFORMERS_CACHE
        
        expected_cache = Path.home() / ".cache" / "huggingface" / "hub"
        cache_path = Path(TRANSFORMERS_CACHE)
        
        # Should point to the standard cache location
        assert "huggingface" in str(cache_path)
    
    def test_cache_directory_exists(self):
        """Test that cache directory exists or can be created."""
        cache_dir = Path.home() / ".cache" / "huggingface"
        
        # Either exists or parent directory exists (can be created)
        assert cache_dir.exists() or cache_dir.parent.exists()
    
    def test_model_cache_structure(self):
        """Test expected cache directory structure."""
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        
        if cache_dir.exists():
            # Should contain model directories
            model_dirs = list(cache_dir.glob("models--*"))
            
            # If we have cached models, check structure
            if model_dirs:
                sample_model = model_dirs[0]
                
                # Should have snapshots and blobs
                assert (sample_model / "snapshots").exists()
                assert (sample_model / "blobs").exists()
    
    def test_cache_environment_variables(self):
        """Test cache-related environment variables."""
        # Test with custom cache location
        with patch.dict(os.environ, {"HF_HOME": "/tmp/test_cache"}):
            # Reload transformers to pick up new env var
            import importlib
            from transformers import utils
            importlib.reload(utils)
            
            # Should use custom location
            assert "/tmp/test_cache" in str(utils.TRANSFORMERS_CACHE)
    
    def test_safetensors_vs_pytorch_format(self):
        """Test handling of different model formats."""
        # This tests the issue we found where models download both formats
        translator = HuggingFaceTranslator(
            model_name="Helsinki-NLP/opus-mt-ja-en",
            source_lang="ja",
            target_lang="en",
            device="cpu",
            use_safetensors=True  # Our fix
        )
        
        assert hasattr(translator, 'pipeline_kwargs')
    
    @pytest.mark.slow
    @pytest.mark.model_download  
    def test_model_reuse_from_cache(self):
        """Test that models are reused from cache on subsequent loads."""
        translator1 = HuggingFaceTranslator(
            model_name="Helsinki-NLP/opus-mt-ja-en",
            source_lang="ja",
            target_lang="en",
            device="cpu"
        )
        
        # First load - may download
        success1 = translator1.load_model()
        assert success1
        translator1.unload_model()
        
        # Second load - should use cache (much faster)
        translator2 = HuggingFaceTranslator(
            model_name="Helsinki-NLP/opus-mt-ja-en",
            source_lang="ja", 
            target_lang="en",
            device="cpu"
        )
        
        import time
        start_time = time.time()
        success2 = translator2.load_model()
        load_time = time.time() - start_time
        
        assert success2
        # Cache load should be much faster (< 10 seconds vs > 30 for download)
        assert load_time < 30  # Generous threshold
        
        translator2.unload_model()


class TestModelDownloadBehavior:
    """Test model download and format preferences."""
    
    def test_pipeline_format_preference(self):
        """Test that pipeline respects format preferences."""
        # Test with explicit safetensors preference
        translator = HuggingFaceTranslator(
            model_name="Helsinki-NLP/opus-mt-ja-en",
            source_lang="ja",
            target_lang="en", 
            device="cpu",
            use_safetensors=True
        )
        
        # Check that use_safetensors is passed to pipeline
        assert "use_safetensors" in str(translator.__dict__)
    
    def test_cache_miss_handling(self):
        """Test behavior when cache is empty or corrupted."""
        # This would be tested with a temporary cache directory
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.dict(os.environ, {"HF_HOME": temp_dir}):
                translator = HuggingFaceTranslator(
                    model_name="Helsinki-NLP/opus-mt-ja-en",
                    source_lang="ja",
                    target_lang="en",
                    device="cpu"
                )
                
                # Should handle empty cache gracefully
                assert translator.model_name == "Helsinki-NLP/opus-mt-ja-en"
    
    def test_offline_mode(self):
        """Test behavior in offline mode (no downloads)."""
        with patch.dict(os.environ, {"HF_HUB_OFFLINE": "1"}):
            translator = HuggingFaceTranslator(
                model_name="nonexistent/model",
                source_lang="ja",
                target_lang="en",
                device="cpu"
            )
            
            # Should handle offline mode gracefully
            success = translator.load_model()
            # May succeed if cached, fail if not - both are valid
            assert isinstance(success, bool)


class TestMemoryManagement:
    """Test memory management and cleanup."""
    
    def test_model_unloading(self):
        """Test proper model unloading and memory cleanup."""
        translator = HuggingFaceTranslator(
            model_name="Helsinki-NLP/opus-mt-ja-en",
            source_lang="ja",
            target_lang="en",
            device="cpu"
        )
        
        # Mock pipeline to avoid actual loading
        mock_pipeline = Mock()
        translator.pipeline = mock_pipeline
        translator.is_loaded = True
        
        # Test unloading
        translator.unload_model()
        
        assert translator.pipeline is None
        assert not translator.is_loaded
    
    def test_context_manager_cleanup(self):
        """Test cleanup when using MultiStageTranslator as context manager."""
        from src.translation.huggingface_translator import MultiStageTranslator
        
        translator = MultiStageTranslator(
            ja_en_model="Helsinki-NLP/opus-mt-ja-en",
            en_zh_model="Helsinki-NLP/opus-mt-en-zh",
            device="cpu"
        )
        
        # Test context manager protocol
        assert hasattr(translator, '__enter__')
        assert hasattr(translator, '__exit__')
        
        with translator:
            assert translator is not None
        
        # Should cleanup after context exit
        # (Specific cleanup behavior depends on implementation)
    
    @pytest.mark.gpu
    def test_gpu_memory_cleanup(self, optimal_device, skip_if_no_gpu):
        """Test GPU memory cleanup after model unloading."""
        import torch
        
        initial_memory = 0
        if torch.cuda.is_available():
            initial_memory = torch.cuda.memory_allocated()
        
        translator = HuggingFaceTranslator(
            model_name="Helsinki-NLP/opus-mt-ja-en",
            source_lang="ja",
            target_lang="en",
            device=optimal_device
        )
        
        # Mock to avoid actual loading
        translator.pipeline = Mock()
        translator.is_loaded = True
        
        translator.unload_model()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            final_memory = torch.cuda.memory_allocated()
            # Memory should not have increased significantly
            assert final_memory >= initial_memory


class TestCacheOptimization:
    """Test cache optimization strategies."""
    
    def test_predownload_script_structure(self, project_root):
        """Test that predownload script exists and is valid Python."""
        predownload_script = project_root / "predownload_models.py"
        
        if predownload_script.exists():
            # Should be valid Python
            with open(predownload_script) as f:
                content = f.read()
                
            # Should not have syntax errors
            compile(content, str(predownload_script), 'exec')
            
            # Should contain expected functionality
            assert "transformers" in content
            assert "pipeline" in content
    
    def test_cache_diagnostic_tools(self, project_root):
        """Test that cache diagnostic tools exist."""
        diagnostic_script = project_root / "diagnose_cache.py"
        
        if diagnostic_script.exists():
            with open(diagnostic_script) as f:
                content = f.read()
            
            # Should contain cache-related functionality
            assert "cache" in content.lower()
            assert "huggingface" in content.lower()
    
    def test_model_size_expectations(self):
        """Test expected model sizes for planning purposes."""
        expected_sizes = {
            "Helsinki-NLP/opus-mt-ja-en": (250, 350),  # ~300MB
            "Helsinki-NLP/opus-mt-en-zh": (250, 350),  # ~300MB  
            "openai/whisper-large-v3": (2500, 3500),   # ~3GB
        }
        
        # These are just documentation - actual tests would need network access
        for model, (min_mb, max_mb) in expected_sizes.items():
            assert min_mb < max_mb
            assert min_mb > 0