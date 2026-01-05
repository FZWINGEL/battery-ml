"""Tests for Sample dataclass and cache."""

import pytest
import numpy as np
import torch
import sys
from pathlib import Path
import tempfile

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestSample:
    """Tests for Sample dataclass."""
    
    def test_create_sample(self):
        """Test sample creation."""
        from src.pipelines.sample import Sample
        
        sample = Sample(
            meta={'cell_id': 'A', 'temperature_C': 25},
            x=np.array([1.0, 2.0, 3.0]),
            y=np.array([0.95]),
        )
        
        assert sample.meta['cell_id'] == 'A'
        assert sample.x is not None
        assert sample.y is not None
    
    def test_to_tensor(self):
        """Test numpy to tensor conversion."""
        from src.pipelines.sample import Sample
        
        sample = Sample(
            meta={'cell_id': 'A'},
            x=np.array([1.0, 2.0, 3.0]),
            y=np.array([0.95]),
        )
        
        sample.to_tensor()
        
        assert isinstance(sample.x, torch.Tensor)
        assert isinstance(sample.y, torch.Tensor)
        assert sample.x.dtype == torch.float32
    
    def test_to_device(self):
        """Test device transfer."""
        from src.pipelines.sample import Sample
        
        sample = Sample(
            meta={'cell_id': 'A'},
            x=torch.tensor([1.0, 2.0, 3.0]),
            y=torch.tensor([0.95]),
        )
        
        sample.to_device('cpu')
        
        assert sample.x.device == torch.device('cpu')
    
    def test_feature_dim(self):
        """Test feature dimension property."""
        from src.pipelines.sample import Sample
        
        sample = Sample(
            x=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        )
        
        assert sample.feature_dim == 5
    
    def test_seq_len(self):
        """Test sequence length property."""
        from src.pipelines.sample import Sample
        
        # Static
        sample_static = Sample(x=np.array([1.0, 2.0, 3.0]))
        assert sample_static.seq_len is None
        
        # Sequence
        sample_seq = Sample(x=np.random.randn(10, 5))
        assert sample_seq.seq_len == 10
    
    def test_dict_features(self):
        """Test dictionary features."""
        from src.pipelines.sample import Sample
        
        sample = Sample(
            x={
                'summary': np.array([1.0, 2.0, 3.0]),
                'ica': np.array([4.0, 5.0]),
            }
        )
        
        sample.to_tensor()
        
        assert isinstance(sample.x['summary'], torch.Tensor)
        assert isinstance(sample.x['ica'], torch.Tensor)


class TestPipelineCache:
    """Tests for PipelineCache."""
    
    def test_cache_hit_miss(self):
        """Test cache hit and miss."""
        from src.pipelines.cache import PipelineCache
        
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = PipelineCache(cache_dir=Path(tmpdir))
            
            # Miss
            result = cache.get(5, 'A', 0, 'test', {'param': 1})
            assert result is None
            
            # Set
            cache.set(5, 'A', 0, 'test', {'param': 1}, {'data': 'test'})
            
            # Hit
            result = cache.get(5, 'A', 0, 'test', {'param': 1})
            assert result == {'data': 'test'}
    
    def test_get_or_compute(self):
        """Test get_or_compute pattern."""
        from src.pipelines.cache import PipelineCache
        
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = PipelineCache(cache_dir=Path(tmpdir))
            
            call_count = [0]
            
            def expensive_fn():
                call_count[0] += 1
                return 'computed'
            
            # First call computes
            result1 = cache.get_or_compute(5, 'A', 0, 'test', {}, expensive_fn)
            assert result1 == 'computed'
            assert call_count[0] == 1
            
            # Second call uses cache
            result2 = cache.get_or_compute(5, 'A', 0, 'test', {}, expensive_fn)
            assert result2 == 'computed'
            assert call_count[0] == 1  # Not called again
    
    def test_clear(self):
        """Test cache clearing."""
        from src.pipelines.cache import PipelineCache
        
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = PipelineCache(cache_dir=Path(tmpdir))
            
            cache.set(5, 'A', 0, 'test', {}, 'data')
            cache.set(5, 'B', 0, 'test', {}, 'data')
            
            count = cache.clear()
            
            assert count == 2
    
    def test_stats(self):
        """Test cache statistics."""
        from src.pipelines.cache import PipelineCache
        
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = PipelineCache(cache_dir=Path(tmpdir))
            
            cache.get(5, 'A', 0, 'test', {})  # Miss
            cache.set(5, 'A', 0, 'test', {}, 'data')
            cache.get(5, 'A', 0, 'test', {})  # Hit
            
            stats = cache.get_stats()
            
            assert stats['hits'] == 1
            assert stats['misses'] == 1
            assert stats['hit_rate'] == 0.5
