"""Tests for models."""

import pytest
import numpy as np
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestMLPModel:
    """Tests for MLPModel."""
    
    def test_forward(self):
        """Test forward pass."""
        from src.models.mlp import MLPModel
        
        model = MLPModel(input_dim=10, output_dim=1, hidden_dims=[32, 16])
        x = torch.randn(8, 10)
        
        output = model(x)
        
        assert output.shape == (8, 1)
    
    def test_predict(self):
        """Test predict method."""
        from src.models.mlp import MLPModel
        
        model = MLPModel(input_dim=10, output_dim=1)
        x = torch.randn(8, 10)
        
        output = model.predict(x)
        
        assert isinstance(output, np.ndarray)
        assert output.shape == (8, 1)


class TestLSTMAttentionModel:
    """Tests for LSTMAttentionModel."""
    
    def test_forward(self):
        """Test forward pass with sequences."""
        from src.models.lstm_attn import LSTMAttentionModel
        
        model = LSTMAttentionModel(
            input_dim=5, 
            output_dim=1, 
            hidden_dim=32, 
            num_layers=2
        )
        
        # (batch, seq_len, features)
        x = torch.randn(4, 10, 5)
        
        output = model(x)
        
        assert output.shape == (4, 1)
    
    def test_explain(self):
        """Test attention weight extraction."""
        from src.models.lstm_attn import LSTMAttentionModel
        
        model = LSTMAttentionModel(input_dim=5, hidden_dim=32)
        x = torch.randn(4, 10, 5)
        
        explanation = model.explain(x)
        
        assert 'attention_weights' in explanation
        assert explanation['attention_weights'] is not None


class TestNeuralODEModel:
    """Tests for NeuralODEModel."""
    
    @pytest.mark.skipif(
        not pytest.importorskip("torchdiffeq", reason="torchdiffeq not installed"),
        reason="torchdiffeq not installed"
    )
    def test_forward(self):
        """Test forward pass with time."""
        try:
            from src.models.neural_ode import NeuralODEModel
            
            model = NeuralODEModel(
                input_dim=5,
                output_dim=1,
                latent_dim=16,
                hidden_dim=32,
                solver='euler',  # Faster for testing
                use_adjoint=False
            )
            
            x = torch.randn(2, 10, 5)
            t = torch.linspace(0, 1, 10)
            
            output = model(x, t=t)
            
            assert output.shape == (2, 1)
        except ImportError:
            pytest.skip("torchdiffeq not installed")
    
    @pytest.mark.skipif(
        not pytest.importorskip("torchdiffeq", reason="torchdiffeq not installed"),
        reason="torchdiffeq not installed"
    )
    def test_explain_trajectory(self):
        """Test trajectory extraction."""
        try:
            from src.models.neural_ode import NeuralODEModel
            
            model = NeuralODEModel(
                input_dim=5,
                latent_dim=8,
                solver='euler',
                use_adjoint=False
            )
            
            x = torch.randn(2, 10, 5)
            t = torch.linspace(0, 1, 10)
            
            explanation = model.explain(x, t)
            
            assert 'trajectory' in explanation
            assert explanation['trajectory'].shape[0] == 10  # seq_len
        except ImportError:
            pytest.skip("torchdiffeq not installed")


class TestLGBMModel:
    """Tests for LGBMModel."""
    
    @pytest.mark.skipif(
        not pytest.importorskip("lightgbm", reason="lightgbm not installed"),
        reason="lightgbm not installed"
    )
    def test_fit_predict(self):
        """Test fit and predict."""
        try:
            from src.models.lgbm import LGBMModel
            
            X_train = np.random.randn(50, 5)
            y_train = np.random.randn(50, 1)
            X_val = np.random.randn(10, 5)
            y_val = np.random.randn(10, 1)
            
            model = LGBMModel(n_estimators=10)
            model.fit(X_train, y_train, X_val, y_val)
            
            predictions = model.predict(X_val)
            
            assert predictions.shape == (10, 1)
        except ImportError:
            pytest.skip("lightgbm not installed")
    
    @pytest.mark.skipif(
        not pytest.importorskip("lightgbm", reason="lightgbm not installed"),
        reason="lightgbm not installed"
    )
    def test_feature_importances(self):
        """Test feature importance."""
        try:
            from src.models.lgbm import LGBMModel
            
            X_train = np.random.randn(50, 5)
            y_train = np.random.randn(50, 1)
            
            model = LGBMModel(n_estimators=10)
            model.fit(X_train, y_train)
            
            importances = model.feature_importances_
            
            assert len(importances) == 5
        except ImportError:
            pytest.skip("lightgbm not installed")


class TestModelRegistry:
    """Tests for ModelRegistry."""
    
    def test_list_available(self):
        """Test listing available models."""
        from src.models.registry import ModelRegistry
        
        # Import to register
        from src.models import mlp, lstm_attn
        
        available = ModelRegistry.list_available()
        
        assert 'mlp' in available
        assert 'lstm_attn' in available
    
    def test_get_model(self):
        """Test getting model by name."""
        from src.models.registry import ModelRegistry
        from src.models import mlp  # Register
        
        model = ModelRegistry.get('mlp', input_dim=10, hidden_dims=[32])
        
        assert model.input_dim == 10
