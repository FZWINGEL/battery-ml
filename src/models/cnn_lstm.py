"""CNN-LSTM model for sequence-based degradation prediction.

Combines 1D CNN for local feature extraction, BiLSTM for temporal dependencies,
and self-attention for interpretability.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List

from .base import BaseModel
from .registry import ModelRegistry


@ModelRegistry.register("cnn_lstm")
class CNNLSTMModel(BaseModel):
    """CNN-LSTM model for sequence-to-sequence prediction.
    
    Architecture:
    1. 1D CNN layers extract local patterns from each timestep
    2. Bidirectional LSTM captures temporal dependencies
    3. Self-attention highlights important timesteps
    4. Output layer produces predictions at each timestep
    
    Suitable for:
    - Multi-target degradation prediction (LAM_NE, LAM_PE, LLI)
    - Variable length sequences
    - Interpretable attention weights
    
    Example usage:
        >>> model = CNNLSTMModel(input_dim=20, output_dim=3, hidden_dim=64)
        >>> output = model(x)  # x: (batch, seq_len, features)
    """
    
    name: str = "cnn_lstm"
    
    def __init__(self,
                 input_dim: int,
                 output_dim: int = 1,
                 hidden_dim: int = 64,
                 cnn_filters: Optional[List[int]] = None,
                 num_layers: int = 2,
                 num_heads: int = 4,
                 dropout: float = 0.1):
        """Initialize the CNN-LSTM model.
        
        Args:
            input_dim: Number of input features per timestep
            output_dim: Number of output predictions per timestep
            hidden_dim: LSTM hidden dimension
            cnn_filters: List of CNN filter sizes [first_layer, second_layer]
            num_layers: Number of LSTM layers
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__(input_dim, output_dim)
        
        if cnn_filters is None:
            cnn_filters = [64, 32]
        
        self.hidden_dim = hidden_dim
        self.cnn_filters = cnn_filters
        
        # 1D CNN layers: process features at each timestep
        self.cnn = nn.Sequential(
            nn.Conv1d(1, cnn_filters[0], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(cnn_filters[0]),
            nn.Conv1d(cnn_filters[0], cnn_filters[1], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(cnn_filters[1]),
        )
        
        # Bidirectional LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=cnn_filters[1],
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        
        # Self-attention layer
        self.attention = nn.MultiheadAttention(
            hidden_dim * 2,  # Bidirectional doubles dimension
            num_heads,
            batch_first=True,
            dropout=dropout,
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        
        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        
        self._last_attn_weights: Optional[torch.Tensor] = None
    
    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            t: Ignored (for interface compatibility with ODE models)
            mask: Optional attention mask
        
        Returns:
            Output tensor of shape (batch, seq_len, output_dim)
        """
        batch_size, seq_len, feat_dim = x.size()
        
        # 1. CNN feature extraction per timestep
        # Reshape: (batch * seq_len, 1, feat_dim)
        x_reshaped = x.view(batch_size * seq_len, 1, feat_dim)
        
        # Apply CNN: (batch * seq_len, cnn_filters[-1], feat_dim)
        cnn_out = self.cnn(x_reshaped)
        
        # Average pool over feature dimension: (batch * seq_len, cnn_filters[-1])
        cnn_out = cnn_out.mean(dim=2)
        
        # Reshape for LSTM: (batch, seq_len, cnn_filters[-1])
        cnn_out = cnn_out.view(batch_size, seq_len, self.cnn_filters[-1])
        
        # 2. LSTM temporal processing: (batch, seq_len, hidden_dim * 2)
        lstm_out, _ = self.lstm(cnn_out)
        
        # 3. Self-attention
        attn_out, attn_weights = self.attention(
            lstm_out, lstm_out, lstm_out,
            key_padding_mask=mask
        )
        self._last_attn_weights = attn_weights.detach()
        
        # Residual connection + layer norm
        attn_out = self.layer_norm(attn_out + lstm_out)
        
        # 4. Decode all timesteps: (batch, seq_len, output_dim)
        output = self.fc(attn_out)
        
        return output
    
    def explain(self, x: torch.Tensor, t: Optional[torch.Tensor] = None,
                **kwargs) -> Dict[str, Any]:
        """Return attention weights for interpretability.
        
        Args:
            x: Input tensor
            t: Ignored (for interface compatibility)
        
        Returns:
            Dictionary with attention weights
        """
        _ = self.forward(x, t)
        return {
            'attention_weights': self._last_attn_weights.cpu().numpy() if self._last_attn_weights is not None else None,
            'hidden_dim': self.hidden_dim,
            'cnn_filters': self.cnn_filters,
        }
