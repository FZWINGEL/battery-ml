"""BiLSTM with self-attention for sequence modeling."""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any

from .base import BaseModel
from .registry import ModelRegistry


@ModelRegistry.register("lstm_attn")
class LSTMAttentionModel(BaseModel):
    """BiLSTM with self-attention for sequence modeling.
    
    Suitable for:
    - Variable length sequences
    - Capturing long-range dependencies
    - Interpretable attention weights
    
    Example usage:
        >>> model = LSTMAttentionModel(input_dim=5, hidden_dim=64)
        >>> output = model(x)  # x shape: (batch, seq_len, input_dim)
    """
    
    def __init__(self,
                 input_dim: int,
                 output_dim: int = 1,
                 hidden_dim: int = 64,
                 num_layers: int = 2,
                 num_heads: int = 4,
                 dropout: float = 0.1):
        """Initialize the model.
        
        Args:
            input_dim: Number of input features per timestep
            output_dim: Number of outputs
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__(input_dim, output_dim)
        
        self.hidden_dim = hidden_dim
        
        self.lstm = nn.LSTM(
            input_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        
        self.attention = nn.MultiheadAttention(
            hidden_dim * 2, num_heads,
            batch_first=True,
            dropout=dropout,
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        
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
            t: Ignored (for interface compatibility)
            mask: Optional attention mask
        
        Returns:
            Output tensor of shape (batch, output_dim)
        """
        # x: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden*2)
        
        # Self-attention
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out, 
                                                  key_padding_mask=mask)
        self._last_attn_weights = attn_weights.detach()
        
        # Residual connection + layer norm
        attn_out = self.layer_norm(attn_out + lstm_out)
        
        # Decode all timesteps
        out = self.fc(attn_out)
        return out
    
    def explain(self, x: torch.Tensor, **kwargs) -> Dict[str, Any]:
        """Return attention weights.
        
        Args:
            x: Input tensor
        
        Returns:
            Dictionary with attention weights
        """
        _ = self.forward(x)
        return {
            'attention_weights': self._last_attn_weights.cpu().numpy() if self._last_attn_weights is not None else None,
        }
