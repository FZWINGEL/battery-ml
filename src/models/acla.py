"""ACLA: Attention-CNN-LSTM-ANODE hybrid model for sequence-based degradation prediction."""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List
import logging

from .base import BaseModel
from .registry import ModelRegistry

logger = logging.getLogger(__name__)

try:
    from torchdiffeq import odeint, odeint_adjoint
    HAS_TORCHDIFFEQ = True
except ImportError:
    HAS_TORCHDIFFEQ = False
    logger.warning("torchdiffeq not installed. Install with: pip install torchdiffeq")


class AttentionLayer(nn.Module):
    """Temporal attention layer for sequence data.
    
    Applies attention mechanism across timesteps to focus on important
    parts of the sequence.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_heads: int = 4):
        """Initialize attention layer.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension for attention (must be divisible by num_heads)
            num_heads: Number of attention heads
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Project input to hidden_dim (divisible by num_heads) before attention
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads,
            batch_first=True,
            dropout=0.1,
        )
        
        # Project back to input_dim after attention
        self.output_proj = nn.Linear(hidden_dim, input_dim)
        
        self.layer_norm = nn.LayerNorm(input_dim)
    
    def forward(self, x: torch.Tensor):
        """Apply attention to input sequence.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
        
        Returns:
            attended: Attended sequence (batch, seq_len, input_dim)
            attn_weights: Attention weights (batch, seq_len, seq_len)
        """
        # Project input to hidden_dim (divisible by num_heads)
        x_proj = self.input_proj(x)
        
        # Self-attention in hidden_dim space
        attn_out, attn_weights = self.attention(x_proj, x_proj, x_proj)
        
        # Project back to input_dim
        attn_out = self.output_proj(attn_out)
        
        # Residual connection + layer norm
        attended = self.layer_norm(attn_out + x)
        
        return attended, attn_weights


class CNNLSTMEncoder(nn.Module):
    """CNN-LSTM encoder for feature extraction from sequences.
    
    Uses 1D CNN to extract local patterns, then LSTM to capture
    temporal dependencies.
    """
    
    def __init__(self, input_dim: int, cnn_filters: Optional[List[int]] = None,
                 lstm_hidden: int = 64):
        """Initialize CNN-LSTM encoder.
        
        Args:
            input_dim: Input feature dimension
            cnn_filters: List of CNN filter sizes [first_layer, second_layer]
            lstm_hidden: LSTM hidden dimension
        """
        super().__init__()
        if cnn_filters is None:
            cnn_filters = [64, 32]
        self.input_dim = input_dim
        self.cnn_filters = cnn_filters
        self.lstm_hidden = lstm_hidden
        
        # 1D CNN layers: process features at each timestep
        self.cnn = nn.Sequential(
            nn.Conv1d(1, cnn_filters[0], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(cnn_filters[0], cnn_filters[1], kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=cnn_filters[1],
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input sequence.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
        
        Returns:
            Encoded representation of shape (batch, seq_len, lstm_hidden)
        """
        batch_size, seq_len, feat_dim = x.size()
        
        # Reshape for CNN: treat each timestep independently
        # (batch * seq_len, 1, feat_dim)
        x_reshaped = x.view(batch_size * seq_len, 1, feat_dim)
        
        # CNN feature extraction
        cnn_out = self.cnn(x_reshaped)  # (batch * seq_len, cnn_filters[1], feat_dim)
        
        # Reshape for LSTM: (batch, seq_len, cnn_filters[1])
        # Average over feature dimension to get single value per timestep
        cnn_out = cnn_out.mean(dim=2)  # (batch * seq_len, cnn_filters[1])
        cnn_out = cnn_out.view(batch_size, seq_len, self.cnn_filters[1])
        
        # LSTM processing
        lstm_out, _ = self.lstm(cnn_out)  # (batch, seq_len, lstm_hidden)
        
        return lstm_out


class ODEFunc(nn.Module):
    """Neural network defining ODE dynamics with augmented dimensions.
    
    Implements ANODE (Augmented Neural ODE) by including augmented
    dimensions in the state space.
    """
    
    def __init__(self, hidden_dim: int, augment_dim: int = 20, 
                 ode_hidden_dim: int = 128):
        """Initialize ODE function.
        
        Args:
            hidden_dim: Dimension of hidden state
            augment_dim: Number of augmented dimensions
            ode_hidden_dim: Hidden dimension in ODE network
        """
        super().__init__()
        total_dim = hidden_dim + augment_dim
        
        self.net = nn.Sequential(
            nn.Linear(total_dim + 1, ode_hidden_dim),  # +1 for time
            nn.Tanh(),
            nn.Linear(ode_hidden_dim, ode_hidden_dim),
            nn.Tanh(),
            nn.Linear(ode_hidden_dim, total_dim),
        )
        
        # Initialize weights with small std for stable ODE integration
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, val=0)
    
    def forward(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute dy/dt.
        
        Args:
            t: Current time (scalar or batch)
            y: Current state of shape (batch, hidden_dim + augment_dim)
        
        Returns:
            dy/dt of shape (batch, hidden_dim + augment_dim)
        """
        # Concatenate time as feature
        if t.dim() == 0:
            t_expanded = t.expand(y.shape[0], 1)
        else:
            t_expanded = t.unsqueeze(-1) if t.dim() == 1 else t
        
        yt = torch.cat([y, t_expanded], dim=-1)
        return self.net(yt)


@ModelRegistry.register("acla")
class ACLAModel(BaseModel):
    """ACLA: Attention-CNN-LSTM-ANODE hybrid model.
    
    Architecture:
    1. Attention: Temporal attention across sequence timesteps
    2. CNN-LSTM: Feature extraction and temporal modeling
    3. ANODE: Augmented Neural ODE for continuous-time dynamics
    4. Output: Sequence-to-sequence predictions
    
    Suitable for:
    - Complex sequence-based degradation prediction
    - Multi-target prediction (e.g., LAM_NE, LAM_PE, LLI)
    - Understanding which timesteps are important (attention)
    - Continuous-time trajectory modeling
    
    Example usage:
        >>> model = ACLAModel(input_dim=20, output_dim=3, hidden_dim=64)
        >>> output = model(x, t=t)  # x: (batch, seq_len, features), t: (seq_len,)
    """
    
    def __init__(self,
                 input_dim: int,
                 output_dim: int = 1,
                 hidden_dim: int = 64,
                 augment_dim: int = 20,
                 cnn_filters: Optional[List[int]] = None,
                 solver: str = "dopri5",
                 rtol: float = 1e-4,
                 atol: float = 1e-5,
                 use_adjoint: bool = False):
        """Initialize the ACLA model.
        
        Args:
            input_dim: Number of input features per timestep
            output_dim: Number of output predictions
            hidden_dim: Hidden dimension for LSTM and ODE
            augment_dim: Number of augmented dimensions for ANODE
            cnn_filters: List of CNN filter sizes [first, second]
            solver: ODE solver ("dopri5", "euler", "rk4", etc.)
            rtol: Relative tolerance for solver
            atol: Absolute tolerance for solver
            use_adjoint: Use adjoint method for memory-efficient gradients.
                         Default False (direct backprop) gives better accuracy and is faster
                         for short sequences. Set True only for very long sequences or
                         memory-constrained scenarios.
        """
        super().__init__(input_dim, output_dim)
        
        if not HAS_TORCHDIFFEQ:
            raise ImportError("torchdiffeq required. Install with: pip install torchdiffeq")
        
        if cnn_filters is None:
            cnn_filters = [64, 32]
        
        self.hidden_dim = hidden_dim
        self.augment_dim = augment_dim
        self.solver = solver
        self.rtol = rtol
        self.atol = atol
        self.use_adjoint = use_adjoint
        
        # 1. Attention layer
        self.attention = AttentionLayer(input_dim, hidden_dim=64, num_heads=4)
        
        # 2. CNN-LSTM encoder
        self.encoder = CNNLSTMEncoder(
            input_dim=input_dim,
            cnn_filters=cnn_filters,
            lstm_hidden=hidden_dim
        )
        
        # 3. Linear layer to initialize ODE state
        self.fc_ode_init = nn.Linear(hidden_dim, hidden_dim)
        
        # 4. ODE function (with augmented dimensions)
        self.ode_func = ODEFunc(hidden_dim, augment_dim)
        
        # 5. Output layer
        self.fc_out = nn.Linear(hidden_dim + augment_dim, output_dim)
        
        self._trajectory: Optional[torch.Tensor] = None
        self._last_attn_weights: Optional[torch.Tensor] = None
    
    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        """Forward pass with attention, CNN-LSTM, and ODE integration.
        
        Args:
            x: Input sequence of shape (batch, seq_len, input_dim)
            t: Time points of shape (batch, seq_len) or (seq_len,)
        
        Returns:
            Predictions of shape (batch, seq_len, output_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # 1. Apply attention
        x_attended, attn_weights = self.attention(x)
        self._last_attn_weights = attn_weights.detach()
        
        # 2. CNN-LSTM encoding
        encoded = self.encoder(x_attended)  # (batch, seq_len, hidden_dim)
        
        # 3. Use first timestep's encoding as ODE initial condition
        # This is more efficient: single ODE integration instead of seq_len calls
        z0 = self.fc_ode_init(encoded[:, 0, :])  # (batch, hidden_dim)
        
        # Augment with zeros for ANODE
        augment = torch.zeros(batch_size, self.augment_dim, device=x.device)
        y0 = torch.cat([z0, augment], dim=1)  # (batch, hidden_dim + augment_dim)
        
        # Get integration times
        if t is None:
            t_span = torch.linspace(0, 1, seq_len, device=x.device)
        else:
            if t.dim() == 2:
                t_span = t[0]  # Assume same times across batch
            else:
                t_span = t
            t_span = t_span - t_span[0]  # Normalize to start at 0
        
        # 4. Single ODE integration for entire trajectory (much more efficient)
        odeint_fn = odeint_adjoint if self.use_adjoint else odeint
        
        trajectory = odeint_fn(
            self.ode_func,
            y0,
            t_span,
            method=self.solver,
            rtol=self.rtol,
            atol=self.atol,
        )  # (seq_len, batch, hidden_dim + augment_dim)
        
        self._trajectory = trajectory.detach()
        
        # 5. Decode all timesteps
        traj = trajectory.permute(1, 0, 2)  # (batch, seq_len, hidden_dim + augment_dim)
        
        # Flatten, decode, reshape
        traj_flat = traj.reshape(-1, self.hidden_dim + self.augment_dim)
        y_flat = self.fc_out(traj_flat)
        output = y_flat.reshape(batch_size, seq_len, -1)
        
        return output
    
    def explain(self, x: torch.Tensor, t: Optional[torch.Tensor] = None,
                **kwargs) -> Dict[str, Any]:
        """Return interpretability information.
        
        Args:
            x: Input sequence
            t: Time points
        
        Returns:
            Dictionary with attention weights and trajectory
        """
        _ = self.forward(x, t)
        return {
            'attention_weights': self._last_attn_weights.cpu().numpy() if self._last_attn_weights is not None else None,
            'trajectory': self._trajectory.cpu().numpy() if self._trajectory is not None else None,
            'hidden_dim': self.hidden_dim,
            'augment_dim': self.augment_dim,
        }
