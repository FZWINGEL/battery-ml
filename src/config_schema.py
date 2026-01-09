"""Pydantic configuration schemas for validation."""

from pydantic import BaseModel, Field, field_validator
from typing import Literal, List, Optional, Dict, Tuple


class DataConfig(BaseModel):
    """Configuration for data loading."""

    experiment_id: int = Field(ge=1, le=5, description="Experiment ID (1-5)")
    base_path: str = Field(description="Base path to raw data")
    cells: List[str] = Field(description="List of cell IDs to load")
    temp_map: Dict[int, List[str]] = Field(
        description="Mapping from temperature (°C) to cell IDs"
    )


class PipelineConfig(BaseModel):
    """Configuration for preprocessing pipelines."""

    name: Literal["summary_set", "summary_cycle", "ica_peaks", "latent_ode_seq"] = (
        Field(description="Pipeline name")
    )

    # ICA-specific params
    sg_window: int = Field(
        default=51, ge=5, le=101, description="Savitzky-Golay window"
    )
    sg_order: int = Field(
        default=3, ge=1, le=5, description="Savitzky-Golay polynomial order"
    )
    num_peaks: int = Field(
        default=3, ge=1, le=10, description="Number of peaks to extract"
    )
    voltage_range: Tuple[float, float] = Field(
        default=(3.0, 4.2), description="Voltage range for ICA"
    )

    # Sequence params
    seq_length: Optional[int] = Field(
        default=None, ge=1, description="Max sequence length"
    )
    time_unit: Literal["days", "throughput_Ah"] = Field(
        default="days", description="Time unit for ODE models"
    )

    # General params
    normalize: bool = Field(default=True, description="Whether to normalize features")
    use_cache: bool = Field(default=True, description="Whether to use caching")
    include_arrhenius: bool = Field(
        default=True, description="Include Arrhenius features"
    )
    arrhenius_Ea: float = Field(
        default=50000.0, gt=0, description="Activation energy (J/mol)"
    )

    @field_validator("sg_window")
    @classmethod
    def sg_window_odd(cls, v: int) -> int:
        if v % 2 == 0:
            raise ValueError("sg_window must be odd")
        return v


class ModelConfig(BaseModel):
    """Configuration for models."""

    name: Literal["lgbm", "mlp", "lstm_attn", "neural_ode", "mamba"] = Field(
        description="Model name"
    )

    # Common params
    hidden_dim: int = Field(default=64, gt=0, description="Hidden dimension")
    hidden_dims: List[int] = Field(
        default=[64, 32], description="Hidden dimensions for MLP"
    )
    dropout: float = Field(default=0.1, ge=0, le=0.9, description="Dropout rate")

    # LSTM params
    num_layers: int = Field(default=2, ge=1, description="Number of LSTM layers")
    num_heads: int = Field(default=4, ge=1, description="Number of attention heads")

    # ODE-specific
    latent_dim: int = Field(default=32, gt=0, description="Latent dimension for ODE")
    solver: str = Field(default="dopri5", description="ODE solver")
    rtol: float = Field(default=1e-4, gt=0, description="Relative tolerance")
    atol: float = Field(default=1e-5, gt=0, description="Absolute tolerance")
    use_adjoint: bool = Field(
        default=True, description="Use adjoint method for gradients"
    )

    # LGBM params
    n_estimators: int = Field(default=1000, ge=1, description="Number of trees")
    learning_rate: float = Field(default=0.05, gt=0, lt=1, description="Learning rate")
    max_depth: int = Field(default=6, ge=1, description="Max tree depth")
    num_leaves: int = Field(default=31, ge=2, description="Number of leaves")
    reg_alpha: float = Field(default=0.1, ge=0, description="L1 regularization")
    reg_lambda: float = Field(default=0.1, ge=0, description="L2 regularization")
    early_stopping_rounds: int = Field(
        default=50, ge=1, description="Early stopping rounds"
    )

    @field_validator("solver")
    @classmethod
    def valid_solver(cls, v: str) -> str:
        valid = ["dopri5", "euler", "rk4", "adaptive_heun", "bosh3", "fehlberg2"]
        if v not in valid:
            raise ValueError(f"solver must be one of {valid}")
        return v


class SplitConfig(BaseModel):
    """Configuration for data splits."""

    strategy: Literal["temperature_holdout", "loco", "random"] = Field(
        description="Split strategy"
    )
    train_temps: List[int] = Field(
        default=[10, 40], description="Training temperatures"
    )
    val_temps: List[int] = Field(default=[25], description="Validation temperatures")
    test_temps: List[int] = Field(default=[25], description="Test temperatures")
    test_cell: Optional[str] = Field(default=None, description="Test cell for LOCO")
    random_seed: int = Field(default=42, description="Random seed for random splits")
    val_fraction: float = Field(
        default=0.2, ge=0, le=1, description="Validation fraction"
    )


class TrainingConfig(BaseModel):
    """Configuration for training."""

    epochs: int = Field(default=200, ge=1, description="Number of epochs")
    batch_size: int = Field(default=32, ge=1, description="Batch size")
    learning_rate: float = Field(default=1e-3, gt=0, lt=1, description="Learning rate")
    weight_decay: float = Field(default=0.01, ge=0, description="Weight decay")
    early_stopping_patience: int = Field(
        default=20, ge=1, description="Early stopping patience"
    )
    gradient_clip: float = Field(
        default=1.0, gt=0, description="Gradient clipping value"
    )
    use_amp: bool = Field(default=True, description="Use automatic mixed precision")
    scheduler_T0: int = Field(
        default=50, ge=1, description="Scheduler T0 for cosine annealing"
    )


class LossConfig(BaseModel):
    """Configuration for loss functions.

    Supports all registered loss types: mse, physics_informed, huber, mape, mae.
    Pass loss-specific parameters directly in this config.
    """

    name: Literal["mse", "physics_informed", "huber", "mape", "mae"] = Field(
        default="mse", description="Loss function name"
    )

    # Common params
    reduction: Literal["mean", "sum", "none"] = Field(
        default="mean", description="Reduction method"
    )

    # Physics-informed params
    monotonicity_weight: float = Field(
        default=0.0,
        ge=0,
        description="Monotonicity regularization weight (physics_informed)",
    )
    smoothness_weight: float = Field(
        default=0.0,
        ge=0,
        description="Smoothness regularization weight (physics_informed)",
    )

    # Huber params
    delta: float = Field(default=1.0, gt=0, description="Huber delta threshold (huber)")

    # MAPE params
    epsilon: float = Field(
        default=1e-8, gt=0, description="Small value to prevent division by zero (mape)"
    )


class TrackingConfig(BaseModel):
    """Configuration for experiment tracking."""

    backend: Literal["local", "mlflow", "wandb", "dual"] = Field(
        default="dual", description="Tracking backend"
    )

    # Local settings
    base_dir: str = Field(default="artifacts/runs", description="Local runs directory")
    use_tensorboard: bool = Field(
        default=True, description="Enable TensorBoard logging"
    )

    # MLflow settings
    mlflow_uri: Optional[str] = Field(
        default="file:./artifacts/mlruns", description="MLflow tracking URI"
    )
    experiment_name: str = Field(
        default="battery_degradation", description="MLflow experiment name"
    )

    # WandB settings
    wandb_project: Optional[str] = Field(default=None, description="W&B project name")


class ExperimentConfig(BaseModel):
    """Top-level experiment configuration."""

    name: str = Field(default="battery_degradation", description="Experiment name")
    seed: int = Field(default=42, description="Random seed")

    data: DataConfig
    pipeline: PipelineConfig
    model: ModelConfig
    split: SplitConfig = Field(default_factory=SplitConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    loss: LossConfig = Field(default_factory=LossConfig)
