# Configuration API Reference

This module defines the Pydantic schemas used for validating the Hydra configurations. These schemas ensure that all experiment parameters are typed and consistent across different runs.

::: src.config_schema
    options:
      show_root_heading: true
      show_source: true
      members:
        - ExperimentConfig
        - DataConfig
        - PipelineConfig
        - ModelConfig
        - TrainingConfig
        - LossConfig
        - TrackingConfig
        - AppConfig
