# Tracking API Reference

The `tracking` module implements experiment management. The `DualTracker` allows for simultaneous logging to local JSON/TensorBoard files and a remote or local MLflow server, ensuring both real-time visualization and long-term experiment comparison.

::: src.tracking.base
    options:
      show_root_heading: true
      show_source: true

::: src.tracking.local
    options:
      show_root_heading: true
      show_source: true

::: src.tracking.mlflow_tracker
    options:
      show_root_heading: true
      show_source: true

::: src.tracking.dual_tracker
    options:
      show_root_heading: true
      show_source: true
