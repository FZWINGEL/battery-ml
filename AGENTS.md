# AGENTS.md

This file documents the minimum workflow for autonomous agents (and humans) to set up, validate, and modify this repository safely.

## Setup

### 1) Create and activate a virtual environment

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```powershell
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

## Common Commands

### Run tests

If your global Python environment has incompatible pytest plugins installed, `pytest` may crash during plugin auto-loading.

Recommended (isolated, avoids third-party plugin autoload):

```powershell
python scripts/run_pytest.py
```

Collect-only (fast sanity check):

```powershell
python scripts/run_pytest.py --collect-only -q
```

### Lint

```powershell
python -m ruff check src tests
```

### Format

```powershell
python -m black src tests
```

### Type check

```powershell
python -m mypy src
```

## Project Layout

- `src/`: library code (data loading, pipelines, models, training, tracking)
- `configs/`: Hydra configurations
- `examples/`: runnable experiment scripts
- `tests/`: pytest unit tests

## Conventions

- Prefer adding new code under `src/` and covering it with tests under `tests/`.
- Keep any generated artifacts out of git (see `.gitignore`).
