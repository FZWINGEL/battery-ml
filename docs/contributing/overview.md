# Contributing Guide

Thank you for your interest in contributing to BatteryML! This guide will help you get started.

## How to Contribute

We welcome contributions in many forms:

- **Bug fixes**: Report and fix bugs
- **New features**: Add pipelines, models, or utilities
- **Documentation**: Improve documentation
- **Examples**: Add example scripts
- **Tests**: Add or improve tests

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub
# Then clone your fork
git clone https://github.com/yourusername/battery-ml.git
cd battery-ml
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
pip install -r docs/requirements.txt  # For documentation
pip install pytest pytest-cov  # For testing
```

### 4. Install in Development Mode

```bash
pip install -e .
```

## Code Style

### Python Style

- Follow PEP 8
- Use type hints
- Maximum line length: 100 characters
- Use Google-style docstrings

### Example

```python
def function_name(param1: int, param2: str) -> bool:
    """Brief description.
    
    Longer description if needed.
    
    Args:
        param1: Description of param1
        param2: Description of param2
    
    Returns:
        Description of return value
    
    Raises:
        ValueError: When something goes wrong
    """
    # Implementation
    pass
```

## Development Workflow

### 1. Create Feature Branch

```bash
git checkout -b feature/my-feature
```

### 2. Make Changes

- Write code
- Add tests
- Update documentation

### 3. Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test
pytest tests/test_my_feature.py -v
```

### 4. Check Code Quality

```bash
# Linting (if configured)
flake8 src/

# Type checking (if configured)
mypy src/
```

### 5. Commit Changes

```bash
git add .
git commit -m "Add feature: description"
```

Use clear, descriptive commit messages.

### 6. Push and Create Pull Request

```bash
git push origin feature/my-feature
```

Then create a pull request on GitHub.

## Pull Request Guidelines

### Before Submitting

- [ ] Code follows style guidelines
- [ ] Tests pass
- [ ] Documentation updated
- [ ] No merge conflicts
- [ ] Commit messages are clear

### PR Description

Include:

- What changes were made
- Why changes were made
- How to test
- Screenshots (if UI changes)

## Testing Guidelines

### Writing Tests

- Place tests in `tests/` directory
- Name test files `test_*.py`
- Use descriptive test names

```python
def test_pipeline_creates_samples():
    """Test that pipeline creates Sample objects."""
    pipeline = SummarySetPipeline()
    samples = pipeline.fit_transform({'df': df})
    assert len(samples) > 0
    assert isinstance(samples[0], Sample)
```

### Test Coverage

Aim for high test coverage, especially for:

- Core functionality
- Edge cases
- Error handling

## Documentation Guidelines

### Docstrings

All public functions/classes should have docstrings:

```python
class MyClass:
    """Brief description.
    
    Longer description if needed.
    
    Attributes:
        attr1: Description of attr1
        attr2: Description of attr2
    """
    pass
```

### Documentation Updates

When adding features:

- Update relevant user guide sections
- Add API reference (auto-generated from docstrings)
- Add examples if applicable

## Project Structure

```text
battery-ml/
├── src/              # Source code
│   ├── data/         # Data loading
│   ├── pipelines/    # Feature pipelines
│   ├── models/       # ML models
│   ├── training/     # Training utilities
│   ├── tracking/     # Experiment tracking
│   └── explainability/  # Interpretability
├── tests/            # Tests
├── examples/         # Example scripts
├── configs/          # Configuration files
└── docs/             # Documentation
```

## Next Steps

- [Adding Pipelines](adding-pipelines.md) - How to add pipelines
- [Adding Models](adding-models.md) - How to add models
- [Adding Losses](adding-losses.md) - How to add loss functions
- [Adding Splits](adding-splits.md) - How to add split strategies
- [Testing](testing.md) - Testing guidelines
- [Code Structure](code-structure.md) - Codebase organization

## Questions?

- Open an issue for bugs or feature requests
- Check existing issues before creating new ones
- Be respectful and constructive in discussions

Thank you for contributing!
