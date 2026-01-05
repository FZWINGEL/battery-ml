# Building Documentation Locally

This directory contains the MkDocs documentation for BatteryML.

## Prerequisites

Install documentation dependencies:

```bash
pip install -r docs/requirements.txt
```

## Building Locally

### Serve Locally (Development)

```bash
mkdocs serve
```

Open your browser to `http://127.0.0.1:8000` to view the documentation.

### Build Static Site

```bash
mkdocs build
```

The static site will be generated in the `site/` directory.

## Documentation Structure

- `getting-started/` - Installation, quickstart, concepts
- `user-guide/` - Detailed usage documentation
- `api/` - API reference (auto-generated)
- `architecture/` - System design and patterns
- `examples/` - Extended examples
- `contributing/` - Contribution guidelines
- `troubleshooting/` - Common issues and solutions
- `theory/` - Background theory
- `reference/` - Glossary and citations

## Updating Documentation

1. Edit markdown files in `docs/`
2. Run `mkdocs serve` to preview changes
3. Commit and push changes
4. Documentation will be automatically deployed via GitHub Actions

## Configuration

Documentation configuration is in `mkdocs.yml` at the project root.

## Troubleshooting

### Import Errors

If you see import errors when building:

```bash
# Ensure src is in PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Missing Dependencies

```bash
pip install -r docs/requirements.txt
```
