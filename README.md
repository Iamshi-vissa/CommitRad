# CommitRad

A Python 3.11 project built with `uv`.

## Quick Start

```bash
# Install dependencies
make setup

# Run tests
make test

# Lint code
make lint

# Format code
make fmt
```

## Project Structure

```
CommitRad/
├── src/
│   └── commitrad/          # Main package
├── tests/                  # Test suite
├── pyproject.toml          # Project configuration
├── Makefile                # Build commands
└── README.md               # This file
```

## Requirements

- Python 3.11+
- uv (package manager)

## Development

This project uses:
- **uv** for dependency management
- **pytest** for testing
- **ruff** for linting and formatting