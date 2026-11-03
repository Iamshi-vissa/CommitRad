.PHONY: setup test lint help

# Default target
help:
	@echo "CommitRad - Available targets:"
	@echo "  setup   - Install dependencies using uv"
	@echo "  test    - Run tests with pytest"
	@echo "  lint    - Run ruff linter"
	@echo "  fmt     - Format code with ruff"

# Install dependencies
setup:
	uv sync --dev

# Run tests
test:
	uv run pytest -v

# Run linter
lint:
	uv run ruff check src tests

# Format code
fmt:
	uv run ruff format src tests

# Run all checks
check: lint test