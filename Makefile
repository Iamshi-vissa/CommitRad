.PHONY: setup test lint format demo

setup:
	uv sync

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check --fix .

demo:
	uv run commitrad mine repos.txt
	uv run commitrad label repos.txt
