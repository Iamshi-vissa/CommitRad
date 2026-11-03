.PHONY: setup test lint format demo precache serve dashboard docker

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
	uv run commitrad features repos.txt
	uv run commitrad train
	uv run commitrad strain

precache:
	uv run commitrad precache https://github.com/pallets/flask https://github.com/psf/requests https://github.com/pallets/click

serve:
	uv run commitrad serve

dashboard:
	uv run commitrad dashboard

docker:
	docker build -t commitrad .
