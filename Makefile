# Developer entry points. Requires `uv` (https://docs.astral.sh/uv/). Everything is
# installed into ./.venv - nothing is installed outside the project.

.PHONY: sync lint format typecheck test check demo build clean help

sync:            ## Create/refresh ./.venv with dev + framework dependencies
	uv sync --group dev --group frameworks

lint:            ## Ruff lint + format check
	uv run ruff check .
	uv run ruff format --check .

format:          ## Ruff format (rewrites files)
	uv run ruff format .
	uv run ruff check --fix .

typecheck:       ## mypy
	uv run mypy

test:            ## pytest
	uv run pytest

check: lint typecheck test ## Everything CI runs

demo:            ## Run the baseline / chaos / healed demo
	uv run python examples/demo_pipeline.py

build:           ## Build wheel + sdist into dist/
	uv build

clean:
	rm -rf dist build .pytest_cache .mypy_cache .ruff_cache .agent-chaos chaos-report.json
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'
