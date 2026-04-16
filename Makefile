.PHONY: install test lint format typecheck validate run run-all maps reports clean-cache

install:
	uv sync --all-extras

test:
	uv run pytest

lint:
	uv run ruff check src/ tests/ scripts/

format:
	uv run ruff format src/ tests/ scripts/

typecheck:
	uv run mypy src/

validate:
	uv run python scripts/validate_aois.py

run:
	uv run python scripts/run_aoi.py --aoi lam_dong

run-all:
	uv run python scripts/run_all.py

maps:
	uv run python scripts/generate_maps.py

reports:
	uv run python scripts/generate_report.py

clean-cache:
	uv run python scripts/clear_cache.py --all
