"""Clear cached artifacts.

Usage:
  uv run python scripts/clear_cache.py --all
  uv run python scripts/clear_cache.py --stage features
"""

from __future__ import annotations

import typer

app = typer.Typer()


@app.command()
def main(
    stage: str = typer.Option("", help="Clear a specific stage (e.g., features, gee_exports)"),
    all: bool = typer.Option(False, "--all", help="Clear entire cache"),
) -> None:
    """Clear cached pipeline artifacts."""
    from coffee_deforestation.cache import clear_cache

    if not stage and not all:
        print("Specify --stage STAGE or --all")
        raise typer.Exit(1)

    target = None if all else stage
    count = clear_cache(stage=target)
    print(f"Cleared {count} cached items" + (f" from stage '{stage}'" if stage else ""))


if __name__ == "__main__":
    app()
