"""Run the full pipeline for a single AOI.

Usage: uv run python scripts/run_aoi.py --aoi lam_dong [--force] [--resilient]
"""

from __future__ import annotations

import typer

app = typer.Typer()


@app.command()
def main(
    aoi: str = typer.Option(..., help="AOI identifier (e.g., lam_dong, huila, sul_de_minas)"),
    force: bool = typer.Option(False, help="Bypass cache and recompute everything"),
    resilient: bool = typer.Option(False, help="Continue on non-fatal errors"),
) -> None:
    """Run the complete pipeline for one AOI."""
    from coffee_deforestation.cache import set_force
    from coffee_deforestation.pipeline import run_aoi

    if force:
        set_force(True)

    outputs = run_aoi(aoi, resilient=resilient)
    print(f"\nPipeline complete for {aoi}. Outputs:")
    for key, path in outputs.items():
        print(f"  {key}: {path}")


if __name__ == "__main__":
    app()
