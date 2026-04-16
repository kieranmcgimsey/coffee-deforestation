"""CLI entry point for Stage 3 report generation.

Usage:
    # Dry-run (default, no API key needed):
    uv run scripts/generate_reports.py

    # Specific AOIs only:
    uv run scripts/generate_reports.py --aoi lam_dong --aoi huila

    # With real LLM calls (requires ANTHROPIC_API_KEY in .env):
    uv run scripts/generate_reports.py --no-dry-run

    # Skip synthesis brief:
    uv run scripts/generate_reports.py --no-synthesis
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer
from loguru import logger

# Ensure src/ is on the path when run directly
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from coffee_deforestation.config import PROJECT_ROOT
from coffee_deforestation.logging_setup import setup_logging as configure_logging
from coffee_deforestation.reporting.agents import run_researcher, run_synthesist, run_writer
from coffee_deforestation.reporting.factcheck import append_factcheck_section, factcheck
from coffee_deforestation.reporting.llm_client import save_report
from coffee_deforestation.stats.schema import AOISummary

app = typer.Typer(help="Generate markdown reports from pipeline outputs.")

_ALL_AOIS = ["lam_dong", "huila", "sul_de_minas"]
_STATS_DIR = PROJECT_ROOT / "outputs" / "stats"
_REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"


def _load_summary(aoi_id: str) -> AOISummary:
    """Load and validate an AOISummary from disk."""
    path = _STATS_DIR / f"summary_{aoi_id}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Stats summary not found: {path}\n"
            f"Run the pipeline first: uv run scripts/run_all.py"
        )
    with open(path) as f:
        data = json.load(f)
    return AOISummary.model_validate(data)


@app.command()
def main(
    aoi: list[str] = typer.Option(
        default=_ALL_AOIS,
        help="AOI IDs to generate reports for. Can be repeated.",
    ),
    dry_run: bool = typer.Option(
        default=True,
        help="Use dry-run mode (no API key needed). Set --no-dry-run for real LLM calls.",
    ),
    synthesis: bool = typer.Option(
        default=True,
        help="Generate cross-AOI synthesis brief.",
    ),
    factcheck_reports: bool = typer.Option(
        default=True,
        help="Run factuality check and append warnings for unmatched numbers.",
    ),
    output_dir: Path = typer.Option(
        default=_REPORTS_DIR,
        help="Output directory for reports.",
    ),
) -> None:
    """Generate per-AOI reports and (optionally) a cross-AOI synthesis brief."""
    configure_logging()
    output_dir.mkdir(parents=True, exist_ok=True)

    aoi_ids = list(aoi)
    logger.info(f"Generating reports for: {aoi_ids} (dry_run={dry_run})")

    summaries: list[AOISummary] = []
    writer_results: list[dict] = []
    generated_paths: list[Path] = []

    for aoi_id in aoi_ids:
        logger.info(f"--- {aoi_id} ---")

        try:
            summary = _load_summary(aoi_id)
        except FileNotFoundError as e:
            logger.error(str(e))
            raise typer.Exit(code=1)

        # Researcher → writer pipeline
        researcher_result = run_researcher(summary, dry_run=dry_run)
        writer_result = run_writer(summary, researcher_result, dry_run=dry_run)
        report_md = writer_result["report_markdown"]

        # Optional factuality cross-check
        if factcheck_reports:
            fc = factcheck(report_md, summary)
            logger.info(
                f"Factcheck {aoi_id}: {fc.matched}/{fc.total_numbers} matched "
                f"({'PASS' if fc.passed else 'FAIL'})"
            )
            if not fc.passed:
                report_md = append_factcheck_section(report_md, fc)

        # Save per-AOI report
        out_path = save_report(report_md, aoi_id, output_dir)
        generated_paths.append(out_path)
        logger.success(f"Saved report: {out_path}")

        summaries.append(summary)
        writer_results.append(writer_result)

        # Print researcher tool call log
        logger.info(
            f"Researcher used {researcher_result['tool_calls_used']} tool calls, "
            f"found {len(researcher_result['findings'])} findings."
        )

    # Cross-AOI synthesis brief
    if synthesis and len(summaries) >= 2:
        logger.info("--- Synthesis brief ---")
        synth_result = run_synthesist(summaries, writer_results, dry_run=dry_run)
        brief_md = synth_result["brief_markdown"]

        brief_path = output_dir / "synthesis_brief.md"
        with open(brief_path, "w") as f:
            f.write(brief_md)
        logger.success(f"Saved synthesis brief: {brief_path}")
        generated_paths.append(brief_path)

        # Print key contrasts
        for c in synth_result["key_contrasts"]:
            logger.info(f"  Contrast: {c}")

    typer.echo("\nReports generated:")
    for p in generated_paths:
        typer.echo(f"  {p.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    app()
