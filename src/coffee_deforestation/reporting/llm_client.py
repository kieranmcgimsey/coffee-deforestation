"""LLM client with dry-run mode and real Anthropic API integration.

What: Provides a unified interface to the Anthropic API with dry-run mode
that returns template-based reports, and a real LLM mode that calls Claude.
Why: Dry-run saves API costs during development; real calls produce richer reports.
Assumes: ANTHROPIC_API_KEY set in .env when dry_run=False.
Produces: Report markdown string.
"""

from __future__ import annotations

from pathlib import Path

from jinja2 import Template
from loguru import logger

from coffee_deforestation.config import PROJECT_ROOT, load_settings
from coffee_deforestation.stats.schema import AOISummary


def get_anthropic_client():
    """Create an Anthropic client using the configured API key.

    Raises ValueError if no API key is configured.
    """
    import anthropic

    settings = load_settings()
    if not settings.anthropic_api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY not set. Set it in .env or environment variables."
        )
    return anthropic.Anthropic(api_key=settings.anthropic_api_key)


def call_llm(
    system: str,
    messages: list[dict],
    model: str | None = None,
    max_tokens: int = 4096,
    tools: list[dict] | None = None,
) -> dict:
    """Make a single LLM call to Claude and return the response.

    Args:
        system: System prompt.
        messages: List of message dicts with role and content.
        model: Model name (defaults to config setting).
        max_tokens: Maximum output tokens.
        tools: Optional tool definitions for tool-use mode.

    Returns dict with:
        content: str (text response),
        tool_calls: list[dict] (if tools were used),
        input_tokens: int,
        output_tokens: int,
        stop_reason: str.
    """
    client = get_anthropic_client()
    settings = load_settings()
    model = model or settings.anthropic_model

    kwargs: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": messages,
    }
    if tools:
        kwargs["tools"] = tools

    logger.info(f"Calling Claude ({model}, max_tokens={max_tokens})")
    response = client.messages.create(**kwargs)

    # Extract text content and tool calls
    text_parts = []
    tool_calls = []
    for block in response.content:
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "tool_use":
            tool_calls.append({
                "id": block.id,
                "name": block.name,
                "input": block.input,
            })

    result = {
        "content": "\n".join(text_parts),
        "tool_calls": tool_calls,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "stop_reason": response.stop_reason,
    }

    logger.info(
        f"Claude response: {result['input_tokens']} in, "
        f"{result['output_tokens']} out, stop={result['stop_reason']}"
    )
    return result

# Template for dry-run mode reports
DRY_RUN_TEMPLATE = """\
# {{ metadata.name }} — Coffee-Linked Deforestation Report

## Executive summary

This report covers the {{ metadata.name }} region in {{ metadata.country }}, \
a {{ metadata.coffee_type }} growing area. Analysis of satellite imagery \
from {{ data_coverage.years_processed | first }} to {{ data_coverage.years_processed | last }} \
identified {{ change_detection.total_hotspots }} hotspot areas \
totalling {{ "%.1f" | format(change_detection.total_area_ha) }} hectares \
of potential coffee-linked deforestation.

## Area context

{{ metadata.name }} ({{ metadata.country }}) is characterized as: {{ metadata.role }}. \
The study area spans {{ "%.2f" | format(metadata.bbox.west) }}°E to \
{{ "%.2f" | format(metadata.bbox.east) }}°E longitude, \
{{ "%.2f" | format(metadata.bbox.south) }}°N to \
{{ "%.2f" | format(metadata.bbox.north) }}°N latitude.

## Headline findings

- **{{ change_detection.total_hotspots }}** coffee-deforestation hotspots detected (rule-based method)
- **{{ "%.1f" | format(change_detection.total_area_ha) }} ha** total area affected
- Largest hotspot: {{ "%.1f" | format(change_detection.largest_hotspot_ha) }} ha
- Validation: {{ "%.1f" | format(validation.coffee_fraction * 100) }}% coffee coverage, \
{{ "%.1f" | format(validation.forest_2000_fraction * 100) }}% forest-2000 coverage

## Hotspot deep-dives

{% for h in top_hotspots[:5] %}
### Hotspot {{ h.hotspot_id }} — Rank #{{ h.rank }}

Located at {{ "%.4f" | format(h.centroid_lat) }}°N, {{ "%.4f" | format(h.centroid_lon) }}°E. \
Area: {{ "%.1f" | format(h.area_ha) }} hectares.

{% endfor %}

## Historical context

Analysis uses Hansen Global Forest Change data (2000 baseline). \
Forest-2000 coverage in this AOI was {{ "%.1f" | format(validation.forest_2000_fraction * 100) }}%.

## Model performance and caveats

This report uses a **rule-based baseline** (Hansen loss ∩ FDP coffee probability > 0.5). \
ML-based classification will be available in Stage 2. Known limitations include \
potential commission errors where non-coffee crops replaced forest in areas adjacent \
to existing coffee plantations.

## Methodology

Rule-based detection: pixels with Hansen forest loss AND Forest Data Partnership \
coffee probability > 0.5 in the most recent layer. Years processed: \
{{ data_coverage.years_processed | join(", ") }}.
"""


def generate_report(
    summary: AOISummary,
    dry_run: bool | None = None,
) -> str:
    """Generate a markdown report for an AOI via the researcher → writer pipeline.

    In dry_run mode (default during development), runs the researcher and writer
    agents deterministically using real statistics — no API key required.
    When dry_run=False, the agents make real Anthropic API calls.
    """
    if dry_run is None:
        settings = load_settings()
        dry_run = settings.llm_dry_run

    from coffee_deforestation.reporting.agents import run_researcher, run_writer

    logger.info(f"Generating {'dry-run' if dry_run else 'LLM'} report for {summary.metadata.aoi_id}")

    researcher_result = run_researcher(summary, dry_run=dry_run)
    writer_result = run_writer(summary, researcher_result, dry_run=dry_run)
    return writer_result["report_markdown"]


def save_report(report: str, aoi_id: str, output_dir: Path | None = None) -> Path:
    """Save a report to markdown file."""
    if output_dir is None:
        output_dir = PROJECT_ROOT / "outputs" / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"report_{aoi_id}.md"
    with open(output_path, "w") as f:
        f.write(report)

    logger.info(f"Saved report to {output_path}")
    return output_path
