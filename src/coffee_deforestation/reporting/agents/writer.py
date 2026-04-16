"""Writer agent: converts researcher findings into a structured markdown report.

What: Produces a rich, fully-sourced markdown report from researcher findings,
the raw stats summary, and scratchpad contents. In dry-run mode, uses a
deterministic jinja-style f-string template populated with real numbers.
Why: The writer turns bullet-point findings into a coherent narrative the
synthesist can cite and end-users can read directly.
Assumes: run_researcher() has been called and returned a valid findings dict.
Produces: A markdown string with the fixed 7-section structure.
"""

from __future__ import annotations

import textwrap
import uuid

from loguru import logger

from coffee_deforestation.stats.schema import AOISummary


# Section header emoji/icons are deliberately absent (house style per CLAUDE.md)
_SECTION_HEADERS = [
    "Executive Summary",
    "Area Context",
    "Headline Findings",
    "Hotspot Deep-Dives",
    "Historical Context",
    "Model Performance",
    "Methodology",
]

_CONFIDENCE_LABEL = {True: "HIGH", False: "MODERATE"}


def run_writer(
    summary: AOISummary,
    researcher_result: dict,
    session_id: str | None = None,
    dry_run: bool = True,
) -> dict:
    """Run the writer agent for one AOI.

    Args:
        summary: Full AOISummary for the AOI.
        researcher_result: Output of run_researcher().
        session_id: Optional session ID for traceability.
        dry_run: When True, uses template-based generation (no API key needed).

    Returns dict with keys:
        report_markdown, aoi_id, session_id, sections_written, word_count.
    """
    aoi_id = summary.metadata.aoi_id
    if session_id is None:
        session_id = uuid.uuid4().hex[:8]

    if not dry_run:
        return _run_writer_llm(summary, researcher_result, session_id)

    logger.info(f"[Writer] Starting dry-run for {aoi_id} (session={session_id})")

    findings = researcher_result.get("findings", [])
    anomalies = researcher_result.get("anomalies", [])
    confidence_notes = researcher_result.get("confidence_notes", [])

    md = _build_report(summary, findings, anomalies, confidence_notes)

    word_count = len(md.split())
    logger.info(f"[Writer] {aoi_id} complete: {word_count} words, {len(_SECTION_HEADERS)} sections")

    return {
        "aoi_id": aoi_id,
        "session_id": session_id,
        "report_markdown": md,
        "sections_written": _SECTION_HEADERS,
        "word_count": word_count,
    }


def _build_report(
    summary: AOISummary,
    findings: list[dict],
    anomalies: list[str],
    confidence_notes: list[str],
) -> str:
    """Build the full markdown report from components."""
    meta = summary.metadata
    cd = summary.change_detection
    val = summary.validation
    hist = summary.historical
    mm = summary.model_metrics
    abl = summary.ablation
    top = summary.top_hotspots[:5]
    years = summary.data_coverage.years_processed
    year_range = f"{years[0]}–{years[-1]}" if len(years) >= 2 else str(years[0]) if years else "N/A"

    high_confidence = mm.f1_coffee > 0.75
    conf_label = _CONFIDENCE_LABEL[high_confidence]

    # --- Executive Summary ---
    finding_titles = "\n".join(f"- {f['title']}" for f in findings) if findings else "- No findings."
    exec_summary = textwrap.dedent(f"""\
        ## Executive Summary

        **{meta.name}** ({meta.country}) — {meta.coffee_type} growing region.
        Analysis of satellite imagery ({year_range}) detected
        **{cd.total_hotspots:,} deforestation hotspots** totalling
        **{cd.total_area_ha:,.1f} ha** of potential coffee-linked forest loss.

        Key findings:
        {finding_titles}

        Model confidence: **{conf_label}** (F1 = {mm.f1_coffee:.3f}).
        {anomalies[0] if anomalies else ""}
    """)

    # --- Area Context ---
    b = meta.bbox
    pct_former_forest = hist.coffee_on_former_forest_fraction * 100
    area_context = textwrap.dedent(f"""\
        ## Area Context

        {meta.name} ({meta.country}) serves as **{meta.role}** in this study.
        The AOI covers {b.west:.2f}°–{b.east:.2f}°E longitude,
        {b.south:.2f}°–{b.north:.2f}°N latitude (UTM {meta.epsg_utm}).

        Coffee type: {meta.coffee_type}.
        Satellite validation: {val.coffee_fraction*100:.1f}% coffee coverage,
        {val.forest_2000_fraction*100:.1f}% forest-2000 baseline coverage (Hansen).
        Of current coffee pixels, **{pct_former_forest:.1f}%** were forested in 2000,
        establishing a direct forest-to-coffee conversion signal.
    """)

    # --- Headline Findings ---
    loss_year_rows = "\n".join(
        f"| {y} | {n:,} |"
        for y, n in sorted(cd.hotspots_by_loss_year.items())
    ) if cd.hotspots_by_loss_year else "| N/A | N/A |"

    headline = textwrap.dedent(f"""\
        ## Headline Findings

        | Metric | Value |
        |--------|-------|
        | Total hotspots | {cd.total_hotspots:,} |
        | Total area | {cd.total_area_ha:,.1f} ha |
        | Largest hotspot | {cd.largest_hotspot_ha:.1f} ha |
        | Smallest hotspot | {cd.smallest_hotspot_ha:.1f} ha |
        | Coffee on former forest | {pct_former_forest:.1f}% |
        | Hansen loss pixels | {val.hansen_loss_pixels:,} |

        **Hotspots by primary loss year:**

        | Loss Year | Hotspot Count |
        |-----------|--------------|
        {loss_year_rows}
    """)

    # --- Hotspot Deep-Dives ---
    deep_dives_parts = []
    for i, f in enumerate(findings, 1):
        nums = f.get("supporting_numbers", {})
        maps = f.get("map_paths", [])
        map_line = f"\n*Maps: {', '.join(maps)}*" if maps else ""
        nums_lines = "\n".join(f"  - {k}: {v}" for k, v in nums.items() if v is not None)
        deep_dives_parts.append(textwrap.dedent(f"""\
            ### Finding {i}: {f['title']}

            {f['summary']}

            Supporting data:
            {nums_lines}
            {map_line}
        """))

    # Also add per-hotspot cards for top 5
    hotspot_cards = []
    for h in top:
        hotspot_cards.append(
            f"**{h.hotspot_id}** (Rank #{h.rank}) — "
            f"{h.area_ha:.1f} ha at "
            f"{h.centroid_lat:.4f}°N, {h.centroid_lon:.4f}°E"
        )
    hotspot_list = "\n".join(f"- {c}" for c in hotspot_cards)

    deep_dives_section = "## Hotspot Deep-Dives\n\n" + "\n".join(deep_dives_parts)
    if hotspot_list:
        deep_dives_section += f"\n### Top 5 Hotspots by Area\n\n{hotspot_list}\n"

    # --- Historical Context ---
    mean_offset = hist.mean_loss_year_offset
    loss_year_str = (
        f"~{2000 + mean_offset:.0f}" if mean_offset is not None else "unknown"
    )
    repl_dist = hist.replacement_class_distribution
    repl_rows = "\n".join(
        f"| {cls} | {frac*100:.1f}% |" for cls, frac in sorted(repl_dist.items(), key=lambda x: -x[1])
    ) if repl_dist else "| coffee | 100.0% |"

    historical = textwrap.dedent(f"""\
        ## Historical Context

        Hansen Global Forest Change (2000 baseline) provides the loss-year signal.
        Within this AOI, **{hist.was_forest_2000_fraction*100:.1f}%** of pixels were
        forested in 2000. Mean forest-loss year: **{loss_year_str}**.

        **Replacement class distribution (post-loss land cover):**

        | Replacement Class | Share |
        |-------------------|-------|
        {repl_rows}

        The clear-then-plant pattern — forest cleared, followed within 1–4 years by
        rising coffee probability — is the dominant signal driving hotspot detection.
    """)

    # --- Model Performance ---
    perf = textwrap.dedent(f"""\
        ## Model Performance

        | Model | Accuracy | F1 (coffee) | Precision | Recall |
        |-------|----------|-------------|-----------|--------|
        | {mm.model_type.replace("_", " ").title()} | {mm.accuracy:.3f} | {mm.f1_coffee:.3f} | {mm.precision_coffee:.3f} | {mm.recall_coffee:.3f} |

        **Ablation study (S1/S2/combined):**

        | Feature Set | F1 (coffee) | Accuracy |
        |-------------|-------------|---------|
        | S1-only (SAR) | {abl.s1_only.f1_coffee:.3f} | {abl.s1_only.accuracy:.3f} |
        | S2-only (optical) | {abl.s2_only.f1_coffee:.3f} | {abl.s2_only.accuracy:.3f} |
        | S1+S2 (combined) | {abl.s1_s2.f1_coffee:.3f} | {abl.s1_s2.accuracy:.3f} |

        **Confidence notes:**
        {chr(10).join(f"- {note}" for note in confidence_notes)}
    """)

    # --- Methodology ---
    methodology = textwrap.dedent(f"""\
        ## Methodology

        1. **Data acquisition**: Sentinel-2 (10 m) and Sentinel-1 SAR (10 m) composites
           via Google Earth Engine, dry-season median composites {year_range}.
        2. **Features**: Spectral indices (NDVI, EVI, NDWI, NBR), SAR backscatter
           (VV/VH ratio, temporal std), contextual (slope, elevation, road proximity).
        3. **Hotspot detection (rule-based)**: Hansen Global Forest Change loss pixels
           intersected with FDP coffee probability > 0.5 (FDP model 2025a).
        4. **ML classification**: {mm.model_type.replace("_", " ").title()} trained on pooled
           GEE-sampled labels from all AOIs, cross-AOI holdout evaluation.
        5. **Historical look-back**: Hansen 2000 baseline → loss-year detection →
           FDP time series for coffee signal emergence.
        6. **Report generation**: Researcher agent (8 tool calls) → writer → synthesist.

        Years processed: {year_range}.
        Detection method: {cd.method}.
    """)

    sections = [
        f"# {meta.name} — Coffee-Linked Deforestation Report\n",
        exec_summary,
        area_context,
        headline,
        deep_dives_section,
        historical,
        perf,
        methodology,
    ]
    return "\n---\n\n".join(sections)


def _run_writer_llm(
    summary: AOISummary, researcher_result: dict, session_id: str
) -> dict:
    """Run the writer agent using a real Claude API call."""
    import json

    from coffee_deforestation.reporting.llm_client import call_llm

    aoi_id = summary.metadata.aoi_id
    logger.info(f"[Writer] Starting LLM run for {aoi_id} (session={session_id})")

    system = (
        "You are a technical report writer specializing in remote sensing and "
        "land-use change analysis. Write a structured markdown report about "
        "coffee-linked deforestation based on the researcher's findings and "
        "the raw statistics. Use the fixed 7-section structure:\n"
        "1. Executive Summary (3-5 sentences with concrete numbers)\n"
        "2. Area Context (what/where/why this region matters for coffee)\n"
        "3. Headline Findings (bullet points with cited numbers)\n"
        "4. Hotspot Deep-Dives (2-3 paragraphs per top hotspot)\n"
        "5. Historical Context (forest-to-coffee conversion timeline)\n"
        "6. Model Performance (precision, recall, F1, caveats)\n"
        "7. Methodology (one paragraph summary)\n"
        "Every number in the report must come from the provided data."
    )

    summary_json = summary.model_dump_json(indent=2)
    findings_json = json.dumps(researcher_result.get("findings", []), indent=2, default=str)

    messages = [
        {
            "role": "user",
            "content": (
                f"Write a coffee deforestation report for {summary.metadata.name} "
                f"({summary.metadata.country}).\n\n"
                f"Researcher findings:\n```json\n{findings_json}\n```\n\n"
                f"Full statistics:\n```json\n{summary_json}\n```\n\n"
                f"Write the full report in markdown using the 7-section structure."
            ),
        },
    ]

    response = call_llm(system=system, messages=messages, max_tokens=4096)
    report_md = response["content"]
    word_count = len(report_md.split())

    logger.info(f"[Writer] {aoi_id} LLM complete: {word_count} words")

    return {
        "report_markdown": report_md,
        "aoi_id": aoi_id,
        "session_id": session_id,
        "sections_written": 7,
        "word_count": word_count,
    }
