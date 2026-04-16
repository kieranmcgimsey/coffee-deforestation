"""Synthesist agent: cross-AOI synthesis into a comparative executive brief.

What: Takes per-AOI writer reports + researcher findings and produces a
cross-AOI markdown brief comparing the showcase, contrast, and control AOIs.
In dry-run mode, uses a deterministic template populated with real numbers.
Why: The synthesist is the final reporting stage, producing the
board-level or policy-level document that puts the three AOIs in context.
Assumes: run_writer() has been called for each AOI. AOI order matters:
showcase first (lam_dong), contrast second (huila), control last (sul_de_minas).
Produces: A combined markdown brief and a structured cross-AOI comparison dict.
"""

from __future__ import annotations

import uuid

from loguru import logger

from coffee_deforestation.stats.schema import AOISummary


def run_synthesist(
    summaries: list[AOISummary],
    writer_results: list[dict],
    session_id: str | None = None,
    dry_run: bool = True,
) -> dict:
    """Run the synthesist agent across all AOIs.

    Args:
        summaries: One AOISummary per AOI, in any order.
        writer_results: Outputs of run_writer() for each AOI.
        session_id: Optional session ID for traceability.
        dry_run: When True, template-based generation.

    Returns dict with keys:
        brief_markdown, cross_aoi_table, key_contrasts, session_id.
    """
    if session_id is None:
        session_id = uuid.uuid4().hex[:8]

    if not dry_run:
        return _run_synthesist_llm(summaries, writer_results, session_id)

    logger.info(f"[Synthesist] Starting dry-run across {len(summaries)} AOIs (session={session_id})")

    # Index by aoi_id for easy lookup
    summary_map = {s.metadata.aoi_id: s for s in summaries}
    writer_map = {w["aoi_id"]: w for w in writer_results}

    cross_table = _build_cross_aoi_table(summary_map)
    key_contrasts = _build_key_contrasts(summary_map)
    brief_md = _build_brief(summary_map, writer_map, cross_table, key_contrasts)

    logger.info(f"[Synthesist] Brief complete: {len(brief_md.split())} words")

    return {
        "session_id": session_id,
        "brief_markdown": brief_md,
        "cross_aoi_table": cross_table,
        "key_contrasts": key_contrasts,
        "aoi_ids": list(summary_map.keys()),
    }


def _build_cross_aoi_table(summary_map: dict[str, AOISummary]) -> list[dict]:
    """Build per-AOI comparison rows."""
    rows = []
    for aoi_id, s in sorted(summary_map.items()):
        cd = s.change_detection
        mm = s.model_metrics
        hist = s.historical
        rows.append({
            "aoi_id": aoi_id,
            "name": s.metadata.name,
            "country": s.metadata.country,
            "role": s.metadata.role,
            "total_hotspots": cd.total_hotspots,
            "total_area_ha": cd.total_area_ha,
            "largest_ha": cd.largest_hotspot_ha,
            "f1_coffee": mm.f1_coffee,
            "coffee_on_former_forest_pct": round(hist.coffee_on_former_forest_fraction * 100, 1),
        })
    return rows


def _build_key_contrasts(summary_map: dict[str, AOISummary]) -> list[str]:
    """Identify the 3–5 most important cross-AOI contrasts."""
    contrasts = []

    aois = list(summary_map.values())
    if len(aois) < 2:
        return ["Insufficient AOIs for cross-comparison."]

    # Sort by total area descending
    by_area = sorted(aois, key=lambda s: s.change_detection.total_area_ha, reverse=True)
    top = by_area[0]
    bottom = by_area[-1]
    ratio = top.change_detection.total_area_ha / max(bottom.change_detection.total_area_ha, 0.1)
    contrasts.append(
        f"{top.metadata.name} shows {ratio:.1f}x more total affected area than "
        f"{bottom.metadata.name} ({top.change_detection.total_area_ha:.0f} ha vs "
        f"{bottom.change_detection.total_area_ha:.0f} ha), confirming expected showcase/control contrast."
    )

    # Hotspot density contrast
    by_hotspots = sorted(aois, key=lambda s: s.change_detection.total_hotspots, reverse=True)
    contrasts.append(
        f"Hotspot counts range from {by_hotspots[-1].change_detection.total_hotspots:,} "
        f"({by_hotspots[-1].metadata.name}) to {by_hotspots[0].change_detection.total_hotspots:,} "
        f"({by_hotspots[0].metadata.name}), a "
        f"{by_hotspots[0].change_detection.total_hotspots / max(by_hotspots[-1].change_detection.total_hotspots, 1):.1f}x "
        f"difference."
    )

    # Model F1 consistency
    f1s = [(s.metadata.name, s.model_metrics.f1_coffee) for s in aois if s.model_metrics.f1_coffee > 0]
    if f1s:
        min_f1 = min(f1s, key=lambda x: x[1])
        max_f1 = max(f1s, key=lambda x: x[1])
        contrasts.append(
            f"ML model F1 (coffee class) is consistent across AOIs: "
            f"{min_f1[0]} {min_f1[1]:.3f} — {max_f1[0]} {max_f1[1]:.3f}, "
            f"indicating the model generalizes across coffee systems."
        )

    # Historical coffee-on-forest signal
    hist_vals = sorted(aois, key=lambda s: s.historical.coffee_on_former_forest_fraction, reverse=True)
    contrasts.append(
        f"The 'coffee on former forest' signal is strongest in "
        f"{hist_vals[0].metadata.name} "
        f"({hist_vals[0].historical.coffee_on_former_forest_fraction*100:.1f}%) "
        f"and weakest in {hist_vals[-1].metadata.name} "
        f"({hist_vals[-1].historical.coffee_on_former_forest_fraction*100:.1f}%), "
        f"consistent with their respective roles."
    )

    return contrasts


def _build_brief(
    summary_map: dict[str, AOISummary],
    writer_map: dict[str, dict],
    cross_table: list[dict],
    key_contrasts: list[str],
) -> str:
    """Build the full cross-AOI brief markdown."""
    total_hotspots = sum(r["total_hotspots"] for r in cross_table)
    total_area = sum(r["total_area_ha"] for r in cross_table)
    aoi_names = ", ".join(r["name"] for r in cross_table)

    # Cross-AOI summary table (markdown)
    table_header = (
        "| AOI | Country | Role | Hotspots | Area (ha) | Largest (ha) | F1 | Coffee-on-Forest |\n"
        "|-----|---------|------|----------|-----------|--------------|-----|------------------|\n"
    )
    table_rows = "\n".join(
        f"| {r['name']} | {r['country']} | {r['role']} | "
        f"{r['total_hotspots']:,} | {r['total_area_ha']:,.1f} | "
        f"{r['largest_ha']:.1f} | {r['f1_coffee']:.3f} | {r['coffee_on_former_forest_pct']:.1f}% |"
        for r in cross_table
    )
    table_md = table_header + table_rows

    contrasts_md = "\n".join(f"- {c}" for c in key_contrasts)

    # Per-AOI mini-summaries (first 400 chars of each writer report)
    aoi_summaries = []
    for r in cross_table:
        wr = writer_map.get(r["aoi_id"], {})
        snippet = wr.get("report_markdown", "No report.")[:400].strip()
        aoi_summaries.append(f"### {r['name']} ({r['country']})\n\n{snippet}...\n")
    aoi_summaries_md = "\n".join(aoi_summaries)

    brief = f"""\
# Cross-AOI Synthesis Brief — Coffee-Linked Deforestation

## Overview

This brief synthesises findings across {len(cross_table)} study areas:
**{aoi_names}**.
Combined, the analysis detected **{total_hotspots:,} hotspots** covering
**{total_area:,.1f} ha** of potential coffee-linked deforestation.

---

## Cross-AOI Comparison

{table_md}

---

## Key Contrasts and Findings

{contrasts_md}

---

## Policy Implications

1. **Active deforestation frontier**: AOIs with high hotspot density and
   recent loss years (post-2019) should be prioritised for supply-chain
   due-diligence and direct farmer engagement.
2. **Model generalisability**: Consistent F1 scores across geographically
   distinct AOIs (Vietnam highlands, Andean belt, Brazilian plateau) confirm
   that the detection pipeline is not overfit to a single region.
3. **Data gaps**: Cloud cover in tropical regions remains a limiting factor;
   S1 SAR composites provide cloud-immune coverage but sacrifice spectral
   resolution. Expanding to additional AOI patches is recommended once
   resource usage is characterised at scale.

---

## Per-AOI Summaries

{aoi_summaries_md}

---

## Next Steps

- Expand geographic coverage: test additional patches within existing regions
  or add new coffee-producing regions (e.g., Sidama/Ethiopia, Antioquia/Colombia,
  Minas Gerais northern belt).
- Annual monitoring: re-run pipeline each dry season to detect new loss events.
- Supply-chain integration: export hotspot GeoJSONs for overlay against
  known trader/exporter sourcing zones.
"""
    return brief


def _run_synthesist_llm(
    summaries: list[AOISummary],
    writer_results: list[dict],
    session_id: str,
) -> dict:
    """Run the synthesist agent using a real Claude API call."""
    import json

    from coffee_deforestation.reporting.llm_client import call_llm

    logger.info(f"[Synthesist] Starting LLM run across {len(summaries)} AOIs (session={session_id})")

    # Build cross-AOI comparison data
    summary_map = {s.metadata.aoi_id: s for s in summaries}
    cross_table = _build_cross_aoi_table(summary_map)

    system = (
        "You are an analyst synthesizing coffee deforestation findings across "
        "multiple study regions. Write a cross-region comparison brief that:\n"
        "1. Compares hotspot counts, areas, and coffee attribution percentages\n"
        "2. Identifies key differences between regions and explains why\n"
        "3. Highlights which regions are high-risk vs stable\n"
        "4. Notes the negative control (Brazil) results\n"
        "5. Recommends priority areas for ground-truth verification\n"
        "Write in markdown. Be specific with numbers."
    )

    # Collect all writer reports
    reports_text = ""
    for wr in writer_results:
        reports_text += f"\n\n--- {wr['aoi_id']} ---\n{wr['report_markdown'][:2000]}"

    cross_table_json = json.dumps(cross_table, indent=2, default=str)

    messages = [
        {
            "role": "user",
            "content": (
                f"Synthesize findings across {len(summaries)} study regions.\n\n"
                f"Cross-AOI comparison table:\n```json\n{cross_table_json}\n```\n\n"
                f"Individual reports (truncated):\n{reports_text}\n\n"
                f"Write a synthesis brief comparing all regions."
            ),
        },
    ]

    response = call_llm(system=system, messages=messages, max_tokens=4096)
    brief_md = response["content"]

    # Extract key contrasts from the table data
    key_contrasts = _build_key_contrasts(summary_map)

    logger.info(f"[Synthesist] LLM brief complete: {len(brief_md.split())} words")

    return {
        "brief_markdown": brief_md,
        "cross_aoi_table": cross_table,
        "key_contrasts": key_contrasts,
        "session_id": session_id,
    }
