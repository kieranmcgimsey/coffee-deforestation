"""Researcher agent: investigates AOI data and writes findings to a scratchpad.

What: Simulates the researcher agent's tool-use loop, producing structured findings
from the stats summary and hotspot data. In dry-run mode, uses a deterministic
script of tool calls to produce consistent, real-data-grounded findings.
Why: The researcher identifies the 3–5 most interesting patterns for the writer.
Assumes: Stats summary and hotspot GeoJSON are available.
Produces: A findings dict and populated scratchpad.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from loguru import logger

from coffee_deforestation.config import PROJECT_ROOT
from coffee_deforestation.stats.schema import AOISummary


# Per-AOI contextual descriptions for enriching dry-run reports
_AOI_CONTEXT = {
    "lam_dong": {
        "region_desc": "the Central Highlands of Vietnam",
        "coffee_desc": "Robusta coffee grown at 800–1600m elevation under intensive cultivation",
        "deforestation_driver": "smallholder expansion converting montane forest to coffee farms",
        "anomaly": "High density of small (<5 ha) hotspots clustered along forest edge suggests incremental encroachment rather than large-scale clearing.",
    },
    "huila": {
        "region_desc": "the Andean coffee belt of southern Colombia",
        "coffee_desc": "shade-grown Arabica at 1400–2000m, often under forest canopy remnants",
        "deforestation_driver": "forest conversion for specialty coffee expansion in high-altitude zones",
        "anomaly": "Several large hotspots (>30 ha) align with known road expansion corridors, suggesting infrastructure-driven access.",
    },
    "sul_de_minas": {
        "region_desc": "the established Arabica belt of Minas Gerais, Brazil",
        "coffee_desc": "large-scale mechanised Arabica cultivation on rolling terrain",
        "deforestation_driver": "minimal new expansion; most losses are plantation renewal or riparian buffer clearance",
        "anomaly": "Negative control as expected: hotspot density and area are substantially lower than the Vietnam showcase.",
    },
}

_DEFAULT_CONTEXT = {
    "region_desc": "a coffee-producing region",
    "coffee_desc": "coffee cultivation",
    "deforestation_driver": "agricultural expansion",
    "anomaly": "No significant anomalies detected.",
}


def run_researcher(
    summary: AOISummary,
    session_id: str | None = None,
    dry_run: bool = True,
) -> dict:
    """Run the researcher agent for one AOI.

    In dry_run mode, executes a scripted sequence of tool calls (≤8) using
    real data from the stats summary and hotspot GeoJSON.

    Returns: findings dict with keys: findings, anomalies, confidence_notes,
             tool_call_log, scratchpad_contents.
    """
    aoi_id = summary.metadata.aoi_id
    if session_id is None:
        session_id = uuid.uuid4().hex[:8]

    if not dry_run:
        return _run_researcher_llm(summary, session_id)

    logger.info(f"[Researcher] Starting dry-run for {aoi_id} (session={session_id})")

    from coffee_deforestation.reporting.tools import (
        compare_periods,
        get_historical_context,
        get_hotspot_details,
        query_stats,
        render_hotspot_map,
        scratchpad_write,
    )

    ctx = _AOI_CONTEXT.get(aoi_id, _DEFAULT_CONTEXT)
    tool_log: list[dict] = []
    top_hotspots = summary.top_hotspots[:5]
    total_ha = summary.change_detection.total_area_ha
    n_hotspots = summary.change_detection.total_hotspots
    largest_ha = summary.change_detection.largest_hotspot_ha

    # ---- Tool call 1: query large hotspots ----
    large_hotspots = query_stats("area_ha > 10", aoi_id)
    tool_log.append({"tool": "query_stats", "args": {"filter": "area_ha > 10"}, "n_results": len(large_hotspots)})
    large_ha_count = len(large_hotspots)

    # ---- Tool call 2: get detail on the largest hotspot ----
    h1_id = top_hotspots[0].hotspot_id if top_hotspots else f"{aoi_id}_h001"
    h1_detail = get_hotspot_details(h1_id, aoi_id)
    tool_log.append({"tool": "get_hotspot_details", "args": {"hotspot_id": h1_id}})

    # ---- Tool call 3: historical context for largest hotspot ----
    h1_history = get_historical_context(h1_id, aoi_id)
    tool_log.append({"tool": "get_historical_context", "args": {"polygon_id": h1_id}})
    loss_year = h1_history.get("loss_year", 2018)
    coffee_year = h1_history.get("coffee_signal_first_year", 2020)

    # ---- Tool call 4: compare cumulative loss 2019 → 2023 ----
    trend = compare_periods(2019, 2023, "loss_cumulative_ha", aoi_id)
    tool_log.append({"tool": "compare_periods", "args": {"year_a": 2019, "year_b": 2023, "metric": "loss_cumulative_ha"}})

    # ---- Tool call 5: render top hotspot map ----
    map_path = render_hotspot_map(
        h1_id, aoi_id,
        layers=["hotspot_boundary", "coffee_prob", "hansen_loss"],
        style="static",
    )
    tool_log.append({"tool": "render_hotspot_map", "args": {"hotspot_id": h1_id, "layers": ["hotspot_boundary", "coffee_prob", "hansen_loss"]}})
    map_paths = [] if map_path.startswith("RENDER_FAILED") else [map_path]

    # ---- Tool call 6: query recent losses (2021+) ----
    recent = query_stats("loss_year >= 2021", aoi_id) if any(
        f.get("loss_year") is not None for f in large_hotspots
    ) else []
    tool_log.append({"tool": "query_stats", "args": {"filter": "loss_year >= 2021"}, "n_results": len(recent)})

    # ---- Tool call 7: write finding 1 to scratchpad ----
    finding1_text = (
        f"The largest hotspot ({h1_id}, {largest_ha:.1f} ha) at "
        f"({top_hotspots[0].centroid_lat:.4f}°N, {top_hotspots[0].centroid_lon:.4f}°E) "
        f"lost forest in {loss_year} with coffee signal appearing {coffee_year - loss_year} "
        f"year(s) later in {coffee_year}. This lag is consistent with a clear-then-plant pattern."
    ) if top_hotspots else "Largest hotspot analysis not available."

    scratchpad_write("finding_1_largest_hotspot", finding1_text, aoi_id, session_id)
    tool_log.append({"tool": "scratchpad_write", "args": {"key": "finding_1_largest_hotspot"}})

    # ---- Tool call 8: write finding 2 to scratchpad ----
    coffee_on_ff = summary.historical.coffee_on_former_forest_fraction
    finding2_text = (
        f"Historical look-back shows {coffee_on_ff:.1%} of current coffee pixels were "
        f"forested in 2000. Of {n_hotspots} detected hotspots, {large_ha_count} exceed "
        f"10 ha ({large_ha_count / max(n_hotspots, 1) * 100:.1f}% of total), "
        f"accounting for the majority of the {total_ha:.0f} ha total affected area. "
        f"Context: {ctx['anomaly']}"
    )
    scratchpad_write("finding_2_scale_context", finding2_text, aoi_id, session_id)
    tool_log.append({"tool": "scratchpad_write", "args": {"key": "finding_2_scale_context"}})

    # Build structured findings output
    findings = [
        {
            "title": f"Largest hotspot: {largest_ha:.1f}-ha clearing near {h1_id}",
            "summary": finding1_text,
            "supporting_numbers": {
                "area_ha": largest_ha,
                "loss_year": loss_year,
                "coffee_signal_year": coffee_year,
                "lag_years": coffee_year - loss_year,
            },
            "map_paths": map_paths,
        },
        {
            "title": f"Scale: {large_ha_count} hotspots >10 ha, {total_ha:.0f} ha total",
            "summary": finding2_text,
            "supporting_numbers": {
                "n_hotspots_over_10ha": large_ha_count,
                "total_hotspots": n_hotspots,
                "total_area_ha": total_ha,
                "coffee_on_former_forest_pct": round(coffee_on_ff * 100, 1),
            },
            "map_paths": [],
        },
        {
            "title": f"Temporal trend: {trend.get('pct_change', 0):.1f}% change 2019–2023",
            "summary": trend.get("interpretation", "Trend data unavailable."),
            "supporting_numbers": {
                "value_2019": trend.get("value_a"),
                "value_2023": trend.get("value_b"),
                "delta_ha": trend.get("delta"),
                "pct_change": trend.get("pct_change"),
            },
            "map_paths": [],
        },
    ]

    anomalies = [ctx["anomaly"]]

    confidence_notes = [
        f"Rule-based detection (Hansen ∩ FDP >0.5) may include false positives near mixed forest-crop boundaries.",
        f"ML F1 score: {summary.model_metrics.f1_coffee:.3f} — model confidence rated {'high' if summary.model_metrics.f1_coffee > 0.75 else 'moderate'}.",
        f"Historical look-back uses Hansen 30m data; small clearings <0.5 ha may be missed.",
    ]

    result = {
        "aoi_id": aoi_id,
        "session_id": session_id,
        "findings": findings,
        "anomalies": anomalies,
        "confidence_notes": confidence_notes,
        "tool_call_log": tool_log,
        "tool_calls_used": len(tool_log),
        "scratchpad_contents": {
            "finding_1_largest_hotspot": finding1_text,
            "finding_2_scale_context": finding2_text,
        },
    }

    logger.info(
        f"[Researcher] {aoi_id} complete: {len(findings)} findings, "
        f"{len(tool_log)} tool calls"
    )
    return result


def _run_researcher_llm(summary: AOISummary, session_id: str) -> dict:
    """Run the researcher agent using real Claude API calls with tool use."""
    from coffee_deforestation.reporting.llm_client import call_llm
    from coffee_deforestation.reporting.tools import (
        compare_periods,
        get_historical_context,
        get_hotspot_details,
        query_stats,
        render_hotspot_map,
        scratchpad_write,
    )

    aoi_id = summary.metadata.aoi_id
    logger.info(f"[Researcher] Starting LLM run for {aoi_id} (session={session_id})")

    # System prompt
    system = (
        "You are a remote sensing researcher analyzing satellite-based coffee "
        "deforestation data. You have access to tools to query hotspot statistics, "
        "get historical context, compare time periods, and render maps. "
        "Investigate the data and identify the 3-5 most interesting findings. "
        "Focus on: largest hotspots, temporal trends, spatial patterns, and anomalies. "
        "You have a maximum of 8 tool calls. Use them wisely."
    )

    # Tool definitions
    tools = [
        {
            "name": "query_stats",
            "description": "Query hotspot statistics with a filter expression. Returns up to 20 matching hotspots.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "filter_expr": {"type": "string", "description": "Filter like 'area_ha > 10'"},
                },
                "required": ["filter_expr"],
            },
        },
        {
            "name": "get_hotspot_details",
            "description": "Get detailed metadata for a specific hotspot including area, location, loss year, NDVI trajectory.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "hotspot_id": {"type": "string"},
                },
                "required": ["hotspot_id"],
            },
        },
        {
            "name": "get_historical_context",
            "description": "Get Hansen 2000-2024 forest trajectory for a hotspot: was_forest_2000, loss_year, recovery status.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "polygon_id": {"type": "string"},
                },
                "required": ["polygon_id"],
            },
        },
        {
            "name": "compare_periods",
            "description": "Compare a metric between two years. Metrics: ndvi_mean, loss_cumulative_ha, coffee_area_ha.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "year_a": {"type": "integer"},
                    "year_b": {"type": "integer"},
                    "metric": {"type": "string"},
                },
                "required": ["year_a", "year_b", "metric"],
            },
        },
    ]

    # Initial message with summary data
    summary_json = summary.model_dump_json(indent=2)
    messages = [
        {
            "role": "user",
            "content": (
                f"Analyze this coffee deforestation data for {summary.metadata.name} "
                f"({summary.metadata.country}).\n\n"
                f"Stats summary:\n```json\n{summary_json}\n```\n\n"
                f"Use your tools to investigate. Identify 3-5 key findings. "
                f"After investigating, write your findings as a JSON object with keys: "
                f"findings (list of {{title, summary, supporting_numbers}}), "
                f"anomalies (list of strings), confidence_notes (list of strings)."
            ),
        },
    ]

    # Tool-use loop (max 8 calls)
    tool_log: list[dict] = []
    tool_dispatch = {
        "query_stats": lambda args: query_stats(args["filter_expr"], aoi_id),
        "get_hotspot_details": lambda args: get_hotspot_details(args["hotspot_id"], aoi_id),
        "get_historical_context": lambda args: get_historical_context(args["polygon_id"], aoi_id),
        "compare_periods": lambda args: compare_periods(
            args["year_a"], args["year_b"], args["metric"], aoi_id
        ),
    }

    final_text = ""
    for turn in range(8):
        response = call_llm(system=system, messages=messages, tools=tools, max_tokens=4096)

        if response["tool_calls"]:
            # Process tool calls
            tool_results = []
            for tc in response["tool_calls"]:
                tool_name = tc["name"]
                tool_input = tc["input"]
                tool_log.append({"tool": tool_name, "args": tool_input})

                handler = tool_dispatch.get(tool_name)
                if handler:
                    try:
                        result = handler(tool_input)
                    except Exception as e:
                        result = {"error": str(e)}
                else:
                    result = {"error": f"Unknown tool: {tool_name}"}

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tc["id"],
                    "content": json.dumps(result, default=str)[:4000],
                })

            # Add assistant message and tool results
            messages.append({"role": "assistant", "content": response["tool_calls"]})
            messages.append({"role": "user", "content": tool_results})
        else:
            # No more tool calls — final response
            final_text = response["content"]
            break

    # Parse findings from the final text
    findings = []
    anomalies = []
    confidence_notes = []

    try:
        # Try to extract JSON from the response
        import re
        json_match = re.search(r"\{[\s\S]*\}", final_text)
        if json_match:
            parsed = json.loads(json_match.group())
            findings = parsed.get("findings", [])
            anomalies = parsed.get("anomalies", [])
            confidence_notes = parsed.get("confidence_notes", [])
    except (json.JSONDecodeError, TypeError):
        # If JSON parsing fails, create a single finding from the text
        findings = [{"title": "LLM Analysis", "summary": final_text, "supporting_numbers": {}}]

    logger.info(
        f"[Researcher] {aoi_id} LLM complete: {len(findings)} findings, "
        f"{len(tool_log)} tool calls"
    )

    return {
        "aoi_id": aoi_id,
        "session_id": session_id,
        "findings": findings,
        "anomalies": anomalies,
        "confidence_notes": confidence_notes,
        "tool_call_log": tool_log,
        "tool_calls_used": len(tool_log),
        "scratchpad_contents": {},
    }
