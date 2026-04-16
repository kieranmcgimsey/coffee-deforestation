"""Tool: compare_periods — delta computation across years.

What: Returns the change in a named metric between two years for an AOI or hotspot.
Why: The researcher agent needs to characterise temporal trends (acceleration or
deceleration of deforestation, NDVI recovery, etc.).
Assumes: Stats summary JSON is available with per-year fields from Stage 4.
Produces: A dict with the delta, pct_change, and interpretation.
"""

from __future__ import annotations

import json

from loguru import logger

from coffee_deforestation.config import PROJECT_ROOT

_VALID_METRICS = {
    "coffee_area_ha", "forest_area_ha", "ndvi_mean",
    "loss_cumulative_ha", "hotspot_count",
}


def compare_periods(
    year_a: int,
    year_b: int,
    metric: str,
    aoi_id: str,
    scope: str = "aoi",
) -> dict:
    """Compare a metric between two years.

    Args:
        year_a: baseline year
        year_b: comparison year (must be > year_a)
        metric: one of coffee_area_ha, forest_area_ha, ndvi_mean,
                loss_cumulative_ha, hotspot_count
        aoi_id: AOI identifier
        scope: 'aoi' for AOI-wide, or a hotspot_id for hotspot-level

    Returns dict with: value_a, value_b, delta, pct_change, direction, interpretation
    """
    if metric not in _VALID_METRICS:
        return {"error": f"Unknown metric {metric!r}. Valid: {sorted(_VALID_METRICS)}"}

    if year_a >= year_b:
        return {"error": "year_a must be less than year_b"}

    # Load stats summary
    summary_path = PROJECT_ROOT / "outputs" / "stats" / f"summary_{aoi_id}.json"
    if not summary_path.exists():
        return {"error": f"Stats not found for {aoi_id}"}

    with open(summary_path) as f:
        data = json.load(f)

    # Extract real per-year data from summary JSON
    historical = data.get("historical", {})
    change_det = data.get("change_detection", {})

    ndvi_by_year = historical.get("ndvi_by_year", {})
    area_by_year = change_det.get("area_ha_by_loss_year", {})
    hotspots_by_year = change_det.get("hotspots_by_loss_year", {})
    total_ha = change_det.get("total_area_ha", 0)

    # Build series from real data
    series: dict[str, dict[int, float]] = {}

    if ndvi_by_year:
        series["ndvi_mean"] = {int(k): v for k, v in ndvi_by_year.items()}

    if area_by_year:
        # Cumulative loss area by year
        sorted_years = sorted(
            ((int(k), v) for k, v in area_by_year.items()), key=lambda x: x[0]
        )
        cumulative: dict[int, float] = {}
        running = 0.0
        for yr, area in sorted_years:
            running += area
            cumulative[yr] = round(running, 1)
        series["loss_cumulative_ha"] = cumulative
        series["coffee_area_ha"] = {int(k): v for k, v in area_by_year.items()}

    if hotspots_by_year:
        # Cumulative hotspot count by year
        sorted_hby = sorted(
            ((int(k), v) for k, v in hotspots_by_year.items()), key=lambda x: x[0]
        )
        cum_count: dict[int, float] = {}
        running_count = 0
        for yr, count in sorted_hby:
            running_count += count
            cum_count[yr] = float(running_count)
        series["hotspot_count"] = cum_count

    # Forest area: derive from NDVI if available (proxy)
    if ndvi_by_year:
        forest_frac = data.get("validation", {}).get("forest_2000_fraction", 0.5)
        bbox = data.get("metadata", {}).get("bbox", {})
        # Rough area estimate
        approx_area = total_ha / max(forest_frac, 0.01) if forest_frac > 0 else 10000
        series["forest_area_ha"] = {
            int(k): round(approx_area * forest_frac * (v / max(ndvi_by_year.values())), 1)
            for k, v in ndvi_by_year.items()
        }

    if metric not in series:
        return {
            "error": (
                f"No per-year data available for metric {metric!r} in {aoi_id}. "
                "Per-year statistics may not have been computed — run the pipeline "
                "with multi-temporal analysis enabled."
            )
        }

    metric_series = series[metric]
    val_a = metric_series.get(year_a)
    val_b = metric_series.get(year_b)

    if val_a is None or val_b is None:
        available = sorted(metric_series.keys())
        return {"error": f"No data for year {year_a} or {year_b}. Available: {available}"}

    delta = round(val_b - val_a, 3)
    pct_change = round(delta / max(abs(val_a), 1e-9) * 100, 1)

    direction = "increase" if delta > 0 else ("decrease" if delta < 0 else "no change")

    # Interpretation
    interpretations = {
        "coffee_area_ha": f"Coffee-linked clearing {'expanded' if delta > 0 else 'contracted'} by {abs(delta):.1f} ha between {year_a} and {year_b}.",
        "forest_area_ha": f"Forest cover {'decreased' if delta < 0 else 'increased'} by {abs(delta):.1f} ha.",
        "ndvi_mean": f"Mean NDVI {'declined' if delta < 0 else 'improved'} by {abs(delta):.3f}, indicating {'vegetation loss' if delta < 0 else 'recovery'}.",
        "loss_cumulative_ha": f"Cumulative loss reached {val_b:.1f} ha by {year_b}, up {abs(delta):.1f} ha from {year_a}.",
        "hotspot_count": f"Hotspot count {'grew' if delta > 0 else 'fell'} from {int(val_a)} to {int(val_b)} between {year_a} and {year_b}.",
    }

    return {
        "aoi_id": aoi_id,
        "scope": scope,
        "metric": metric,
        "year_a": year_a,
        "year_b": year_b,
        "value_a": val_a,
        "value_b": val_b,
        "delta": delta,
        "pct_change": pct_change,
        "direction": direction,
        "interpretation": interpretations[metric],
    }
