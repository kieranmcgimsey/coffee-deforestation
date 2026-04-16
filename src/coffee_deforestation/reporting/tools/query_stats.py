"""Tool: query_stats — safe query over the hotspots table.

What: Evaluates a safe subset of filter expressions over the hotspot GeoJSON
feature table and returns matching records.
Why: The researcher agent needs to find hotspots by criteria (e.g. area > 10 ha,
loss_year >= 2022) without raw database access.
Assumes: Hotspot GeoJSON exists. Filter expressions are vetted for safety.
Produces: Up to 20 matching hotspot records as a list of dicts.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from loguru import logger

from coffee_deforestation.config import PROJECT_ROOT

# Allowed filter expression tokens — reject anything dangerous
_SAFE_TOKEN = re.compile(
    r"^[\w\s><=!.()\[\]0-9\"']+$"
)
_ALLOWED_FIELDS = {
    "area_ha", "rank", "loss_year", "centroid_lat", "centroid_lon",
    "replacement_class", "hotspot_id",
}


def _is_safe_expression(expr: str) -> bool:
    """Return True if the expression uses only allowed tokens."""
    if not _SAFE_TOKEN.match(expr):
        return False
    # Block any obvious injection patterns
    blocked = ["import", "exec", "eval", "__", "open", "os", "sys"]
    return not any(b in expr for b in blocked)


def query_stats(
    filter_expr: str,
    aoi_id: str,
    max_results: int = 20,
) -> list[dict]:
    """Query the hotspot table with a safe filter expression.

    Example filter expressions:
        "area_ha > 10"
        "loss_year >= 2022 and area_ha > 5"
        "rank <= 10"

    Returns list of matching hotspot property dicts.
    """
    if not _is_safe_expression(filter_expr):
        logger.warning(f"Rejected unsafe filter expression: {filter_expr!r}")
        return [{"error": "Unsafe filter expression rejected"}]

    geojson_path = PROJECT_ROOT / "outputs" / "vectors" / f"hotspots_{aoi_id}.geojson"
    if not geojson_path.exists():
        return [{"error": f"Hotspot file not found for {aoi_id}"}]

    with open(geojson_path) as f:
        geojson = json.load(f)

    features = geojson.get("features", [])
    results = []

    for feat in features:
        props = feat["properties"]
        # Evaluate filter in a restricted namespace
        try:
            ns = {k: props.get(k) for k in _ALLOWED_FIELDS}
            # Replace "and" / "or" with Python operators
            py_expr = filter_expr.replace(" and ", " and ").replace(" or ", " or ")
            if eval(py_expr, {"__builtins__": {}}, ns):  # noqa: S307
                results.append({
                    "hotspot_id": props.get("hotspot_id"),
                    "area_ha": props.get("area_ha"),
                    "rank": props.get("rank"),
                    "loss_year": props.get("loss_year"),
                    "centroid_lon": props.get("centroid_lon"),
                    "centroid_lat": props.get("centroid_lat"),
                    "replacement_class": props.get("replacement_class", "coffee"),
                })
        except (ValueError, KeyError, TypeError) as e:
            logger.debug(f"Skipping feature in query_stats: {e}")
            continue

    logger.info(
        f"query_stats({filter_expr!r}, {aoi_id}): {len(results)} results "
        f"(returning up to {max_results})"
    )
    return results[:max_results]
