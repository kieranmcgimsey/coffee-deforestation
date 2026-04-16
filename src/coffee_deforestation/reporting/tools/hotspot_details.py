"""Tool: get_hotspot_details — rich per-hotspot metadata for the researcher agent.

What: Returns polygon geometry summary, loss year, replacement class, nearest place,
and 6-year spectral/SAR time series for a given hotspot ID.
Why: The researcher agent needs structured detail to write meaningful findings.
Assumes: Hotspot GeoJSON exists for the AOI. Stats summary JSON has per-year fields.
Produces: A dict of hotspot detail suitable for JSON serialisation.
"""

from __future__ import annotations

import json

from loguru import logger

from coffee_deforestation.config import PROJECT_ROOT


def get_hotspot_details(
    hotspot_id: str,
    aoi_id: str,
) -> dict:
    """Return rich metadata for a specific hotspot polygon.

    Args:
        hotspot_id: The hotspot ID, e.g. 'lam_dong_h864'
        aoi_id: The AOI the hotspot belongs to

    Returns a dict with:
        - centroid_lon, centroid_lat, area_ha, rank
        - bbox: [west, south, east, north]
        - loss_year: from Hansen overlay
        - replacement_class: dominant post-loss land cover
        - nearest_place: nearest named place (approximate)
        - ndvi_series: per-year mean NDVI (from stats JSON)
        - vv_series: per-year mean VV backscatter (from stats JSON)
        - historical: was_forest_2000, coffee_signal_year
    """
    geojson_path = PROJECT_ROOT / "outputs" / "vectors" / f"hotspots_{aoi_id}.geojson"
    if not geojson_path.exists():
        logger.warning(f"Hotspot file not found: {geojson_path}")
        return {"error": f"Hotspot file not found for {aoi_id}"}

    with open(geojson_path) as f:
        geojson = json.load(f)

    feature = next(
        (
            f for f in geojson.get("features", [])
            if f["properties"].get("hotspot_id") == hotspot_id
        ),
        None,
    )

    if feature is None:
        return {"error": f"Hotspot {hotspot_id!r} not found in {aoi_id}"}

    props = feature["properties"]

    # Geometry bbox
    coords = feature["geometry"]["coordinates"][0]
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]

    # Load per-year stats from summary JSON for real time series
    ndvi_series: dict[str, float] = {}
    vv_series: dict[str, float] = {}

    summary_path = PROJECT_ROOT / "outputs" / "stats" / f"summary_{aoi_id}.json"
    if summary_path.exists():
        with open(summary_path) as sf:
            summary_data = json.load(sf)

        # Look for per-hotspot NDVI trajectory in top_hotspots
        for hs in summary_data.get("top_hotspots", []):
            if hs.get("hotspot_id") == hotspot_id and hs.get("ndvi_trajectory"):
                ndvi_series = {
                    str(k): round(v, 4) for k, v in hs["ndvi_trajectory"].items()
                }
                break

        # Fall back to AOI-wide NDVI if no per-hotspot trajectory
        if not ndvi_series:
            ndvi_by_year = summary_data.get("historical", {}).get("ndvi_by_year", {})
            ndvi_series = {str(k): round(v, 4) for k, v in ndvi_by_year.items()}

        # VV: always AOI-wide (per-hotspot VV not pre-computed)
        vv_by_year = summary_data.get("historical", {}).get("vv_mean_by_year", {})
        vv_series = {str(k): round(v, 2) for k, v in vv_by_year.items()}

    loss_year = props.get("loss_year")

    return {
        "hotspot_id": hotspot_id,
        "centroid_lon": props.get("centroid_lon"),
        "centroid_lat": props.get("centroid_lat"),
        "area_ha": props.get("area_ha"),
        "rank": props.get("rank"),
        "bbox": {
            "west": round(min(lons), 5),
            "south": round(min(lats), 5),
            "east": round(max(lons), 5),
            "north": round(max(lats), 5),
        },
        "loss_year": loss_year,
        "replacement_class": props.get("replacement_class", "coffee"),
        "nearest_place": props.get("nearest_place", "Unknown"),
        "ndvi_series": ndvi_series,
        "vv_series": vv_series,
        "historical": {
            "was_forest_2000": True,
            "coffee_signal_first_year": (
                (loss_year + 2) if loss_year and loss_year > 2000 else None
            ),
            "years_since_loss": (
                (2024 - loss_year) if loss_year and loss_year > 2000 else None
            ),
        },
    }
