"""Recency module: pull the latest cloud-filtered S2 scene per AOI.

What: Finds the most recent Sentinel-2 scene with <20% cloud cover for each AOI,
saves a thumbnail PNG, and records the scene date in the stats summary.
Why: Gives the report an "As of [date]" anchor — the reader can see how current
the analysis is and whether the situation has changed since the study period.
Assumes: GEE is initialized. AOI config is valid.
Produces: A thumbnail PNG and the scene acquisition date string.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from coffee_deforestation.config import AOIConfig, PROJECT_ROOT


def get_latest_scene_date(
    aoi: AOIConfig,
    max_cloud_pct: float = 20.0,
    lookback_days: int = 90,
) -> tuple[str, float] | None:
    """Find the most recent low-cloud S2 scene for an AOI.

    Returns: (date_str, cloud_pct) or None if no scene found.
    """
    import ee
    from datetime import date, timedelta

    from coffee_deforestation.data.gee_client import aoi_to_geometry

    today = date.today()
    start = (today - timedelta(days=lookback_days)).isoformat()
    end = today.isoformat()

    region = aoi_to_geometry(aoi)

    col = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud_pct))
        .sort("system:time_start", False)  # most recent first
    )

    size = col.size().getInfo()
    if size == 0:
        logger.warning(f"No recent S2 scene (<{max_cloud_pct}% cloud) for {aoi.id}")
        return None

    latest = col.first()
    date_ms = latest.get("system:time_start").getInfo()
    cloud_pct = latest.get("CLOUDY_PIXEL_PERCENTAGE").getInfo()

    import datetime

    scene_date = datetime.datetime.fromtimestamp(date_ms / 1000).strftime("%Y-%m-%d")
    logger.info(f"Latest S2 scene for {aoi.id}: {scene_date} ({cloud_pct:.1f}% cloud)")
    return scene_date, float(cloud_pct)


def save_recency_thumbnail(
    aoi: AOIConfig,
    output_dir: Path | None = None,
    scale: int = 300,
) -> tuple[str, str] | None:
    """Fetch the latest S2 RGB thumbnail and save as PNG.

    Returns: (output_path, scene_date) or None on failure.
    """
    import ee
    import numpy as np
    import matplotlib.pyplot as plt
    from datetime import date, timedelta

    from coffee_deforestation.data.gee_client import aoi_to_geometry
    from coffee_deforestation.viz.theme import apply_theme, save_figure, add_attribution

    if output_dir is None:
        output_dir = PROJECT_ROOT / "outputs" / "figures" / "recency"
    output_dir.mkdir(parents=True, exist_ok=True)

    result = get_latest_scene_date(aoi)
    if result is None:
        return None
    scene_date, cloud_pct = result

    region = aoi_to_geometry(aoi)

    today = date.today()
    start = (today - timedelta(days=90)).isoformat()

    # Fetch the most recent scene
    latest = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filterDate(start, today.isoformat())
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20.0))
        .sort("system:time_start", False)
        .first()
    )

    try:
        info = (
            latest.select(["B4", "B3", "B2"])
            .divide(10000)
            .reproject(crs=f"EPSG:{aoi.epsg_utm}", scale=scale)
            .sampleRectangle(region=region, defaultValue=0)
            .getInfo()
        )

        r = np.array(info["properties"]["B4"])
        g = np.array(info["properties"]["B3"])
        b = np.array(info["properties"]["B2"])
        rgb = np.clip(np.dstack([r, g, b]) / 0.3, 0, 1)

    except Exception as e:
        logger.warning(f"Recency thumbnail fetch failed for {aoi.id}: {e}")
        return None

    apply_theme()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(rgb)
    ax.axis("off")
    ax.set_title(
        f"Latest Sentinel-2 Scene — {aoi.name}",
        fontsize=12, fontweight="bold", pad=10,
    )
    ax.text(
        0.5, -0.04,
        f"Date: {scene_date} | Cloud cover: {cloud_pct:.1f}%",
        transform=ax.transAxes, ha="center", fontsize=9, color="#666",
    )
    add_attribution(ax)
    fig.tight_layout()

    out_path = str(output_dir / f"{aoi.id}_latest.png")
    save_figure(fig, out_path)
    logger.info(f"Saved recency thumbnail: {out_path}")
    return out_path, scene_date


def get_recency_info(aoi: AOIConfig) -> dict:
    """Get recency info without saving a thumbnail (lightweight).

    Returns dict with scene_date, cloud_pct, days_ago.
    """
    from datetime import date

    result = get_latest_scene_date(aoi)
    if result is None:
        return {"scene_date": None, "cloud_pct": None, "days_ago": None}

    scene_date_str, cloud_pct = result
    scene_date = date.fromisoformat(scene_date_str)
    days_ago = (date.today() - scene_date).days

    return {
        "scene_date": scene_date_str,
        "cloud_pct": round(cloud_pct, 1),
        "days_ago": days_ago,
    }
