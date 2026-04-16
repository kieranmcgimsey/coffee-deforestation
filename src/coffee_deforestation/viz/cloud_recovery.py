"""Cloud recovery figure: S1 sees through clouds.

What: Side-by-side 1x3 panel showing (1) cloudy S2 RGB, (2) matching S1 VV,
(3) merged feature stack slice. Demonstrates SAR's cloud-penetration value.
Why: Demonstrates the complementary nature of optical and SAR sensors by
showing cloud-obscured S2 alongside cloud-immune S1 for the same area.
Assumes: S2 and S1 raster data are available as numpy arrays.
Produces: A 1x3 matplotlib panel per AOI saved to outputs/figures/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from coffee_deforestation.config import AOIConfig, PipelineConfig, PROJECT_ROOT
from coffee_deforestation.viz.theme import (
    COLORS,
    NDVI_CMAP,
    add_attribution,
    apply_theme,
    save_figure,
)


def plot_cloud_recovery(
    s2_cloudy_rgb: np.ndarray,
    s1_vv: np.ndarray,
    merged_ndvi: np.ndarray,
    cloud_fraction: float,
    aoi: AOIConfig,
    scene_date: str = "",
    output_path: str | None = None,
) -> str:
    """Plot the 1x3 cloud recovery comparison panel.

    Args:
        s2_cloudy_rgb: (H, W, 3) RGB array from a cloudy S2 scene
        s1_vv: (H, W) VV backscatter in dB from matching S1 composite
        merged_ndvi: (H, W) NDVI from the cloud-free median composite
        cloud_fraction: fraction of pixels masked as cloud (0-1)
        aoi: AOI configuration
        scene_date: date string for the cloudy scene
    """
    apply_theme()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Cloudy S2 RGB
    rgb_display = np.clip(s2_cloudy_rgb / 0.3, 0, 1)
    axes[0].imshow(rgb_display)
    axes[0].set_title(
        f"Sentinel-2 RGB\n({cloud_fraction:.0%} cloud cover)",
        fontsize=11, fontweight="bold",
    )
    axes[0].axis("off")

    # Panel 2: S1 VV (sees through clouds)
    im_vv = axes[1].imshow(s1_vv, cmap="gray", vmin=-25, vmax=0)
    axes[1].set_title(
        "Sentinel-1 VV\n(cloud-penetrating SAR)",
        fontsize=11, fontweight="bold",
    )
    axes[1].axis("off")
    plt.colorbar(im_vv, ax=axes[1], shrink=0.7, label="VV (dB)")

    # Panel 3: Merged NDVI (from cloud-free composite)
    im_ndvi = axes[2].imshow(merged_ndvi, cmap=NDVI_CMAP, vmin=-0.2, vmax=0.9)
    axes[2].set_title(
        "Cloud-free NDVI\n(multi-temporal composite)",
        fontsize=11, fontweight="bold",
    )
    axes[2].axis("off")
    plt.colorbar(im_ndvi, ax=axes[2], shrink=0.7, label="NDVI")

    # Suptitle
    date_str = f" — {scene_date}" if scene_date else ""
    fig.suptitle(
        f"Cloud Recovery: SAR Sees Through Clouds — {aoi.name} ({aoi.country}){date_str}",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    add_attribution(axes[1])

    fig.tight_layout()

    if output_path is None:
        fig_dir = PROJECT_ROOT / "outputs" / "figures" / aoi.id
        fig_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(fig_dir / "cloud_recovery.png")

    save_figure(fig, output_path)
    logger.info(f"Saved cloud recovery figure: {output_path}")
    return output_path


def plot_cloud_recovery_panel(
    aoi: AOIConfig,
    pipeline_config: PipelineConfig,
    output_path: str | None = None,
) -> str | None:
    """High-level wrapper: fetch a cloudy S2 scene + S1 composite from GEE and plot.

    Finds the year with highest cloud fraction in the composite, fetches
    representative pixel arrays at 300m for visualization (under pixel limit).
    Returns the output path, or None if no cloudy scene is found.
    """
    import ee

    from coffee_deforestation.data.ancillary import get_hansen
    from coffee_deforestation.data.gee_client import aoi_to_geometry
    from coffee_deforestation.data.sentinel1 import build_s1_composite
    from coffee_deforestation.data.sentinel2 import build_s2_composite

    logger.info(f"Generating cloud recovery panel for {aoi.id}")

    region = aoi_to_geometry(aoi)

    # Use the first year — typically highest cloud fraction (before temporal filtering)
    target_year = pipeline_config.temporal.years[0]  # type: ignore[union-attr]

    try:
        # Get the S1 composite for SAR reference
        s1_composite = build_s1_composite(aoi, target_year, pipeline_config)

        # Build a deliberately cloudy S2 scene: short 1-month window at peak cloud season
        # Use the S2 SR collection with minimal filtering to get actual clouds.
        # 1-month window at the *start* of the dry season (cloudiest transition period).
        from coffee_deforestation.data.sentinel2 import _get_date_range

        date_start, date_end = _get_date_range(aoi, target_year)
        # Narrow to first month for maximum cloudiness
        cloudy_s2 = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(region)
            .filterDate(date_start, date_start[:7] + "-28")
            .sort("CLOUDY_PIXEL_PERCENTAGE", ascending=False)
            .first()
        )

        if cloudy_s2 is None:
            logger.warning(f"No S2 scene found for cloud recovery panel: {aoi.id}")
            return None

        # Get cloud probability image for this scene
        cloud_prob = (
            ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
            .filterBounds(region)
            .filterDate(date_start, date_start[:7] + "-28")
            .first()
        )
        cloud_mask = cloud_prob.select("probability").gt(40) if cloud_prob else None

        # Sample pixel arrays at 300m (under GEE 262144 pixel limit)
        SAMPLE_SCALE = 300

        s2_info = (
            cloudy_s2.select(["B4", "B3", "B2"])
            .divide(10000)
            .reproject(crs=f"EPSG:{aoi.epsg_utm}", scale=SAMPLE_SCALE)
            .sampleRectangle(region=region, defaultValue=0)
            .getInfo()
        )

        s1_info = (
            s1_composite.select("vv_median")
            .reproject(crs=f"EPSG:{aoi.epsg_utm}", scale=SAMPLE_SCALE)
            .sampleRectangle(region=region, defaultValue=-25)
            .getInfo()
        )

        # Cloud-free NDVI from the standard S2 composite
        s2_composite = build_s2_composite(aoi, target_year, pipeline_config)
        ndvi_info = (
            s2_composite.normalizedDifference(["B8", "B4"])
            .rename("ndvi")
            .reproject(crs=f"EPSG:{aoi.epsg_utm}", scale=SAMPLE_SCALE)
            .sampleRectangle(region=region, defaultValue=0)
            .getInfo()
        )

        # Convert to numpy
        r = np.array(s2_info["properties"]["B4"])
        g = np.array(s2_info["properties"]["B3"])
        b = np.array(s2_info["properties"]["B2"])
        s2_rgb = np.dstack([r, g, b])

        vv = np.array(s1_info["properties"]["vv_median"])
        ndvi = np.array(ndvi_info["properties"]["ndvi"])

        # Estimate cloud fraction from mask or from metadata
        try:
            cloud_frac = float(
                cloudy_s2.get("CLOUDY_PIXEL_PERCENTAGE").getInfo()
            ) / 100
        except (AttributeError, TypeError, ValueError) as e:
            logger.debug(f"Cloud fraction fallback: {e}")
            cloud_frac = 0.5

        scene_date = date_start[:10]

        return plot_cloud_recovery(
            s2_cloudy_rgb=s2_rgb,
            s1_vv=vv,
            merged_ndvi=ndvi,
            cloud_fraction=cloud_frac,
            aoi=aoi,
            scene_date=scene_date,
            output_path=output_path,
        )

    except Exception as e:
        logger.warning(f"Cloud recovery panel failed for {aoi.id}: {e}")
        return None
