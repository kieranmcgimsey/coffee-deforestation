"""Generate interactive maps with GEE tile layers for all AOIs.

What: Creates rich folium maps with satellite imagery, NDVI, coffee probability,
Hansen loss, and hotspot polygons as zoomable tile layers at full resolution.
Why: Tile-served maps eliminate pixel budget limits and render at any zoom level.
Assumes: GEE is authenticated. Hotspot GeoJSONs exist in outputs/vectors/.
Produces: HTML map files in outputs/maps/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import typer
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

app = typer.Typer(help="Generate interactive GEE-tile maps.")

AOI_IDS = ["lam_dong", "huila", "sul_de_minas"]


@app.command()
def main(
    aoi: list[str] = typer.Option(default=AOI_IDS, help="AOI IDs to generate maps for."),
) -> None:
    """Generate rich interactive maps with GEE tile layers."""
    import ee

    from coffee_deforestation.config import PROJECT_ROOT, load_aois, load_pipeline_config
    from coffee_deforestation.data.ancillary import get_fdp_coffee, get_hansen
    from coffee_deforestation.data.gee_client import init_gee
    from coffee_deforestation.data.sentinel2 import build_s2_composite
    from coffee_deforestation.features.indices import compute_ndvi
    from coffee_deforestation.logging_setup import setup_logging
    from coffee_deforestation.viz.interactive import create_rich_map, save_map

    setup_logging()
    init_gee()

    aois_config = load_aois()
    pipeline_config = load_pipeline_config()

    for aoi_id in aoi:
        aoi_cfg = aois_config.get(aoi_id)
        if not aoi_cfg:
            logger.error(f"Unknown AOI: {aoi_id}")
            continue

        logger.info(f"Generating map for {aoi_id}...")

        # Build GEE image objects (lazy — no data downloaded)
        latest_year = pipeline_config.temporal.years[-1]
        s2 = build_s2_composite(aoi_cfg, latest_year, pipeline_config)
        ndvi_img = compute_ndvi(s2)
        coffee = get_fdp_coffee(aoi_cfg)
        hansen = get_hansen(aoi_cfg)

        # Hotspot GeoJSON
        hotspot_path = PROJECT_ROOT / "outputs" / "vectors" / f"hotspots_{aoi_id}.geojson"
        if not hotspot_path.exists():
            hotspot_path = None
            logger.warning(f"No hotspot GeoJSON for {aoi_id}")

        # Create map with GEE tile layers
        m = create_rich_map(
            aoi_cfg,
            hotspot_geojson_path=hotspot_path,
            s2_composite=s2,
            ndvi=ndvi_img,
            coffee_prob=coffee.select("coffee_prob"),
            hansen_loss=hansen,
        )

        # Save
        map_path = save_map(m, aoi_cfg)
        typer.echo(f"  {map_path}")

    typer.echo("\nMaps generated. Open in browser to view.")


if __name__ == "__main__":
    app()
