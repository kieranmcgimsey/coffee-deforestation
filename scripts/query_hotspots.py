"""Query hotspots near a location or by time period.

What: CLI tool for filtering and exporting hotspot data by location, time,
or area — designed for field workers, policy researchers, and journalists.
Why: The raw GeoJSON has 5000+ features. Users need to extract specific
subsets without writing Python code.
Assumes: Hotspot GeoJSONs exist in outputs/vectors/.
Produces: Filtered CSV/GeoJSON to stdout or file.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import typer
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from coffee_deforestation.config import PROJECT_ROOT

app = typer.Typer(help="Query and export hotspot data.")

VECTORS_DIR = PROJECT_ROOT / "outputs" / "vectors"


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance in km between two lat/lon points."""
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


@app.command()
def near(
    lat: float = typer.Argument(..., help="Latitude of the point"),
    lon: float = typer.Argument(..., help="Longitude of the point"),
    km: float = typer.Option(20, help="Search radius in km"),
    aoi: str = typer.Option("lam_dong", help="AOI ID"),
    year_min: int = typer.Option(0, help="Minimum loss year (e.g., 2020)"),
    year_max: int = typer.Option(9999, help="Maximum loss year"),
    limit: int = typer.Option(100, help="Max results"),
    output: str = typer.Option("", help="Output file (csv/geojson). Empty = stdout CSV"),
) -> None:
    """Find hotspots within a radius of a GPS point.

    Example: uv run python scripts/query_hotspots.py near 11.9 108.3 --km 15 --year-min 2020
    """
    geojson_path = VECTORS_DIR / f"hotspots_{aoi}.geojson"
    if not geojson_path.exists():
        typer.echo(f"Error: {geojson_path} not found. Run the pipeline first.", err=True)
        raise typer.Exit(1)

    with open(geojson_path) as f:
        data = json.load(f)

    results = []
    for feat in data.get("features", []):
        props = feat["properties"]
        clat = props.get("centroid_lat", 0)
        clon = props.get("centroid_lon", 0)
        loss_year = props.get("loss_year")

        # Distance filter
        dist = _haversine_km(lat, lon, clat, clon)
        if dist > km:
            continue

        # Year filter
        if loss_year is not None and (loss_year < year_min or loss_year > year_max):
            continue

        results.append({
            "hotspot_id": props.get("hotspot_id", ""),
            "lat": clat,
            "lon": clon,
            "area_ha": props.get("area_ha", 0),
            "loss_year": loss_year,
            "rank": props.get("rank", 0),
            "distance_km": round(dist, 1),
        })

    # Sort by distance
    results.sort(key=lambda r: r["distance_km"])
    results = results[:limit]

    typer.echo(f"Found {len(results)} hotspots within {km} km of ({lat}, {lon})", err=True)
    if year_min > 0:
        typer.echo(f"  Filtered to loss year >= {year_min}", err=True)

    if output and output.endswith(".geojson"):
        # GeoJSON output
        out_features = []
        for r in results:
            matching = [
                f for f in data["features"]
                if f["properties"].get("hotspot_id") == r["hotspot_id"]
            ]
            if matching:
                out_features.append(matching[0])
        out_data = {"type": "FeatureCollection", "features": out_features}
        with open(output, "w") as f:
            json.dump(out_data, f, indent=2)
        typer.echo(f"Saved to {output}", err=True)
    else:
        # CSV output
        writer = csv.DictWriter(
            open(output, "w") if output else sys.stdout,
            fieldnames=["hotspot_id", "lat", "lon", "area_ha", "loss_year", "rank", "distance_km"],
        )
        writer.writeheader()
        writer.writerows(results)
        if output:
            typer.echo(f"Saved to {output}", err=True)


@app.command()
def summary(
    aoi: str = typer.Option("lam_dong", help="AOI ID"),
    year_min: int = typer.Option(0, help="Minimum loss year"),
    year_max: int = typer.Option(9999, help="Maximum loss year"),
) -> None:
    """Print summary statistics for hotspots in an AOI.

    Example: uv run python scripts/query_hotspots.py summary --aoi lam_dong --year-min 2020
    """
    geojson_path = VECTORS_DIR / f"hotspots_{aoi}.geojson"
    if not geojson_path.exists():
        typer.echo(f"Error: {geojson_path} not found.", err=True)
        raise typer.Exit(1)

    with open(geojson_path) as f:
        data = json.load(f)

    total = 0
    total_area = 0.0
    by_year: dict[int, int] = {}
    area_by_year: dict[int, float] = {}

    for feat in data.get("features", []):
        props = feat["properties"]
        ly = props.get("loss_year")
        area = props.get("area_ha", 0)

        if ly is not None and (ly < year_min or ly > year_max):
            continue

        total += 1
        total_area += area
        if ly:
            by_year[ly] = by_year.get(ly, 0) + 1
            area_by_year[ly] = area_by_year.get(ly, 0) + area

    typer.echo(f"\n{aoi} Hotspot Summary")
    if year_min > 0 or year_max < 9999:
        typer.echo(f"  Period: {year_min}–{year_max}")
    typer.echo(f"  Total hotspots: {total:,}")
    typer.echo(f"  Total area: {total_area:,.0f} ha")
    typer.echo(f"\n  By year:")
    for yr in sorted(by_year.keys()):
        typer.echo(f"    {yr}: {by_year[yr]:>5} hotspots, {area_by_year[yr]:>8,.0f} ha")


if __name__ == "__main__":
    app()
