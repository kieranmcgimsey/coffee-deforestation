"""Google Earth Engine client with authentication, export, and polling.

What: Manages GEE authentication (service account or interactive), image export to
Google Drive, task polling, and local file mirroring.
Why: Centralizes all GEE interaction so other modules work with clean abstractions.
Assumes: GEE account is authorized. Service account key path set in .env (optional).
Produces: Authenticated GEE session; exported GeoTIFFs downloaded locally.
"""

from __future__ import annotations

import time
from pathlib import Path

import ee
from loguru import logger

from coffee_deforestation.config import PROJECT_ROOT, AOIConfig, Settings, load_settings

_initialized = False


def init_gee(settings: Settings | None = None) -> None:
    """Initialize GEE with service account or interactive auth."""
    global _initialized
    if _initialized:
        return

    if settings is None:
        settings = load_settings()

    project = settings.gee_project or None

    try:
        if settings.gee_service_account_key_path:
            credentials = ee.ServiceAccountCredentials(
                email=None,
                key_file=settings.gee_service_account_key_path,
            )
            ee.Initialize(credentials, project=project)
            logger.info("GEE initialized with service account")
        else:
            ee.Initialize(project=project)
            logger.info(f"GEE initialized with default credentials (project={project})")
        _initialized = True
    except (ValueError, FileNotFoundError, ImportError, RuntimeError) as e:
        logger.warning(f"GEE default init failed ({e}), attempting interactive auth...")
        ee.Authenticate()
        ee.Initialize(project=project)
        _initialized = True
        logger.info("GEE initialized via interactive auth")


def aoi_to_geometry(aoi: AOIConfig) -> ee.Geometry:
    """Convert an AOI config to a GEE geometry rectangle."""
    return ee.Geometry.Rectangle(aoi.bbox.to_list())


def export_image_to_drive(
    image: ee.Image,
    description: str,
    aoi: AOIConfig,
    folder: str | None = None,
    scale: int = 10,
    max_pixels: int = 1_000_000_000,
    crs: str | None = None,
) -> ee.batch.Task:
    """Start an export of a GEE image to Google Drive.

    Returns the task object for polling.
    """
    if folder is None:
        settings = load_settings()
        folder = settings.google_drive_export_folder

    if crs is None:
        crs = f"EPSG:{aoi.epsg_utm}"

    region = aoi_to_geometry(aoi)

    task = ee.batch.Export.image.toDrive(
        image=image,
        description=description,
        folder=folder,
        region=region,
        scale=scale,
        crs=crs,
        maxPixels=max_pixels,
        fileFormat="GeoTIFF",
    )
    task.start()
    logger.info(f"Started GEE export: {description}")
    return task


def poll_task(task: ee.batch.Task, poll_interval: int = 30, max_retries: int = 120) -> bool:
    """Poll a GEE task until completion. Returns True on success."""
    for i in range(max_retries):
        status = task.status()
        state = status.get("state", "UNKNOWN")

        if state == "COMPLETED":
            logger.info(f"GEE task completed: {status.get('description', 'unknown')}")
            return True
        elif state in ("FAILED", "CANCELLED"):
            error = status.get("error_message", "No error message")
            logger.error(f"GEE task {state}: {error}")
            return False
        else:
            if i % 4 == 0:  # Log every ~2 minutes
                logger.debug(f"GEE task {state}: {status.get('description', '')} (poll {i+1})")
            time.sleep(poll_interval)

    logger.error(f"GEE task timed out after {max_retries * poll_interval}s")
    return False


def export_and_download(
    image: ee.Image,
    description: str,
    aoi: AOIConfig,
    output_dir: Path,
    scale: int = 10,
    crs: str | None = None,
) -> Path | None:
    """Export a GEE image to Drive, poll until done, then download locally.

    Returns the local path to the downloaded GeoTIFF, or None on failure.
    Retries up to 3 times with exponential backoff on failure.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    local_path = output_dir / f"{description}.tif"

    if local_path.exists():
        logger.debug(f"Already downloaded: {local_path}")
        return local_path

    for attempt in range(3):
        try:
            task = export_image_to_drive(
                image=image,
                description=description,
                aoi=aoi,
                scale=scale,
                crs=crs,
            )
            success = poll_task(task)
            if success:
                # Download from Drive using the Drive client
                from coffee_deforestation.data.drive_client import download_from_drive

                downloaded = download_from_drive(description, local_path)
                if downloaded:
                    return local_path
        except Exception as e:
            wait = 2 ** (attempt + 1) * 30
            logger.warning(f"Export attempt {attempt + 1} failed: {e}. Retrying in {wait}s...")
            time.sleep(wait)

    logger.error(f"All export attempts failed for {description}")
    return None


def get_image_info(image: ee.Image, aoi: AOIConfig) -> dict:
    """Get basic info about a GEE image (band names, scale, etc.)."""
    region = aoi_to_geometry(aoi)
    info = image.getInfo()
    band_names = [b["id"] for b in info.get("bands", [])]
    return {
        "band_names": band_names,
        "band_count": len(band_names),
    }


def compute_stats(
    image: ee.Image,
    aoi: AOIConfig,
    scale: int = 100,
) -> dict:
    """Compute mean/min/max stats for an image over an AOI. Used for validation."""
    region = aoi_to_geometry(aoi)
    stats = image.reduceRegion(
        reducer=ee.Reducer.mean().combine(ee.Reducer.minMax(), sharedInputs=True),
        geometry=region,
        scale=scale,
        maxPixels=1_000_000,
        bestEffort=True,
    )
    return stats.getInfo()
