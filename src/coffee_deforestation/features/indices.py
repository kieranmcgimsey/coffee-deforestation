"""Spectral vegetation indices from Sentinel-2.

What: Computes NDVI, EVI, NDWI, NBR, and SAVI from S2 reflectance bands.
Why: Vegetation indices are the primary discriminators between coffee, forest,
and other land cover types.
Assumes: Input is a cloud-masked S2 composite with bands B2-B12 scaled to 0-1.
Produces: ee.Image with index bands added.
"""

from __future__ import annotations

import ee


def compute_ndvi(s2: ee.Image) -> ee.Image:
    """Normalized Difference Vegetation Index: (NIR - Red) / (NIR + Red)."""
    return s2.normalizedDifference(["B8", "B4"]).rename("ndvi")


def compute_evi(s2: ee.Image) -> ee.Image:
    """Enhanced Vegetation Index: 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)."""
    nir = s2.select("B8")
    red = s2.select("B4")
    blue = s2.select("B2")
    evi = nir.subtract(red).multiply(2.5).divide(
        nir.add(red.multiply(6)).subtract(blue.multiply(7.5)).add(1)
    )
    return evi.rename("evi")


def compute_ndwi(s2: ee.Image) -> ee.Image:
    """Normalized Difference Water Index: (Green - NIR) / (Green + NIR)."""
    return s2.normalizedDifference(["B3", "B8"]).rename("ndwi")


def compute_nbr(s2: ee.Image) -> ee.Image:
    """Normalized Burn Ratio: (NIR - SWIR2) / (NIR + SWIR2)."""
    return s2.normalizedDifference(["B8", "B12"]).rename("nbr")


def compute_savi(s2: ee.Image, l: float = 0.5) -> ee.Image:
    """Soil Adjusted Vegetation Index: ((NIR - Red) / (NIR + Red + L)) * (1 + L)."""
    nir = s2.select("B8")
    red = s2.select("B4")
    savi = nir.subtract(red).divide(nir.add(red).add(l)).multiply(1 + l)
    return savi.rename("savi")


def compute_all_indices(s2: ee.Image) -> ee.Image:
    """Compute all spectral indices and add as bands to the input image."""
    return s2.addBands([
        compute_ndvi(s2),
        compute_evi(s2),
        compute_ndwi(s2),
        compute_nbr(s2),
        compute_savi(s2),
    ])
