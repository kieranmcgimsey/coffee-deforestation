"""SAR-derived features from Sentinel-1.

What: Extracts VV/VH statistics and ratios from S1 composites.
Why: SAR backscatter patterns distinguish forest canopy (high VH) from
open agriculture (low VH), and temporal stability indicates land cover type.
Assumes: Input is a speckle-filtered S1 composite with VV, VH bands in dB.
Produces: ee.Image with SAR feature bands.
"""

from __future__ import annotations

import ee


def compute_sar_features(s1: ee.Image) -> ee.Image:
    """Compute SAR features from a single S1 composite.

    Returns: ee.Image with bands: vv_median, vh_median, vv_vh_ratio
    (Named with _median suffix for consistency; these are from the median composite.)
    """
    vv = s1.select("VV").rename("vv_median")
    vh = s1.select("VH").rename("vh_median")

    # VV/VH ratio (in dB domain = VV - VH)
    vv_vh_ratio = vv.subtract(vh).rename("vv_vh_ratio")

    return vv.addBands([vh, vv_vh_ratio])


def compute_temporal_sar_features(
    s1_composites: dict[int, ee.Image],
) -> ee.Image:
    """Compute temporal statistics across multi-year S1 composites.

    Returns: ee.Image with bands: vv_stddev, vh_stddev (inter-annual variability).
    """
    vv_collection = ee.ImageCollection(
        [img.select("VV") for img in s1_composites.values()]
    )
    vh_collection = ee.ImageCollection(
        [img.select("VH") for img in s1_composites.values()]
    )

    vv_stddev = vv_collection.reduce(ee.Reducer.stdDev()).rename("vv_stddev")
    vh_stddev = vh_collection.reduce(ee.Reducer.stdDev()).rename("vh_stddev")

    return vv_stddev.addBands(vh_stddev)
