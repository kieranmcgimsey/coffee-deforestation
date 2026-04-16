# Methodology

Detailed explanation of how the system detects, classifies, and reports coffee-linked deforestation.

## Data Sources

| Layer | GEE Collection | Resolution | Temporal Coverage | Role |
|-------|---------------|------------|-------------------|------|
| Sentinel-2 L2A | `COPERNICUS/S2_SR_HARMONIZED` | 10m | Annual composites 2019-2024 | Optical imagery, spectral indices |
| S2 Cloud Probability | `COPERNICUS/S2_CLOUD_PROBABILITY` | 10m | Per-scene | Cloud masking (threshold > 40%) |
| Sentinel-1 GRD | `COPERNICUS/S1_GRD` | 10m | Annual composites 2019-2024 | SAR backscatter (cloud-immune) |
| Hansen GFC v1.12 | `UMD/hansen/global_forest_change_2024_v1_12` | 30m | Cumulative 2001-2024 | Forest loss detection (all causes) |
| FDP Coffee | `projects/forestdatapartnership/assets/coffee/model_2025a/coffee_2023` | 10m | 2023 snapshot | Coffee probability (0-1) |
| ESA WorldCover | `ESA/WorldCover/v200` | 10m | 2021 | Land cover classification |
| SRTM | `USGS/SRTMGL1_003` | 30m | Static | Elevation, slope |

All data is acquired from Google Earth Engine — no local downloads of raw satellite imagery are needed.

## Preprocessing

### Sentinel-2 (optical)

1. **Cloud masking**: For each S2 scene, the matching cloud probability image is joined via `system:index`. Pixels with cloud probability > 40% are masked. The mask is dilated by 50m (`focal_max`) to catch cloud edges.
2. **Compositing**: All cloud-masked scenes within the dry-season window are median-composited into a single annual image. The median naturally suppresses residual cloud/shadow contamination.
3. **Scaling**: Digital numbers are divided by 10,000 to convert to surface reflectance (0-1 range).

Dry-season windows are region-specific (configured per AOI in `aois.yaml`):
- Vietnam: December-March
- Colombia: December-February
- Brazil: June-September

### Sentinel-1 (SAR)

1. **Filtering**: GRD (Ground Range Detected) images with VV and VH polarisation are selected. Both ascending and descending orbit passes are included.
2. **Speckle filtering**: A focal median filter (radius 50m) is applied in the dB domain to reduce SAR speckle noise.
3. **Compositing**: Median composite across the dry-season window, same as S2.

SAR is cloud-immune — it penetrates clouds and works at night. This provides continuous coverage even in persistently cloudy tropical regions where optical data has gaps.

### What's NOT done

- **Shadow masking**: Not implemented. GEE's proxy evaluation model makes complex shadow projection unreliable inside `collection.map()`. The median composite suppresses most shadow contamination.
- **Spatial harmonization**: S1 and S2 are combined at the feature stack level without explicit reprojection. GEE handles implicit alignment via `addBands()`.
- **SAR cloud gap-filling**: There is no pixel-level fusion where SAR fills cloud-masked S2 pixels. The integration is at the feature level (both sensors contribute features to the ML stack).

## Feature Engineering

The system builds a 19-band per-pixel feature stack:

| # | Feature | Source | Description |
|---|---------|--------|-------------|
| 1-5 | NDVI, EVI, NDWI, NBR, SAVI | S2 | Vegetation indices computed from NIR, Red, Green, SWIR bands |
| 6-9 | B4, B8, B11, B12 | S2 | Raw reflectance bands (Red, NIR, SWIR1, SWIR2) |
| 10-12 | VV, VH, VV/VH ratio | S1 | SAR backscatter and cross-polarisation ratio |
| 13-14 | Elevation, Slope | SRTM | Terrain context (coffee grows at specific altitudes) |
| 15 | Distance to forest edge | Hansen | Proximity to intact forest boundary |
| 16 | Distance to road | WorldCover | Proxy via built-up class distance |
| 17 | NDVI delta | S2 multi-year | NDVI(2024) minus NDVI(2019) — vegetation change over 5 years |
| 18-19 | VV stddev, VH stddev | S1 multi-year | Inter-annual SAR variability — stable land vs changing |

Features 17-19 are **temporal features** that capture change over the monitoring period. They are computed from multi-year composites (all 6 years), not just the latest year.

## Hotspot Detection (Rule-Based)

The primary detection method uses a spatial intersection of two datasets:

1. **Hansen forest loss**: Every pixel where `loss == 1` (tree cover lost between 2001-2024)
2. **FDP coffee probability**: Every pixel where `probability > 0.5` (model estimates >50% chance of coffee)

Pixels satisfying BOTH conditions are candidates. Connected candidate pixels are grouped into contiguous polygons ("hotspots") via GEE's `reduceToVectors()`. Each hotspot polygon is:
- Measured in hectares (area computed in UTM projection)
- Assigned a loss year (mode of Hansen `lossyear` values within the polygon, via batch `reduceRegions`)
- Ranked by area (largest first)
- Filtered by minimum area (default 0.5 ha, configurable)

### What this means and doesn't mean

A hotspot means: **"forest was lost here AND coffee is present here now."**

It does NOT mean coffee caused the deforestation. Possible explanations:
1. Forest was cleared specifically to plant coffee (the scenario we aim to detect)
2. Forest was cleared for other reasons, and coffee was planted years later on already-cleared land
3. The FDP model incorrectly identifies the land cover as coffee (false positive)

Ground-truth field verification is essential to distinguish these scenarios.

## Deforestation Attribution

Beyond coffee-linked hotspots, the system classifies ALL Hansen forest loss pixels by their current land cover. This answers: **"What replaced the forest?"**

For each loss pixel:
- If FDP coffee probability > 50% → **coffee**
- Else, remap ESA WorldCover class:
  - Cropland (code 40) → **other agriculture**
  - Built-up (code 50) → **built-up / industrial**
  - Bare/sparse (code 60), grassland (code 30) → **bare / degraded**
  - Water (code 80) → **water**
  - Tree cover (code 10), shrubland (code 20), wetland (code 90) → **forest regrowth**

This is computed via a single `reduceRegion(frequencyHistogram)` call per AOI — very fast. Per-year breakdown uses one call per Hansen loss year (2005-2023).

## Machine Learning

### Why ML if rule-based detection already works?

The rule-based method uses fixed thresholds on external products (Hansen + FDP). The ML pipeline provides:

1. **Feature importance**: Which spectral and SAR features actually distinguish coffee from forest? NDVI? VV backscatter? Elevation?
2. **Cross-region generalisation**: Does coffee look the same spectrally in Vietnam vs Colombia vs Brazil?
3. **Sensor ablation**: Does SAR actually help, or is optical alone sufficient?
4. **Independent validation**: Where our model agrees with FDP → high confidence. Where they disagree → areas needing field verification.

### Training

- **Labels**: FDP coffee probability > 50% → class "coffee". ESA WorldCover → classes "forest", "cropland", "built/bare", "water".
- **Sampling**: 800 balanced samples per class per AOI via GEE `stratifiedSample` at 30m resolution. Total: ~12,000 samples across 3 regions.
- **Models**: Random Forest (500 trees) and XGBoost (500 rounds, max_depth=6), both with balanced class weights.
- **Training data**: Pooled across all AOIs before 80/20 train/test split.

### Evaluation

Three evaluation modes:

1. **Pooled holdout**: Train on 80% of all pooled samples, test on 20%. This is the optimistic estimate (F1 = 0.78).
2. **Cross-AOI holdout**: Train on 2 regions, test on the 3rd (rotated). This tests geographic generalisation (F1 = 0.48-0.63).
3. **Sensor ablation**: Train S1-only, S2-only, and S1+S2 models on the same data. Tests multi-sensor fusion value.

### Results

| Evaluation | RF F1 (coffee) | Interpretation |
|------------|---------------|----------------|
| Pooled holdout | 0.783 | Good — model distinguishes coffee from other classes |
| Cross-AOI: Vietnam held out | 0.491 | Weak — Robusta has different spectral signature |
| Cross-AOI: Colombia held out | 0.483 | Weak — shade-grown Arabica hard to detect |
| Cross-AOI: Brazil held out | 0.625 | Moderate — industrial Arabica more detectable |

| Ablation | F1 | Interpretation |
|----------|-----|----------------|
| S2 only (optical) | 0.670 | Optical carries most signal |
| S1 only (SAR) | 0.588 | SAR alone is weak for coffee |
| S1 + S2 (combined) | 0.682 | Multi-sensor fusion adds ~1-2% F1 |

### Label Circularity

The ML models are trained on labels derived from FDP coffee probability + WorldCover. They are validated against the same label sources. This means reported F1 scores measure **agreement with proxy labels**, not real-world accuracy. The ML cannot, by construction, outperform its label source. Its value is diagnostic (feature importance, generalisation testing), not operational classification.

## LLM Reporting

Three-agent pipeline using the Anthropic Claude API:

1. **Researcher agent**: Receives the AOI stats summary. Uses 6 tools (query_stats, get_hotspot_details, get_historical_context, compare_periods, render_hotspot_map, scratchpad_write) in a tool-use loop (max 8 calls). Produces structured findings.

2. **Writer agent**: Receives researcher findings + raw stats. Produces a 7-section markdown report (executive summary, area context, headline findings, hotspot deep-dives, historical context, model performance, methodology).

3. **Synthesist agent**: Receives all writer reports. Produces a cross-AOI comparison brief.

**Dry-run mode** (default): Deterministic template-based generation using real statistics. No API key needed. Produces structured, data-grounded reports.

**Real mode** (`--no-dry-run`): Actual Claude API calls. Requires `ANTHROPIC_API_KEY` in `.env`. Costs ~$0.10-0.50 per full run.

A **factuality checker** (`factcheck.py`) verifies every number in the generated report against the source stats JSON within 1% relative tolerance.

## Validation

| Layer | Method | Coverage |
|-------|--------|----------|
| Data quality | AOI validation (min coffee %, min forest %, min loss pixels) | All AOIs |
| Schema validation | Pydantic models enforce type safety at every pipeline boundary | All outputs |
| Report factuality | Regex number extraction + cross-check against source JSON | All reports |
| ML validation | Cross-AOI holdout + sensor ablation | All 3 regions |
| Negative control | Brazil AOI confirms lower coffee attribution (32% vs 49-55%) | 1 region |

### What's missing

- **No independent ground truth**: All validation is against satellite-derived products, not field surveys
- **No per-pixel accuracy assessment**: No stratified random sampling protocol
- **No uncertainty quantification**: Point estimates without confidence intervals

Recommended validation steps for operational deployment:
1. Cross-validate against MapBiomas annual LULC for Brazil
2. Acquire Planet NICFI basemap (3-5m) for visual spot-checking
3. Partner with in-country cooperatives for stratified field verification
4. Conduct FDP threshold sensitivity analysis (40% / 50% / 60%)
