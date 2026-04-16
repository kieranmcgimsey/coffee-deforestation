# Extending the System

How to add new regions, scale to global coverage, adapt for other commodities, and improve ML performance.

## Adding a New Region

### 1. Choose your bounding box

You need the lat/lon coordinates of the area you want to monitor. Tools for finding bbox coordinates:
- [bboxfinder.com](http://bboxfinder.com/) — draw a rectangle on a map
- Google Maps — right-click to get coordinates
- QGIS — draw a polygon and read the extent

Keep each AOI to roughly **100km × 100km** (about 1° × 1° in the tropics). Larger regions hit GEE processing limits.

### 2. Determine the dry season

The pipeline builds cloud-free composites from the dry season (least cloud cover). You need start/end months:

| Region | Typical Dry Season | cross_year |
|--------|-------------------|------------|
| Southeast Asia (Vietnam, Indonesia) | December-March | true |
| Central America / Colombia | December-February | true |
| East Africa (Ethiopia, Kenya) | October-February | true |
| Brazil (south) | June-September | false |
| West Africa (Ivory Coast, Ghana) | November-March | true |

If unsure, check [climatemps.com](https://www.climatemps.com/) or Google "dry season [your country]".

`cross_year: true` means the dry season spans a year boundary (e.g., Dec 2023 to Mar 2024 is the "2024" composite).

### 3. Add to config

Edit `config/aois.yaml`:

```yaml
sidama:
  name: "Sidama Zone"
  country: "Ethiopia"
  coffee_type: "Arabica, wild"
  role: "East African highland coffee"
  bbox:
    west: 38.0
    south: 6.0
    east: 39.0
    north: 7.0
  dry_season:
    start_month: 10
    end_month: 2
    cross_year: true
  # epsg_utm is auto-computed from the bbox center — you don't need to specify it
```

**You do NOT need to know the UTM zone.** The system auto-computes it from the bbox center coordinates.

### 4. Validate and run

```bash
uv run python scripts/validate_aois.py          # check GEE can access data
uv run python scripts/run_aoi.py --aoi sidama   # run pipeline (~30 seconds)
uv run python scripts/generate_maps.py --aoi sidama  # interactive map
uv run python scripts/generate_report.py         # HTML report
```

### 5. Check FDP coverage

The FDP coffee probability model does NOT cover the entire globe uniformly. Some regions have strong predictions; others have nothing. After running the pipeline:

1. Open the interactive map
2. Toggle on the "FDP Coffee Probability" layer
3. If it's blank or near-zero, the FDP model doesn't cover your region

If FDP has poor coverage, your hotspot detection will find very few results — this is a data limitation, not a pipeline bug.

## Scaling to Country or Global Coverage

### Architecture for scale

The system processes each AOI independently. Scaling to a country is just defining more tiles in `aois.yaml`. The pipeline loops over them sequentially.

```
50 tiles × 30 seconds each = 25 minutes for a major coffee-producing country
```

### Step-by-step: monitoring all of Vietnam's coffee regions

```yaml
# config/aois.yaml — add multiple tiles
vietnam_north:
  name: "Dak Lak North"
  bbox: {west: 107.5, south: 12.5, east: 108.5, north: 13.5}
  # ... other fields ...

vietnam_central:
  name: "Lam Dong Central"
  bbox: {west: 107.8, south: 11.4, east: 108.8, north: 12.4}
  # ... other fields ...

vietnam_south:
  name: "Binh Phuoc"
  bbox: {west: 106.5, south: 11.0, east: 107.5, north: 12.0}
  # ... other fields ...
```

Then `uv run python scripts/run_all.py` processes all of them, trains ML across all tiles, and generates a cross-region synthesis.

### Handling the 5,000 hotspot cap

GEE's `getInfo()` has a hard limit of 5,000 elements per FeatureCollection. In dense coffee regions, a single tile can easily exceed this.

**Option 1: Increase minimum area** (simplest)

In `config/pipeline.yaml`, change:
```yaml
change_detection:
  min_hotspot_area_ha: 2.0  # was 0.5
```
This keeps only larger hotspots, reducing count while retaining the most significant ones.

**Option 2: Use raster-only statistics**

For screening purposes, you don't need individual hotspot polygons. The deforestation attribution analysis (`run_full_analysis.py`) uses `reduceRegion` which has no element count limit — it returns aggregate statistics (total area, per-year breakdown, replacement fractions) for arbitrarily large regions.

**Option 3: Tile the extraction**

Split each 100km tile into 4 sub-tiles of 50km, process hotspots for each, then merge the GeoJSON files locally:

```python
# Pseudocode — not yet implemented
for sub_tile in split_bbox(main_bbox, n=4):
    hotspots = run_aoi(sub_tile)
    all_hotspots.extend(hotspots)
```

**Option 4: GEE batch export**

Export the hotspot FeatureCollection to Google Cloud Storage as a GeoJSON asset. This bypasses the 5,000 limit entirely but requires OAuth credentials (not a service account) or a GCS bucket. The `data/gee_client.py` has an `export_image_to_drive` function that could be adapted for FeatureCollection export.

### Identifying coffee zones to tile

You don't need to manually define every tile. Sources for finding where coffee grows:

1. **FDP layer itself**: Run a coarse scan at 1km resolution across the tropics, keeping tiles where mean FDP probability > 5%
2. **ICO data**: The [International Coffee Organization](https://ico.org/) publishes production statistics by country and region
3. **FAO GAUL**: Administrative boundaries crossed with agricultural census data
4. **Scientific literature**: Published coffee suitability maps (e.g., Bunn et al. 2015)

## Improving ML Performance

### The generalisation problem

Our cross-AOI holdout shows that coffee spectral signatures vary significantly across regions:

| Held-out region | F1 | Why |
|----------------|-----|-----|
| Vietnam | 0.491 | Robusta grown as monoculture — looks different from shade-grown Arabica |
| Colombia | 0.483 | Shade-grown at high altitude — spectrally similar to forest |
| Brazil | 0.625 | Industrial monoculture — most distinct spectral signature |

A model trained on 2 regions fails to detect coffee in the 3rd. This is expected — coffee varietals, growing practices, elevation, and surrounding landscape all affect the spectral signature.

### Strategy 1: Regional model bank

Maintain separate models per coffee geography:
- Southeast Asia Robusta (trained on Vietnam + Indonesia)
- Andean Arabica (trained on Colombia + Peru)
- East African Arabica (trained on Ethiopia + Kenya)
- Brazilian industrial (trained on Minas Gerais + Espírito Santo)

Route each new AOI to the most appropriate model based on geography and coffee type. The existing cross-AOI holdout framework provides the F1 threshold for deciding when a model applies (F1 > 0.70) vs when retraining is needed.

### Strategy 2: Few-shot fine-tuning

For a new region:
1. Run the pipeline with the rule-based method (Hansen + FDP) — no ML needed
2. Sample 200-500 labeled points from the FDP/WorldCover overlay in the new region
3. Fine-tune the base model on these new samples + 50% of the original training pool
4. Evaluate on a held-out split from the new region

The pipeline already supports this — `ml/labels.py` samples training data per AOI, and `ml/train.py` pools samples before training.

### Strategy 3: Feature engineering improvements

The current 19-feature stack could be extended:
- **GLCM texture features**: Grey-level co-occurrence matrix features from SAR capture canopy structure better than raw VV/VH
- **Phenology features**: Coffee has seasonal NDVI patterns (flowering, cherry ripening) that differ from forest
- **Multi-temporal composites**: Include multiple seasons per year (wet + dry) instead of just dry season
- **Sentinel-2 Red Edge bands**: B5, B6, B7 are sensitive to leaf chlorophyll content — potentially useful for distinguishing coffee from other crops

## Adapting for Other Commodities

The system is commodity-agnostic except for the FDP coffee probability layer. Replacing this layer adapts the system for any crop.

### Palm oil

Replace FDP with:
- IIASA palm oil probability map (available on GEE)
- WRI palm oil concession boundaries (vector overlay)

Palm oil is actually easier to detect than coffee — oil palm plantations have a very uniform canopy with strong VV SAR return and distinctive NDVI seasonality.

### Cocoa

Replace FDP with:
- ETH Zurich cocoa map for West Africa
- Custom classification using the ML pipeline (cocoa is spectrally between coffee and forest)

The S1/S2 ablation framework would reveal whether SAR adds value for cocoa detection (cocoa is typically shade-grown, making optical detection difficult).

### Soy and cattle ranching

Replace FDP with:
- MapBiomas annual LULC (Brazil) — has specific soy and pasture classes
- GLAD cropland extent maps (global)

Soy has a very distinct temporal NDVI signature: rapid green-up → plateau → harvest → bare soil. The temporal features (ndvi_delta, SAR variability) would capture this well.

### Multi-commodity EUDR compliance

The EU Deforestation Regulation covers 7 commodities: coffee, cocoa, palm oil, soy, beef (pasture), rubber, and wood. A multi-commodity version of this system would:

1. Run all relevant probability layers in parallel for each AOI
2. Attribute each loss pixel to the most likely commodity (highest probability wins)
3. Produce a unified risk assessment per sourcing region
4. Flag post-December 2020 loss for EUDR compliance

The pipeline architecture supports this — the attribution module (`change/deforestation_attribution.py`) already classifies all loss by replacement land cover. Adding more commodity layers is straightforward.
