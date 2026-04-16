# Architecture & Design Decisions

Technical architecture, data flow, caching strategy, and key design choices.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Google Earth Engine (cloud)                    │
│  Sentinel-2  Sentinel-1  Hansen GFC  FDP Coffee  WorldCover     │
└──────────┬──────────┬────────┬──────────┬──────────┬────────────┘
           │          │        │          │          │
           ▼          ▼        ▼          ▼          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Pipeline (pipeline.py)                         │
│                                                                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐         │
│  │ Validate │→│ Composites│→│ Features  │→│  Change   │         │
│  │   AOI    │  │  S2 + S1  │  │  19-band  │  │ Detection │         │
│  └──────────┘  └──────────┘  │   stack   │  │ Hotspots  │         │
│                               └──────────┘  └─────┬─────┘         │
│                                                     │               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐         │               │
│  │    ML    │  │   Stats  │  │  Report  │◄────────┘               │
│  │ RF/XGB   │  │ Summary  │  │  Agents  │                         │
│  └──────────┘  └──────────┘  └──────────┘                         │
└─────────────────────────────────────────────────────────────────┘
           │          │        │          │
           ▼          ▼        ▼          ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Outputs (local)                            │
│  GeoJSON/CSV    Stats JSON    HTML Reports    Interactive Maps    │
│  GeoTIFFs       Figures       Markdown        Year Slider         │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow

1. **GEE server-side**: All satellite data acquisition and heavy computation happens on GEE servers. The pipeline constructs `ee.Image` objects (lazy computation graphs) that are only evaluated when results are needed.

2. **Local downloads**: Three mechanisms pull data from GEE to local:
   - `reduceRegion()` — aggregate statistics (mean NDVI, pixel counts). Returns a small dict. Used for validation, attribution, per-year stats.
   - `reduceToVectors()` → `getInfo()` — converts raster to vector polygons. Returns GeoJSON. Limited to 5,000 features. Used for hotspot detection.
   - `sampleRectangle()` — downloads a raster patch as numpy arrays. Limited to 262,144 pixels. Used for figures (300m resolution) and ML classification maps.

3. **GEE tile serving**: `ee.Image.getMapId()` returns tile URLs that folium/leaflet can render at any zoom level. This provides full 10m resolution in interactive maps without any pixel budget constraint. Requires authenticated session.

## Caching Strategy

Every expensive operation is cached using content-addressed hashing:

```python
@cached(stage="s2_composite")
def build_s2_composite(aoi, year, config):
    ...
```

The `@cached` decorator:
1. Computes a SHA-256 hash of all arguments (pydantic models serialize canonically)
2. Checks if `outputs/cache/{stage}/{hash}/` exists
3. If yes, returns cached result. If no, runs the function and stores the result.

Cache keys include all input parameters, so changing any config value (bbox, threshold, year range) automatically invalidates the relevant cache entries. No manual cache management needed.

Force re-computation: `uv run python scripts/clear_cache.py --stage s2_composite` or `--all`.

## Key Design Decisions

### GEE-only data access (no Drive export)

**Decision**: All data stays on GEE servers. No GeoTIFF exports to Google Drive.

**Why**: Service accounts don't have Drive storage quota. Export-to-Drive requires personal OAuth credentials, which complicates deployment.

**Trade-off**: Classification GeoTIFFs are limited to ~300m resolution (sampleRectangle pixel budget). Interactive maps use GEE tile serving at full 10m to compensate.

**To change**: Re-enable `export_and_download` in `pipeline.py` with personal OAuth auth. The function exists in `data/gee_client.py`.

### Dry-run LLM agents by default

**Decision**: All three reporting agents (researcher, writer, synthesist) default to dry-run mode using deterministic templates with real statistics.

**Why**: Saves API cost during development. Produces consistent, verifiable outputs. No API key needed for most users.

**Trade-off**: Dry-run reports read like filled-in forms rather than natural language analysis. Real Claude API calls produce richer, more contextual reports.

**To change**: Set `ANTHROPIC_API_KEY` in `.env` and run with `--no-dry-run`.

### Content-addressed caching (not time-based)

**Decision**: Cache keys are SHA-256 of input parameters, not timestamps.

**Why**: Deterministic — same inputs always produce the same cache key. Re-running with identical config is instant. Changing any parameter triggers recomputation automatically.

**Trade-off**: If GEE updates a dataset (e.g., new Hansen version), old cache entries are not invalidated. Use `--force` or `clear_cache.py` to refresh.

### Cross-AOI ML training (not per-region)

**Decision**: Training data is pooled from all AOIs before ML training. Models are evaluated via cross-AOI holdout (train on 2, test on 1).

**Why**: Tests geographic generalisation — the scientifically interesting question. Per-AOI training would overfit to individual regions without revealing whether the approach works globally.

**Trade-off**: Pooled F1 (0.78) is misleadingly high. The honest metric is cross-AOI holdout F1 (0.48-0.63), which shows the model does NOT generalise well.

### 5,000 hotspot cap per AOI

**Decision**: Hotspot FeatureCollections are limited to 5,000 features via `.limit(5000)` before `getInfo()`.

**Why**: GEE's `getInfo()` hard-fails above 5,000 elements. The limit sorts by area (largest first), so the most significant hotspots are always retained.

**Trade-off**: In dense regions, smaller hotspots (<1 ha) may be excluded. Aggregate statistics (total area, attribution fractions) are unaffected because they use `reduceRegion` which has no element limit.

## Module Responsibilities

| Module | What it does | Depends on |
|--------|-------------|-----------|
| `config.py` | Loads YAML config, validates with pydantic | — |
| `cache.py` | Content-addressed caching decorator | config |
| `data/sentinel2.py` | S2 cloud-masked composites | config, GEE |
| `data/sentinel1.py` | S1 speckle-filtered composites | config, GEE |
| `data/ancillary.py` | Hansen, FDP, WorldCover, SRTM | config, GEE |
| `features/stack.py` | 19-band feature assembly | data/*, features/* |
| `change/hansen_overlay.py` | Rule-based hotspot detection | data/ancillary |
| `change/hotspots.py` | Polygonise, rank, export | change/hansen_overlay |
| `change/deforestation_attribution.py` | ALL-loss replacement classification | data/ancillary |
| `change/temporal.py` | Before/after composites, NDVI change | data/* |
| `ml/labels.py` | GEE label sampling | data/ancillary, features/stack |
| `ml/train.py` | RF + XGBoost training | ml/labels |
| `ml/evaluate.py` | Cross-AOI holdout, ablation | ml/train |
| `ml/predict.py` | Apply model to feature stack | ml/train, features/stack |
| `stats/schema.py` | Pydantic models for all JSON contracts | — |
| `stats/summarize.py` | Build per-AOI summary JSON | all upstream modules |
| `reporting/llm_client.py` | Anthropic API client + dry-run | config |
| `reporting/agents/*.py` | Researcher, writer, synthesist | reporting/tools, llm_client |
| `reporting/factcheck.py` | Number verification in reports | stats/schema |
| `viz/static.py` | All matplotlib figures | viz/theme |
| `viz/interactive.py` | Folium maps with GEE tiles + year slider | data/*, change/* |
| `pipeline.py` | Orchestrates all stages per AOI | everything |

## Testing Strategy

- **Unit tests**: Pure Python logic (schema validation, feature ordering, factcheck regex, config parsing). No GEE dependency.
- **Mocked GEE tests**: `patch.dict("sys.modules", {"ee": mock_ee})` for modules that import ee at function level. Tests pipeline logic without live GEE.
- **Figure tests**: Generate plots with synthetic numpy arrays, verify PNG files are created.
- **Integration**: Full pipeline runs against live GEE (not in CI — requires credentials).

Current: 211 tests, 69% coverage. Modules at <50% are GEE-dependent and cannot be unit-tested without live Earth Engine.
