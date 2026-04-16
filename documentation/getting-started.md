# Getting Started

Complete setup guide from zero to your first deforestation report.

## Prerequisites

- **Python 3.12+** — the codebase uses modern type syntax (`X | None`, `dict[str, float]`)
- **Conda or Miniconda** — for environment isolation and XGBoost native dependencies
- **A Google Earth Engine account** — [sign up free](https://earthengine.google.com/) (non-commercial tier is sufficient)
- **~2 GB disk space** for outputs (figures, maps, stats, vectors, rasters)
- **(Optional) An Anthropic API key** — only needed for real LLM-generated reports. Dry-run mode produces real-data reports without one.

## Step 1: Install

```bash
# Create isolated conda environment
conda create -n coffee-deforestation python=3.12 -y
conda activate coffee-deforestation

# Install uv (fast Python package manager) and all dependencies
pip install uv
uv sync --extra dev
```

### macOS XGBoost fix

XGBoost on Apple Silicon needs the OpenMP library path set:

```bash
# One-time: install OpenMP
conda install -c conda-forge llvm-openmp -y

# Set for current session
export DYLD_LIBRARY_PATH=$(conda info --base)/envs/coffee-deforestation/lib:$DYLD_LIBRARY_PATH

# Or make it permanent via conda activation script:
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export DYLD_LIBRARY_PATH=$CONDA_PREFIX/lib:$DYLD_LIBRARY_PATH' > $CONDA_PREFIX/etc/conda/activate.d/libomp.sh
```

## Step 2: Configure Google Earth Engine

The pipeline acquires all satellite data from GEE. You need to authenticate once.

### Option A: Service account (recommended for automation)

1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Create a project (or use an existing one)
3. Enable the **Earth Engine API** in APIs & Services
4. Create a **Service Account** in IAM & Admin → Service Accounts
5. Download the JSON key file
6. Register the service account at [signup.earthengine.google.com/#!/service_accounts](https://signup.earthengine.google.com/#!/service_accounts)

```bash
# Save credentials to .env (gitignored)
echo 'GEE_SERVICE_ACCOUNT_KEY_PATH=/path/to/your-key.json' >> .env
echo 'GEE_PROJECT=your-gce-project-id' >> .env
```

### Option B: Interactive (quick personal use)

```bash
uv run python -c "import ee; ee.Authenticate(); ee.Initialize(project='your-project-id')"
```

This opens a browser for Google sign-in. Credentials are cached locally.

## Step 3: Validate

Check that GEE is working and the default AOIs are accessible:

```bash
uv run python scripts/validate_aois.py
```

Expected output:
```
lam_dong        | Coffee: 36.80% | Forest-2000: 55.80% | Loss pixels:    12000 | [PASS]
huila           | Coffee: 28.50% | Forest-2000: 62.30% | Loss pixels:     9500 | [PASS]
sul_de_minas    | Coffee: 24.70% | Forest-2000: 30.10% | Loss pixels:    21000 | [PASS]
```

If an AOI fails, it means the FDP coffee layer doesn't have sufficient coverage in that bbox. Adjust the coordinates in `config/aois.yaml`.

## Step 4: Run the Pipeline

### Single AOI (fastest way to test)

```bash
uv run python scripts/run_aoi.py --aoi lam_dong
```

This takes ~30 seconds and produces:
- `outputs/vectors/hotspots_lam_dong.geojson` — 5,000 hotspot polygons with loss_year
- `outputs/vectors/hotspots_lam_dong.csv` — same data in spreadsheet-friendly format
- `outputs/stats/summary_lam_dong.json` — structured statistics
- `outputs/figures/lam_dong/` — 16 diagnostic figures
- `outputs/maps/map_lam_dong.html` — basic interactive map
- `outputs/reports/report_lam_dong.md` — dry-run markdown report

### All AOIs + ML training

```bash
uv run python scripts/run_all.py
```

This runs all 3 AOIs, then pools training data for cross-AOI ML evaluation (RF + XGBoost, ablation study, cross-AOI holdout).

## Step 5: Generate Interactive Maps

```bash
uv run python scripts/generate_maps.py
```

Creates interactive HTML maps with:
- GEE satellite tile layers at full 10m resolution (Sentinel-2 RGB, NDVI, FDP coffee probability, Hansen loss)
- Hotspot polygons colored by loss year (yellow = old, red = recent)
- Year slider with Play button — animate deforestation spreading 2001-2024
- EUDR post-2020 risk highlighting

## Step 6: Generate Reports

```bash
# HTML reports (self-contained, embeds figures)
uv run python scripts/generate_report.py

# Markdown reports via LLM agents (dry-run by default)
uv run python scripts/generate_reports.py
```

Open `outputs/reports/report_lam_dong.html` in your browser.

### With real LLM (optional)

```bash
echo 'ANTHROPIC_API_KEY=sk-ant-your-key' >> .env
uv run python scripts/generate_reports.py --no-dry-run --aoi lam_dong
```

This calls the Anthropic Claude API to generate narrative reports. Costs ~$0.10-0.50 per full run across 3 AOIs.

## Step 7: Run Deforestation Attribution + Temporal Analysis

```bash
uv run python scripts/run_full_analysis.py
```

This adds:
- **Deforestation attribution**: classifies ALL forest loss by replacement land cover (coffee, crops, built-up, bare, regrowth)
- **Temporal analysis**: before/after RGB composites, NDVI change maps, per-year loss statistics from GEE
- **Attribution figures**: pie charts and stacked bar charts by year

## Step 8: Query Hotspots

```bash
# Find hotspots within 15km of a GPS point, post-2020 only
uv run python scripts/query_hotspots.py near 11.9 108.3 --km 15 --year-min 2020

# Summary statistics for a period
uv run python scripts/query_hotspots.py summary --aoi lam_dong --year-min 2020
```

## What You Should See

After a full run, your `outputs/` directory will contain:

```
outputs/
├── figures/lam_dong/          16 PNGs (RGB, NDVI, SAR, attribution, classification, etc.)
├── figures/huila/             16 PNGs
├── figures/sul_de_minas/      16 PNGs
├── maps/                      Interactive HTML maps with GEE tiles
├── rasters/                   Classification GeoTIFFs (~300m resolution)
├── reports/                   HTML + markdown reports
├── stats/                     Per-AOI summary JSONs
└── vectors/                   Hotspot GeoJSON + CSV + GeoPackage
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `EEException: Earth Engine client library not initialized` | Run `validate_aois.py` to check auth. Re-run `ee.Authenticate()` if needed. |
| `User memory limit exceeded` | AOI bbox is too large. Reduce to ~1° × 1° (100km). |
| `Collection query aborted after accumulating over 5000 elements` | Increase `min_hotspot_area_ha` in `pipeline.yaml` or reduce bbox size. |
| XGBoost `Library not loaded: libomp.dylib` | Set `DYLD_LIBRARY_PATH` — see macOS fix above. |
| `ModuleNotFoundError` | Run `uv sync --extra dev` to install all dependencies. |
| FDP coffee probability is zero everywhere | The FDP model doesn't cover this region. Check coverage by toggling the FDP layer on the interactive map. |
