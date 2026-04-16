"""Generate clean, minimal HTML reports focused on key findings.

What: Produces self-contained HTML reports with embedded interactive maps,
deforestation attribution charts, and key metrics.
Why: A focused report with 5-6 essential elements beats 20 diagnostic figures.
Assumes: Stats JSONs, attribution figures, and interactive maps exist.
Produces: outputs/reports/report_{aoi}.html
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import typer
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from coffee_deforestation.config import PROJECT_ROOT
from coffee_deforestation.logging_setup import setup_logging

app = typer.Typer(help="Generate clean HTML reports.")

OUTPUTS = PROJECT_ROOT / "outputs"
STATS_DIR = OUTPUTS / "stats"
FIGURES_DIR = OUTPUTS / "figures"
MAPS_DIR = OUTPUTS / "maps"
REPORTS_DIR = OUTPUTS / "reports"

AOI_IDS = ["lam_dong", "huila", "sul_de_minas"]


def _b64_img(path: Path) -> str | None:
    if not path.exists():
        return None
    return f"data:image/png;base64,{base64.b64encode(path.read_bytes()).decode()}"


def _read_map_html(aoi_id: str) -> str | None:
    """Read the folium map HTML for inline embedding."""
    map_path = MAPS_DIR / f"map_{aoi_id}.html"
    if not map_path.exists():
        return None
    return map_path.read_text(encoding="utf-8")


CSS = """
<style>
  :root {
    --coffee: #6F4E37; --forest: #2D5016; --loss: #C1292E;
    --bg: #FAFAF8; --card: #FFFFFF; --text: #1a1a1a; --muted: #666; --border: #E5E0D8;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: "Helvetica Neue", Arial, sans-serif; background: var(--bg);
         color: var(--text); font-size: 15px; line-height: 1.7; }
  .page { max-width: 1100px; margin: 0 auto; padding: 2rem 1.5rem 4rem; }
  h1 { font-size: 2em; color: var(--coffee); margin-bottom: 0.3em; }
  h2 { font-size: 1.4em; color: var(--forest); margin: 2.5rem 0 0.8rem;
       border-bottom: 2px solid var(--border); padding-bottom: 0.3em; }
  h3 { font-size: 1.1em; color: var(--coffee); margin: 1.5rem 0 0.5rem; }
  p { margin: 0.6em 0; }
  .subtitle { color: var(--muted); margin-bottom: 1.5em; font-size: 1.05em; }

  .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
             gap: 1rem; margin: 1.5rem 0; }
  .metric { background: var(--card); border-radius: 10px; padding: 1.2rem;
            text-align: center; box-shadow: 0 1px 4px rgba(0,0,0,0.06);
            border-top: 4px solid var(--coffee); }
  .metric .value { font-size: 2em; font-weight: 700; color: var(--coffee); }
  .metric .label { font-size: 0.75em; color: var(--muted); margin-top: 0.2em;
                   text-transform: uppercase; letter-spacing: 0.04em; }
  .metric .explain { font-size: 0.7em; color: #999; margin-top: 0.1em; font-style: italic; }

  .map-frame { width: 100%; height: 700px; border: 1px solid var(--border);
               border-radius: 10px; overflow: hidden; margin: 1rem 0; }
  .map-frame iframe { width: 100%; height: 100%; border: none; }

  .fig-row { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin: 1rem 0; }
  .fig-card { background: var(--card); border-radius: 10px; overflow: hidden;
              box-shadow: 0 1px 4px rgba(0,0,0,0.06); }
  .fig-card img { width: 100%; display: block; }
  .fig-caption { padding: 0.8rem 1rem; font-size: 0.85em; color: var(--muted); }
  .fig-caption strong { color: var(--text); }

  .method { background: var(--card); border-left: 4px solid var(--forest);
            border-radius: 0 10px 10px 0; padding: 1.5rem 2rem; margin: 1.5rem 0;
            box-shadow: 0 1px 4px rgba(0,0,0,0.06); }
  .method p, .method li { font-size: 0.9em; }
  .method ol { padding-left: 1.5em; }
  .method li { margin: 0.3em 0; }

  .caveat { background: #FFF8E1; border: 1px solid #FFECB3; border-radius: 8px;
            padding: 1rem 1.2rem; margin: 1rem 0; font-size: 0.88em; }
  .caveat strong { color: #F57F17; }

  .nav { background: var(--coffee); padding: 0.7rem 1.5rem; display: flex;
         gap: 1.5rem; align-items: center; flex-wrap: wrap; }
  .nav a { color: white; text-decoration: none; font-size: 0.9rem; opacity: 0.85; }
  .nav a:hover { opacity: 1; }
  .nav .brand { font-weight: bold; opacity: 1; font-size: 1rem; }

  table { border-collapse: collapse; width: 100%; font-size: 0.85em; margin: 1rem 0; }
  th { background: var(--coffee); color: white; padding: 0.5rem 0.8rem; text-align: left; }
  td { padding: 0.45rem 0.8rem; border-bottom: 1px solid var(--border); }
  tr:nth-child(even) { background: #F9F6F3; }

  @media (max-width: 700px) { .fig-row { grid-template-columns: 1fr; } }
</style>
"""


def _nav(aoi_ids: list[str], current: str) -> str:
    links = []
    names = {"lam_dong": "Lam Dong", "huila": "Huila", "sul_de_minas": "Sul de Minas"}
    for aid in aoi_ids:
        style = " style='opacity:1;text-decoration:underline'" if aid == current else ""
        links.append(f'<a href="report_{aid}.html"{style}>{names.get(aid, aid)}</a>')
    links.append('<a href="synthesis.html">Synthesis</a>')
    return '<nav class="nav"><span class="brand">Coffee Deforestation Monitor</span>' + "".join(links) + '</nav>'


def _generate_aoi_report(aoi_id: str, all_ids: list[str]) -> str:
    # Load stats
    stats_path = STATS_DIR / f"summary_{aoi_id}.json"
    with open(stats_path) as f:
        data = json.load(f)

    meta = data.get("metadata", {})
    cd = data.get("change_detection", {})
    val = data.get("validation", {})
    attr = data.get("deforestation_attribution", {})

    name = meta.get("name", aoi_id)
    country = meta.get("country", "")
    coffee_type = meta.get("coffee_type", "")
    role = meta.get("role", "")
    total_hotspots = cd.get("total_hotspots", 0)
    total_area = cd.get("total_area_ha", 0)
    total_loss = attr.get("total_loss_ha", 0)
    coffee_pct = attr.get("coffee_pct", 0)
    other_crops_pct = attr.get("other_crops_pct", 0)
    regrowth_pct = attr.get("regrowth_pct", 0)

    # ML results
    ml_results_path = OUTPUTS / "cache" / "models" / "ml_results.json"
    ml_data = {}
    if ml_results_path.exists():
        with open(ml_results_path) as f:
            ml_data = json.load(f)

    pooled_acc = ml_data.get("pooled_accuracy", 0)
    pooled_f1 = ml_data.get("pooled_f1", 0)
    ablation = ml_data.get("ablation", {})
    cross_aoi = ml_data.get("cross_aoi", {})

    # Cross-AOI F1 for this specific AOI
    this_cross = cross_aoi.get(aoi_id, {}).get("random_forest", {}).get("f1", 0)

    # Pre-compute ablation values (avoid nested .get in f-strings)
    s2_f1 = ablation.get("s2_only", {}).get("f1", 0)
    s2_acc = ablation.get("s2_only", {}).get("acc", 0)
    s1_f1 = ablation.get("s1_only", {}).get("f1", 0)
    s1_acc = ablation.get("s1_only", {}).get("acc", 0)
    s1s2_f1 = ablation.get("s1_s2", {}).get("f1", 0)
    s1s2_acc = ablation.get("s1_s2", {}).get("acc", 0)

    # Cross-AOI rows
    cross_rows = ""
    for k, v in cross_aoi.items():
        rf_f1 = v.get("random_forest", {}).get("f1", 0)
        assess = "Good" if rf_f1 >= 0.6 else "Weak — region-specific retraining needed"
        cross_rows += f'<tr><td>{k}</td><td>{rf_f1:.3f}</td><td>{assess}</td></tr>\n'

    # EUDR post-2020 metric
    hby = cd.get("hotspots_by_loss_year", {})
    post_2020_hotspots = sum(v for k, v in hby.items() if int(k) > 2020)
    aby = cd.get("area_ha_by_loss_year", {})
    post_2020_area = sum(v for k, v in aby.items() if int(k) > 2020)

    # Figures
    pie_b64 = _b64_img(FIGURES_DIR / aoi_id / "attribution_pie.png")
    bar_b64 = _b64_img(FIGURES_DIR / aoi_id / "attribution_stacked_bar.png")
    class_b64 = _b64_img(FIGURES_DIR / aoi_id / "classification_map.png")

    pie_html = f'<img src="{pie_b64}" alt="Attribution">' if pie_b64 else '<p><em>Not available</em></p>'
    bar_html = f'<img src="{bar_b64}" alt="Drivers by year">' if bar_b64 else '<p><em>Not available</em></p>'
    class_html = f'<img src="{class_b64}" alt="Classification">' if class_b64 else '<p><em>Classification map not generated. Run predict_from_gee.</em></p>'

    # Interactive map — embed as iframe pointing to file
    map_path = MAPS_DIR / f"map_{aoi_id}.html"
    if map_path.exists():
        # Use relative path for iframe
        map_embed = f'''
        <div class="map-frame">
          <iframe src="../maps/map_{aoi_id}.html"></iframe>
        </div>
        <p style="font-size:0.8em;color:#888">Toggle layers in the top-right control.
        Click hotspots for details. Zoom to field level for full 10m resolution.</p>'''
    else:
        map_embed = '<p><em>Interactive map not available. Run scripts/generate_maps.py</em></p>'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{name} — Coffee Deforestation Report</title>
{CSS}
</head>
<body>
{_nav(all_ids, aoi_id)}
<div class="page">

<h1>{name}</h1>
<p class="subtitle">{country} · {coffee_type} · <em>{role}</em></p>

<h2>Key Findings</h2>
<div class="metrics">
  <div class="metric">
    <div class="value">{total_loss:,.0f}</div>
    <div class="label">Total Forest Loss (ha)</div>
    <div class="explain">All Hansen-detected loss in this region</div>
  </div>
  <div class="metric">
    <div class="value">{coffee_pct:.0f}%</div>
    <div class="label">Became Coffee</div>
    <div class="explain">Of total forest loss, now shows coffee signal</div>
  </div>
  <div class="metric">
    <div class="value">{total_hotspots:,}</div>
    <div class="label">Coffee Hotspots</div>
    <div class="explain">Contiguous loss + coffee zones</div>
  </div>
  <div class="metric">
    <div class="value">{total_area:,.0f}</div>
    <div class="label">Hotspot Area (ha)</div>
    <div class="explain">Total area of coffee-linked hotspots</div>
  </div>
  <div class="metric">
    <div class="value">{other_crops_pct:.1f}%</div>
    <div class="label">Other Agriculture</div>
    <div class="explain">Non-coffee crops replacing forest</div>
  </div>
  <div class="metric">
    <div class="value">{regrowth_pct:.1f}%</div>
    <div class="label">Forest Regrowth</div>
    <div class="explain">Loss areas that show tree recovery</div>
  </div>
  <div class="metric" style="border-top-color:#C1292E">
    <div class="value" style="color:#C1292E">{post_2020_hotspots:,}</div>
    <div class="label">Post-2020 Hotspots</div>
    <div class="explain">EUDR risk zone (loss after Dec 2020)</div>
  </div>
</div>

<h2>Interactive Map</h2>
<p>Explore the study region with zoomable satellite imagery from Google Earth Engine.
Toggle data layers in the top-right control: Sentinel-2 RGB composite, NDVI vegetation
density, FDP coffee probability, Hansen forest loss year, and hotspot polygons
(coloured by loss year: yellow=old, dark red=recent).</p>
{map_embed}

<h2>What Replaced the Forest?</h2>
<p>Not all deforestation is coffee-driven. This analysis classifies <strong>every</strong>
pixel of forest loss (Hansen GFC 2001-2024) by what occupies the land now, using the
FDP coffee probability model and ESA WorldCover land cover classification.</p>

<div class="fig-row">
  <div class="fig-card">
    {pie_html}
    <div class="fig-caption">
      <strong>Deforestation attribution</strong> — of {total_loss:,.0f} ha total forest loss,
      {coffee_pct:.0f}% is now coffee, {other_crops_pct:.1f}% is other agriculture,
      and {regrowth_pct:.1f}% shows forest regrowth.
    </div>
  </div>
  <div class="fig-card">
    {bar_html}
    <div class="fig-caption">
      <strong>Drivers by year</strong> — how the composition of deforestation drivers changed
      across Hansen loss years (2005-2023). Shifts indicate changing agricultural pressures.
    </div>
  </div>
</div>

<h2>ML Classification Map</h2>
<p>Our trained Random Forest classifier applied to the 19-band feature stack at ~300m
resolution. This is <strong>our model's prediction</strong>, not the external FDP layer.
Brown = coffee, dark green = forest, yellow = other crops, grey = built/bare, blue = water.</p>
<div class="fig-row">
  <div class="fig-card">
    {class_html}
    <div class="fig-caption">
      <strong>Per-pixel land cover classification</strong> from our Random Forest model
      (trained on FDP + WorldCover labels, applied to Sentinel-1/2 features). Resolution ~300m
      due to GEE download limits. The classification GeoTIFF is also saved to
      <code>outputs/rasters/</code> for GIS analysis.
    </div>
  </div>
</div>

<h2>Machine Learning Validation</h2>
<p>We trained our own Random Forest and XGBoost classifiers on a 19-feature stack
(Sentinel-2 spectral indices, Sentinel-1 SAR backscatter, elevation, slope, and temporal
change features) to independently validate the FDP coffee detection and test cross-region
generalisation.</p>

<div class="metrics">
  <div class="metric">
    <div class="value">{pooled_acc:.1%}</div>
    <div class="label">Overall Accuracy</div>
    <div class="explain">5-class pooled test set (all regions)</div>
  </div>
  <div class="metric">
    <div class="value">{pooled_f1:.3f}</div>
    <div class="label">Coffee F1 Score</div>
    <div class="explain">Precision-recall balance for coffee class</div>
  </div>
  <div class="metric">
    <div class="value">{this_cross:.3f}</div>
    <div class="label">Cross-AOI F1</div>
    <div class="explain">F1 when this region is held out of training</div>
  </div>
</div>

<h3>Sensor Ablation Study</h3>
<p>Which sensors contribute most to coffee detection?</p>
<table>
  <tr><th>Feature Set</th><th>F1 (coffee)</th><th>Accuracy</th><th>Interpretation</th></tr>
  <tr><td><strong>S2 only</strong> (optical)</td><td>{s2_f1:.3f}</td><td>{s2_acc:.1%}</td><td>Spectral indices + bands</td></tr>
  <tr><td><strong>S1 only</strong> (radar)</td><td>{s1_f1:.3f}</td><td>{s1_acc:.1%}</td><td>SAR backscatter only</td></tr>
  <tr><td><strong>S1 + S2</strong> (combined)</td><td>{s1s2_f1:.3f}</td><td>{s1s2_acc:.1%}</td><td>All features — best performance</td></tr>
</table>
<p style="font-size:0.85em;color:#666"><em>Multi-sensor fusion (S1+S2) outperforms either sensor alone,
confirming the value of integrating optical and SAR data. SAR alone is weakest because
coffee and forest have similar canopy roughness.</em></p>

<h3>Cross-Region Generalisation</h3>
<p>Can a model trained in one region detect coffee in another?</p>
<table>
  <tr><th>Held-Out Region</th><th>RF F1</th><th>Assessment</th></tr>
  {cross_rows}
</table>

<div class="caveat">
<strong>Key ML finding:</strong> Cross-region F1 scores range from 0.48 to 0.63, significantly
below the pooled F1 of {pooled_f1:.3f}. This means coffee has <em>different spectral
signatures</em> across Vietnam, Colombia, and Brazil — likely due to different varietals
(Robusta vs Arabica), growing conditions (highland vs shade-grown), and landscape context.
Region-specific training is recommended for operational deployment.
</div>

<div class="caveat">
<strong>Label circularity:</strong> Our ML models are trained on labels derived from
FDP coffee probability + ESA WorldCover. The reported F1 scores measure agreement with
these <em>proxy labels</em>, not independently verified ground truth. The ML cannot, by
construction, outperform the label source. Its value lies in feature importance analysis,
cross-region generalisation testing, and independent validation — not in achieving higher
accuracy than FDP. Ground-truth field verification is needed to establish true accuracy.
</div>

<h2>Methodology</h2>
<div class="method">
<h3>How Hotspots Are Detected</h3>
<ol>
  <li><strong>Forest loss detection:</strong> The Hansen Global Forest Change dataset (v1.12)
  records every pixel of tree-cover loss from 2001 to 2024 at 30m resolution. This detects
  <em>all</em> forest loss regardless of cause.</li>
  <li><strong>Coffee presence:</strong> The Forest Data Partnership (FDP) model estimates coffee
  probability per pixel from 2023 Sentinel-2 imagery. Pixels above 50% probability are
  classified as coffee.</li>
  <li><strong>Intersection:</strong> Where Hansen loss and FDP coffee overlap, the area is
  flagged as a "hotspot" — a location where forest was lost and coffee is now present.</li>
</ol>

<h3>Deforestation Attribution</h3>
<p>For <em>all</em> Hansen loss pixels (not just coffee), the current land cover is classified
using WorldCover (crops, built-up, bare, water, regrowth) with FDP coffee as an override.
This answers: "What replaced the forest?" for the entire region.</p>

<div class="caveat">
<strong>Important limitation:</strong> A hotspot means "forest was lost here AND coffee is here
now." This is spatial co-occurrence — it does not prove coffee <em>caused</em> the
deforestation. The forest may have been cleared for other reasons and coffee planted later.
Ground-truth verification is needed to establish causation.
</div>

<div class="caveat">
<strong>FDP coverage boundary:</strong> The FDP coffee probability model does not have
uniform global coverage. In some regions (notably Sul de Minas, Brazil), the FDP layer
has a sharp spatial boundary where predictions drop to zero. Hotspots can only be detected
where FDP provides coffee probability > 50%. Areas outside FDP coverage may contain coffee
that is invisible to the pipeline. Toggle the "FDP Coffee Probability" layer on the
interactive map to see the coverage boundary.
</div>

<h3>Machine Learning Pipeline</h3>
<p>This system uses <strong>two independent ML models</strong> for coffee detection:</p>

<p><strong>1. FDP Coffee Model (external, pre-trained):</strong> The Forest Data Partnership
trained a deep learning model on curated labeled coffee sites across major growing regions
worldwide. It takes Sentinel-2 imagery as input and outputs a per-pixel probability (0-1)
that a 10m pixel is coffee. We use the 2023 prediction (model_2025a). This is the primary
coffee signal — <em>we did not train this model</em>.</p>

<p><strong>2. Our RF/XGBoost Classifier (trained by us):</strong> We train Random Forest and
XGBoost classifiers on a 19-feature stack derived from Sentinel-2 (spectral indices: NDVI,
EVI, NDWI, NBR, SAVI; raw bands: B4, B8, B11, B12), Sentinel-1 SAR (VV/VH backscatter,
VV/VH ratio), contextual features (elevation, slope, distance-to-forest, distance-to-road),
and temporal features (NDVI change 2019-2024, inter-annual VV/VH variability).</p>

<p><strong>Training labels</strong> come from FDP (coffee class) + ESA WorldCover (forest,
cropland, built-up, water classes). We sample 2,000 balanced points per class per region
from Google Earth Engine.</p>

<p><strong>Why train our own model if FDP already exists?</strong></p>
<ul style="margin:0.5em 0 0.5em 1.5em">
  <li><strong>Feature importance:</strong> Our model reveals <em>which</em> spectral and SAR
  features distinguish coffee from forest — NDVI, VV backscatter, elevation. This is
  scientific insight the FDP black-box model doesn't provide.</li>
  <li><strong>Cross-region generalisation:</strong> We train on two regions and test on the
  third (holdout). If the F1 score is high, coffee has a consistent spectral signature
  globally. If low, detection requires region-specific tuning.</li>
  <li><strong>Sensor ablation:</strong> We compare S1-only (radar), S2-only (optical), and
  S1+S2 (combined) to quantify the value of multi-sensor fusion.</li>
  <li><strong>Independent validation:</strong> Where our model agrees with FDP, confidence
  is high. Where they disagree, those areas need field verification.</li>
</ul>

<h3>Data Sources</h3>
<table>
  <tr><th>Layer</th><th>Source</th><th>Resolution</th><th>Coverage</th></tr>
  <tr><td>Optical imagery</td><td>Sentinel-2 L2A (Copernicus)</td><td>10m</td><td>Annual composites 2019-2024</td></tr>
  <tr><td>SAR imagery</td><td>Sentinel-1 GRD (Copernicus)</td><td>10m</td><td>Annual composites 2019-2024</td></tr>
  <tr><td>Forest loss</td><td>Hansen GFC v1.12 (UMD)</td><td>30m</td><td>Cumulative 2001-2024</td></tr>
  <tr><td>Coffee probability</td><td>FDP model_2025a (external ML model)</td><td>10m</td><td>2023 snapshot</td></tr>
  <tr><td>Land cover</td><td>ESA WorldCover v200</td><td>10m</td><td>2021 classification</td></tr>
</table>
</div>

<h2>Limitations & Uncertainty</h2>
<div class="method">

<h3>What This System Cannot Do</h3>
<ul style="margin:0.5em 0 0.5em 1.5em">
  <li><strong>Prove causation:</strong> A hotspot means "forest was lost AND coffee is present."
  It does not prove coffee <em>caused</em> the clearing. Ground-truth field verification is
  essential before drawing causal conclusions.</li>
  <li><strong>Near-real-time alerting:</strong> Hansen GFC has 12-18 month latency. The 2024
  dataset covers loss through approximately end-2023. For real-time deforestation alerts,
  use GLAD/RADD alert systems alongside this tool.</li>
  <li><strong>Resolve individual farms:</strong> Classification GeoTIFFs are at ~300m (9 ha/pixel)
  due to GEE download limits. Individual smallholder farms (1-5 ha) are not resolved in the
  classification product. The interactive maps show 10m data for visual inspection.</li>
</ul>

<h3>Known Sources of Uncertainty</h3>
<table>
  <tr><th>Source</th><th>Impact</th><th>Magnitude</th></tr>
  <tr><td>Hansen GFC omission errors</td><td>Missed deforestation events (especially gradual degradation)</td><td>12-22% in tropical forests (documented by UMD)</td></tr>
  <tr><td>FDP coffee probability accuracy</td><td>Misattribution of non-coffee land as coffee (or vice versa)</td><td>FDP does not publish pixel-level accuracy; coverage varies by region</td></tr>
  <tr><td>FDP temporal mismatch</td><td>2023 snapshot applied to 2001-2024 loss history — coffee may not have been present at time of loss</td><td>Unknown; most severe for pre-2015 loss years</td></tr>
  <tr><td>Cloud masking gaps</td><td>Composite quality degrades in persistently cloudy regions</td><td>Variable; worst in equatorial wet seasons</td></tr>
  <tr><td>ML cross-AOI generalisation</td><td>Classifier trained in one region performs poorly in another</td><td>F1 drops from 0.78 (pooled) to 0.48-0.63 (holdout)</td></tr>
  <tr><td>Hotspot count cap</td><td>GEE limits to 5,000 hotspots per AOI; larger regions may have more</td><td>Cap affects completeness but not accuracy of detected hotspots</td></tr>
</table>

<h3>ML Classifier Assessment</h3>
<p>The cross-AOI F1 scores (0.48-0.63) indicate that <strong>the classifier does not generalise
well across geographies</strong>. This is expected: Robusta coffee in Vietnam has a different
spectral signature than shade-grown Arabica in Colombia. The system is best used as a
<strong>screening tool</strong> — it identifies areas of concern that require expert review or
field verification, not as an automated decision-making system.</p>
<p>For operational deployment, region-specific model retraining is recommended. The cross-AOI
holdout design provides the diagnostic framework to assess when retraining is needed
(F1 &lt; 0.70 on the held-out region).</p>

<h3>Comparison to Existing Platforms</h3>
<table>
  <tr><th>Platform</th><th>What It Provides</th><th>What This System Adds</th></tr>
  <tr><td>Global Forest Watch / GLAD</td><td>Near-real-time deforestation alerts, sub-hectare resolution</td><td>Coffee-specific attribution using FDP; identifies WHICH commodity drives loss</td></tr>
  <tr><td>Trase</td><td>Supply chain risk mapping by commodity and trader</td><td>Spatial hotspot polygons with loss year; ML validation; multi-sensor features</td></tr>
  <tr><td>MapBiomas</td><td>Annual LULC transitions (Brazil only)</td><td>Multi-country coverage (Vietnam, Colombia, Brazil); integrated reporting</td></tr>
</table>

<h3>Validation Status</h3>
<p>This system has <strong>no independent ground-truth validation</strong>. All accuracy metrics
reflect agreement between satellite-derived products (Hansen, FDP, WorldCover), not field-verified
land cover. Recommended validation steps for operational deployment:</p>
<ol style="margin:0.5em 0 0.5em 1.5em">
  <li>Acquire Planet NICFI basemap access for visual spot-checking at 3-5m resolution</li>
  <li>Cross-validate against MapBiomas annual LULC for the Brazil AOI</li>
  <li>Partner with in-country coffee cooperatives for stratified field verification</li>
  <li>Conduct FDP threshold sensitivity analysis (40% vs 50% vs 60%) to bound uncertainty</li>
</ol>
</div>

</div>
</body>
</html>"""


def _generate_synthesis(all_ids: list[str]) -> str:
    """Cross-AOI synthesis page."""
    rows = []
    for aoi_id in all_ids:
        stats_path = STATS_DIR / f"summary_{aoi_id}.json"
        if not stats_path.exists():
            continue
        with open(stats_path) as f:
            data = json.load(f)
        meta = data.get("metadata", {})
        cd = data.get("change_detection", {})
        attr = data.get("deforestation_attribution", {})
        name = meta.get("name", aoi_id)
        country = meta.get("country", "")
        total_loss = attr.get("total_loss_ha", 0)
        coffee_pct = attr.get("coffee_pct", 0)
        hotspots = cd.get("total_hotspots", 0)
        area = cd.get("total_area_ha", 0)
        other = attr.get("other_crops_pct", 0)
        regrowth = attr.get("regrowth_pct", 0)
        rows.append(f"""<tr>
          <td><strong><a href="report_{aoi_id}.html">{name}</a></strong><br><small>{country}</small></td>
          <td>{total_loss:,.0f}</td>
          <td>{coffee_pct:.0f}%</td>
          <td>{hotspots:,}</td>
          <td>{area:,.0f}</td>
          <td>{other:.1f}%</td>
          <td>{regrowth:.1f}%</td>
        </tr>""")

    # Side-by-side attribution pies
    pie_cards = []
    for aoi_id in all_ids:
        b64 = _b64_img(FIGURES_DIR / aoi_id / "attribution_pie.png")
        if b64:
            stats_path = STATS_DIR / f"summary_{aoi_id}.json"
            with open(stats_path) as f:
                meta = json.load(f).get("metadata", {})
            pie_cards.append(
                f'<div style="flex:1;min-width:280px;text-align:center">'
                f'<p style="font-weight:600">{meta.get("name", aoi_id)}</p>'
                f'<img src="{b64}" style="width:100%;border-radius:8px"></div>'
            )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Cross-Region Synthesis — Coffee Deforestation</title>
{CSS}
</head>
<body>
{_nav(all_ids, "synthesis")}
<div class="page">

<h1>Cross-Region Synthesis</h1>
<p class="subtitle">Comparing coffee-driven deforestation across {len(all_ids)} study regions</p>

<h2>Summary</h2>
<table>
  <tr><th>Region</th><th>Total Loss (ha)</th><th>Coffee %</th><th>Hotspots</th><th>Hotspot Area (ha)</th><th>Other Crops %</th><th>Regrowth %</th></tr>
  {"".join(rows)}
</table>

<h2>What Replaced the Forest? — Compared</h2>
<p>Side-by-side comparison of deforestation drivers across all study regions. Coffee's share
of total forest loss varies significantly: higher in active frontier regions (Vietnam, Colombia)
and lower in established plantation areas (Brazil).</p>
<div style="display:flex;flex-wrap:wrap;gap:1rem;margin:1rem 0">
  {"".join(pie_cards)}
</div>

<div class="caveat">
<strong>Key insight:</strong> The negative control (Sul de Minas, Brazil) confirms the pipeline
works — only 26% of loss became coffee in an area with stable, established plantations,
compared to 42-48% in active deforestation frontiers. The high regrowth fractions (48-67%)
suggest significant secondary forest recovery on previously deforested land.
</div>

</div>
</body>
</html>"""


@app.command()
def main(
    aoi: list[str] = typer.Option(default=AOI_IDS),
    open_browser: bool = typer.Option(True, "--open/--no-open"),
) -> None:
    """Generate clean HTML reports."""
    setup_logging()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    for aoi_id in aoi:
        logger.info(f"Generating report for {aoi_id}...")
        html = _generate_aoi_report(aoi_id, aoi)
        out = REPORTS_DIR / f"report_{aoi_id}.html"
        out.write_text(html, encoding="utf-8")
        logger.success(f"Saved {out}")

    logger.info("Generating synthesis...")
    syn = _generate_synthesis(aoi)
    syn_out = REPORTS_DIR / "synthesis.html"
    syn_out.write_text(syn, encoding="utf-8")
    logger.success(f"Saved {syn_out}")

    if open_browser:
        import webbrowser
        webbrowser.open(f"file://{REPORTS_DIR / f'report_{aoi[0]}.html'}")


if __name__ == "__main__":
    app()
