"""Generate a single interactive global map of coffee-linked deforestation with year slider.

What: Creates a world-scale folium map with GEE tile layers showing coffee-linked
deforestation accumulating year by year (2001-2024), with a slider and play button.
Why: Demonstrates the system works at global scale across all coffee-producing countries.
Assumes: GEE is authenticated.
Produces: outputs/maps/global_coffee_deforestation.html
"""

from __future__ import annotations

import sys
from pathlib import Path

import folium
from folium.plugins import MiniMap
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


COFFEE_COUNTRIES = [
    {"name": "Brazil", "lat": -14, "lon": -51, "region": "Americas"},
    {"name": "Colombia", "lat": 4, "lon": -74, "region": "Americas"},
    {"name": "Honduras", "lat": 15, "lon": -87, "region": "Americas"},
    {"name": "Peru", "lat": -10, "lon": -76, "region": "Americas"},
    {"name": "Mexico", "lat": 19, "lon": -99, "region": "Americas"},
    {"name": "Guatemala", "lat": 15, "lon": -90, "region": "Americas"},
    {"name": "Nicaragua", "lat": 13, "lon": -85, "region": "Americas"},
    {"name": "Costa Rica", "lat": 10, "lon": -84, "region": "Americas"},
    {"name": "Ethiopia", "lat": 8, "lon": 39, "region": "Africa"},
    {"name": "Uganda", "lat": 1, "lon": 32, "region": "Africa"},
    {"name": "Ivory Coast", "lat": 7, "lon": -6, "region": "Africa"},
    {"name": "Tanzania", "lat": -6, "lon": 35, "region": "Africa"},
    {"name": "Kenya", "lat": 0, "lon": 37, "region": "Africa"},
    {"name": "DR Congo", "lat": -3, "lon": 24, "region": "Africa"},
    {"name": "Cameroon", "lat": 6, "lon": 12, "region": "Africa"},
    {"name": "Vietnam", "lat": 12, "lon": 108, "region": "Asia"},
    {"name": "Indonesia", "lat": -2, "lon": 115, "region": "Asia"},
    {"name": "India", "lat": 12, "lon": 76, "region": "Asia"},
    {"name": "Laos", "lat": 18, "lon": 103, "region": "Asia"},
    {"name": "Philippines", "lat": 10, "lon": 124, "region": "Asia"},
    {"name": "Papua New Guinea", "lat": -6, "lon": 147, "region": "Asia"},
]


def generate_global_map() -> Path:
    """Create the global interactive map with per-year coffee deforestation slider."""
    import ee

    from coffee_deforestation.data.gee_client import init_gee
    from coffee_deforestation.viz.interactive import add_gee_tile_layer

    init_gee()

    logger.info("Building global coffee deforestation map with year slider...")

    m = folium.Map(
        location=[5, 20],
        zoom_start=3,
        tiles="CartoDB positron",
        attr="CartoDB",
    )

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri", name="Satellite Imagery", overlay=False,
    ).add_to(m)

    # --- GEE data ---
    hansen = ee.Image("UMD/hansen/global_forest_change_2024_v1_12")
    fdp = ee.Image(
        "projects/forestdatapartnership/assets/coffee/model_2025a/coffee_2023"
    ).select("probability")
    coffee_mask = fdp.gt(0.5)
    loss_year = hansen.select("lossyear")

    # --- Static context layers ---
    fdp_vis = fdp.visualize(min=0, max=1, palette=["#FFFFFF00", "#E8A33D80", "#6F4E37"])
    add_gee_tile_layer(m, fdp_vis, {}, "FDP Coffee Probability", opacity=0.7, shown=False)

    treecover_vis = hansen.select("treecover2000").visualize(
        min=0, max=100, palette=["#F5F1EB", "#90EE90", "#2D5016"])
    add_gee_tile_layer(m, treecover_vis, {}, "Forest Cover 2000", opacity=0.5, shown=False)
    logger.info("  Added context layers")

    # --- Per-year cumulative coffee deforestation tile layers ---
    year_tile_urls = {}
    for year in range(2001, 2025):
        year_code = year - 2000
        cumulative = loss_year.lte(year_code).And(loss_year.gt(0)).And(coffee_mask).selfMask()

        # Color by loss year within the cumulative set
        colored = loss_year.updateMask(
            loss_year.lte(year_code).And(loss_year.gt(0)).And(coffee_mask)
        ).selfMask().visualize(
            min=1, max=24,
            palette=["#FFEDA0", "#FEB24C", "#FC4E2A", "#BD0026", "#800026"],
        )

        try:
            map_id = colored.getMapId({})
            url = map_id["tile_fetcher"].url_format
            year_tile_urls[year] = url
        except Exception as e:
            logger.warning(f"  Failed to get tile for {year}: {e}")

    logger.info(f"  Generated {len(year_tile_urls)} year tile layers")

    # --- Country markers ---
    country_group = folium.FeatureGroup(name="Coffee Countries")
    for c in COFFEE_COUNTRIES:
        folium.CircleMarker(
            location=[c["lat"], c["lon"]], radius=5,
            color="#6F4E37", fill=True, fill_color="#6F4E37", fill_opacity=0.8,
            tooltip=c["name"],
        ).add_to(country_group)
    country_group.add_to(m)

    # --- Study region boxes ---
    from coffee_deforestation.config import load_aois
    aoi_group = folium.FeatureGroup(name="Study Regions")
    for aoi_name, aoi in load_aois().items():
        folium.Polygon(
            locations=[
                [aoi.bbox.south, aoi.bbox.west], [aoi.bbox.south, aoi.bbox.east],
                [aoi.bbox.north, aoi.bbox.east], [aoi.bbox.north, aoi.bbox.west],
                [aoi.bbox.south, aoi.bbox.west],
            ],
            color="#C1292E", weight=3, fill=True, fill_color="#C1292E", fill_opacity=0.1,
            tooltip=f"{aoi.name} — study region",
        ).add_to(aoi_group)
    aoi_group.add_to(m)

    # Controls (BEFORE the slider JS so layers exist)
    folium.LayerControl(collapsed=True).add_to(m)
    MiniMap(toggle_display=True, position="bottomleft", zoom_level_offset=-6).add_to(m)

    # --- JavaScript year slider with GEE tile switching ---
    import json
    urls_json = json.dumps(year_tile_urls)
    map_var = m.get_name()

    slider_js = f"""
    <script>
    (function _initGlobalSlider() {{
        var map = null;
        for (var k in window) {{
            try {{
                if (window[k] && window[k]._leaflet_id && window[k].getCenter) {{
                    map = window[k];
                    break;
                }}
            }} catch(e) {{}}
        }}
        if (!map) {{ setTimeout(_initGlobalSlider, 200); return; }}

        var yearUrls = {urls_json};
        var currentLayer = null;
        var currentYear = 2024;
        var isPlaying = false;
        var playInterval = null;

        function updateYear(year) {{
            currentYear = year;
            if (currentLayer) map.removeLayer(currentLayer);

            var url = yearUrls[String(year)];
            if (url) {{
                currentLayer = L.tileLayer(url, {{
                    attribution: 'Google Earth Engine',
                    opacity: 0.85,
                    maxZoom: 18,
                }}).addTo(map);
            }}

            document.getElementById('gYearLabel').textContent = '2001 — ' + year;
            document.getElementById('gYearSlider').value = year;
        }}

        // Build slider UI
        var div = document.createElement('div');
        div.innerHTML =
            '<div style="position:fixed;bottom:20px;left:50%;transform:translateX(-50%);z-index:1000;' +
            'background:rgba(255,255,255,0.97);padding:14px 24px;border-radius:12px;' +
            'box-shadow:0 4px 20px rgba(0,0,0,0.3);font-family:sans-serif;min-width:480px;max-width:90vw">' +
            '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">' +
            '<span id="gYearLabel" style="font-size:20px;font-weight:700;color:#6F4E37">2001 — 2024</span>' +
            '</div>' +
            '<input type="range" id="gYearSlider" min="2001" max="2024" value="2024" step="1" ' +
            'style="width:100%;height:8px;accent-color:#C1292E;cursor:pointer">' +
            '<div style="display:flex;justify-content:space-between;font-size:10px;color:#888;margin-top:2px">' +
            '<span>2001</span><span>2005</span><span>2010</span><span>2015</span><span>2020</span><span>2024</span></div>' +
            '<div style="display:flex;gap:8px;margin-top:8px;align-items:center">' +
            '<button id="gPlayBtn" style="background:#6F4E37;color:white;border:none;padding:5px 12px;' +
            'border-radius:6px;cursor:pointer;font-size:12px;font-weight:600">&#9654; Play</button>' +
            '<span style="font-size:11px;color:#888">Cumulative coffee-linked deforestation (Hansen \\u2229 FDP)</span>' +
            '<div style="flex:1"></div>' +
            '<div style="display:flex;align-items:center;gap:3px;font-size:10px;color:#666">' +
            '<div style="width:60px;height:8px;background:linear-gradient(to right,#FFEDA0,#FEB24C,#FC4E2A,#800026);border-radius:3px"></div>' +
            '<span>Old\\u2192Recent</span></div></div></div>';
        document.body.appendChild(div);

        document.getElementById('gYearSlider').addEventListener('input', function(e) {{
            updateYear(parseInt(e.target.value));
        }});

        document.getElementById('gPlayBtn').addEventListener('click', function() {{
            if (isPlaying) {{
                clearInterval(playInterval);
                isPlaying = false;
                this.innerHTML = '&#9654; Play';
                return;
            }}
            isPlaying = true;
            this.innerHTML = '&#9724; Stop';
            var yr = 2001;
            updateYear(yr);
            playInterval = setInterval(function() {{
                yr++;
                if (yr > 2024) {{
                    clearInterval(playInterval);
                    isPlaying = false;
                    document.getElementById('gPlayBtn').innerHTML = '&#9654; Play';
                    return;
                }}
                updateYear(yr);
            }}, 500);
        }});

        // Initial render
        updateYear(2024);
    }})();
    </script>
    """
    m.get_root().html.add_child(folium.Element(slider_js))

    # Legend
    legend_html = """
    <div style="position:fixed; top:10px; right:10px; z-index:1000;
                background:white; padding:12px 16px; border-radius:10px;
                box-shadow:0 2px 8px rgba(0,0,0,0.25); font-family:sans-serif;
                font-size:12px; max-width:200px;">
        <b style="font-size:13px;color:#6F4E37">Coffee & Deforestation</b>
        <div style="margin-top:6px">
            <div style="display:flex;align-items:center;gap:5px;margin:3px 0">
                <div style="width:50px;height:8px;background:linear-gradient(to right,#FFEDA0,#FC4E2A,#800026);border-radius:2px"></div>
                <span>Loss year (old→recent)</span>
            </div>
            <div style="display:flex;align-items:center;gap:5px;margin:3px 0">
                <div style="width:12px;height:12px;background:#6F4E37;border-radius:50%"></div>
                <span>Coffee country</span>
            </div>
            <div style="display:flex;align-items:center;gap:5px;margin:3px 0">
                <div style="width:12px;height:12px;border:2px solid #C1292E;border-radius:2px"></div>
                <span>Study region</span>
            </div>
        </div>
        <div style="margin-top:6px;font-size:10px;color:#888">
            Zoom in for 10m resolution
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # Title
    title_html = """
    <div style="position:fixed; top:10px; left:50%; transform:translateX(-50%); z-index:1000;
                background:rgba(111,78,55,0.95); padding:8px 20px; border-radius:8px;
                box-shadow:0 2px 8px rgba(0,0,0,0.3); font-family:sans-serif;">
        <span style="color:white;font-size:15px;font-weight:700">Global Coffee-Linked Deforestation</span>
        <span style="color:#E8A33D;font-size:11px;margin-left:10px">Hansen GFC × FDP Coffee</span>
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    # Save
    output_dir = Path(REPO_ROOT) / "outputs" / "maps"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "global_coffee_deforestation.html"
    m.save(str(output_path))
    logger.info(f"Saved global map: {output_path}")
    return output_path


if __name__ == "__main__":
    path = generate_global_map()
    print(f"\nGlobal map saved: {path}")
    import webbrowser
    webbrowser.open(f"file://{path}")
