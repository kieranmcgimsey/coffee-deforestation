"""Interactive folium maps with GEE tile layers and temporal slider.

What: Creates zoomable HTML maps with satellite imagery from GEE rendered as
tile layers at full resolution, and hotspot polygons with an interactive
year slider for temporal exploration.
Why: Interactive maps at native resolution are far more compelling than static
pixel grids. The time slider shows deforestation spreading year by year.
Assumes: GEE is authenticated. GeoJSON hotspots are available.
Produces: Self-contained HTML maps embeddable in reports.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import ee

import folium
from folium.plugins import MiniMap
from loguru import logger

from coffee_deforestation.config import AOIConfig, PROJECT_ROOT
from coffee_deforestation.viz.theme import COLORS


def add_gee_tile_layer(
    m: folium.Map,
    image: ee.Image,
    vis_params: dict,
    name: str,
    opacity: float = 0.7,
    shown: bool = True,
) -> None:
    """Add a GEE image as a zoomable tile layer to a folium map."""
    try:
        map_id_dict = image.getMapId(vis_params)  # type: ignore[union-attr]
        tiles_url = map_id_dict["tile_fetcher"].url_format
        folium.TileLayer(
            tiles=tiles_url,
            attr="Google Earth Engine",
            name=name,
            overlay=True,
            control=True,
            opacity=opacity,
            show=shown,
        ).add_to(m)
        logger.debug(f"Added GEE tile layer: {name}")
    except Exception as e:
        logger.warning(f"Failed to add GEE tile layer '{name}': {e}")


def create_rich_map(
    aoi: AOIConfig,
    hotspot_geojson_path: Path | None = None,
    s2_composite: ee.Image | None = None,
    ndvi: ee.Image | None = None,
    coffee_prob: ee.Image | None = None,
    hansen_loss: ee.Image | None = None,
    zoom_start: int | None = None,
) -> folium.Map:
    """Create a rich interactive map with GEE tile layers and year slider."""
    center_lat = (aoi.bbox.south + aoi.bbox.north) / 2
    center_lon = (aoi.bbox.west + aoi.bbox.east) / 2

    if zoom_start is None:
        extent_deg = max(aoi.bbox.width_deg, aoi.bbox.height_deg)
        if extent_deg > 2:
            zoom_start = 8
        elif extent_deg > 1:
            zoom_start = 9
        elif extent_deg > 0.5:
            zoom_start = 10
        else:
            zoom_start = 11

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom_start,
        tiles="CartoDB positron",
        attr="CartoDB",
    )

    # Esri World Imagery
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Satellite Imagery",
        overlay=False,
    ).add_to(m)

    # --- GEE tile layers ---
    if s2_composite is not None:
        add_gee_tile_layer(m, s2_composite,
            vis_params={"bands": ["B4", "B3", "B2"], "min": 0, "max": 0.3},
            name="Sentinel-2 RGB (2024)", shown=False)

    if ndvi is not None:
        ndvi_vis = ndvi.visualize(min=0, max=0.8,
            palette=["#8B4513", "#F4D35E", "#90EE90", "#2D5016"])
        add_gee_tile_layer(m, ndvi_vis, vis_params={},
            name="NDVI", opacity=0.7, shown=False)

    if coffee_prob is not None:
        coffee_vis = coffee_prob.visualize(min=0, max=1,
            palette=["#F5F1EB", "#E8A33D", "#6F4E37"])
        add_gee_tile_layer(m, coffee_vis, vis_params={},
            name="FDP Coffee Probability", opacity=0.6, shown=False)

    if hansen_loss is not None:
        loss_vis = hansen_loss.select("lossyear").selfMask().visualize(
            min=1, max=23, palette=["#FFEDA0", "#FEB24C", "#FC4E2A", "#BD0026", "#800026"])
        add_gee_tile_layer(m, loss_vis, vis_params={},
            name="Hansen Forest Loss Year", opacity=0.6, shown=False)

    # --- AOI boundary ---
    bbox_coords = [
        [aoi.bbox.south, aoi.bbox.west], [aoi.bbox.south, aoi.bbox.east],
        [aoi.bbox.north, aoi.bbox.east], [aoi.bbox.north, aoi.bbox.west],
        [aoi.bbox.south, aoi.bbox.west],
    ]
    folium.Polygon(locations=bbox_coords, color="#333", weight=2, fill=False,
                   dash_array="5 5", popup=f"{aoi.name} ({aoi.country})").add_to(m)

    # --- Hotspot polygons with interactive year slider ---
    if hotspot_geojson_path and hotspot_geojson_path.exists():
        with open(hotspot_geojson_path) as f:
            hotspot_data = json.load(f)
        _add_hotspots_with_slider(m, hotspot_data, aoi)

    # Controls
    folium.LayerControl(collapsed=False).add_to(m)
    MiniMap(toggle_display=True, position="bottomleft").add_to(m)

    return m


def _add_hotspots_with_slider(m: folium.Map, geojson: dict, aoi: AOIConfig) -> None:
    """Add hotspot GeoJSON with a custom JavaScript year slider.

    Features:
    - Dual-mode slider: single year or cumulative (2001 to selected year)
    - Hotspots colored by loss year (yellow=old, red=recent)
    - Live count + area display updates as slider moves
    - Click for popup with hotspot details
    - Smooth animation playback button
    """
    geojson_str = json.dumps(geojson)
    # Get the actual JS variable name folium uses for this map
    map_var = m.get_name()

    slider_js = f"""
    <script>
    (function _initSlider() {{
        // Find the Leaflet map — poll until it exists
        var map = null;
        for (var k in window) {{
            try {{
                if (window[k] && window[k]._leaflet_id && window[k].getCenter) {{
                    map = window[k];
                    break;
                }}
            }} catch(e) {{}}
        }}
        if (!map) {{
            setTimeout(_initSlider, 200);
            return;
        }}
        var geojsonData = {geojson_str};
        var hotspotsLayer = null;
        var currentMode = 'cumulative';
        var currentYear = 2024;
        var isPlaying = false;
        var playInterval = null;

        function yearColor(year) {{
            if (!year) return '#888';
            var t = Math.max(0, Math.min(1, (year - 2001) / 23));
            var r = Math.round(255 - t * 127);
            var g = Math.round(237 - t * 237);
            var b = Math.round(160 - t * 122);
            return 'rgb(' + r + ',' + g + ',' + b + ')';
        }}

        function fmtNum(n) {{
            return n.toString().replace(/\\B(?=(\\d{{3}})+(?!\\d))/g, ',');
        }}

        function updateHotspots() {{
            if (hotspotsLayer) map.removeLayer(hotspotsLayer);

            var filtered = geojsonData.features.filter(function(f) {{
                var ly = f.properties.loss_year;
                if (!ly) return false;
                if (currentMode === 'cumulative') return ly <= currentYear;
                return ly === currentYear;
            }});

            var totalArea = 0;
            filtered.forEach(function(f) {{ totalArea += f.properties.area_ha || 0; }});

            var post2020Count = 0;
            var post2020Area = 0;
            filtered.forEach(function(f) {{
                if (f.properties.loss_year && f.properties.loss_year > 2020) {{
                    post2020Count++;
                    post2020Area += f.properties.area_ha || 0;
                }}
            }});

            hotspotsLayer = L.geoJSON({{type: 'FeatureCollection', features: filtered}}, {{
                style: function(feature) {{
                    var ly = feature.properties.loss_year;
                    var isEUDR = ly && ly > 2020;
                    return {{
                        fillColor: yearColor(ly),
                        color: isEUDR ? '#FF0000' : yearColor(ly),
                        weight: isEUDR ? 2.5 : 1,
                        fillOpacity: isEUDR ? 0.7 : 0.5,
                        opacity: 0.8
                    }};
                }},
                onEachFeature: function(feature, layer) {{
                    var p = feature.properties;
                    var eudrFlag = (p.loss_year && p.loss_year > 2020) ?
                        '<br><span style="color:#FF0000;font-weight:bold">EUDR RISK (post-2020)</span>' : '';
                    layer.bindPopup(
                        '<div style="font-family:sans-serif">' +
                        '<b style="font-size:14px">Hotspot #' + p.rank + '</b>' + eudrFlag + '<br>' +
                        '<b>Area:</b> ' + (p.area_ha || 0).toFixed(1) + ' ha<br>' +
                        '<b>Loss Year:</b> ' + (p.loss_year || '?') + '<br>' +
                        '<b>ID:</b> ' + (p.hotspot_id || '') +
                        '</div>'
                    );
                    layer.bindTooltip('#' + p.rank + ': ' + (p.area_ha||0).toFixed(0) + 'ha (' + (p.loss_year||'?') + ')');
                }}
            }}).addTo(map);

            var label = currentMode === 'cumulative' ? '2001 — ' + currentYear : '' + currentYear;
            document.getElementById('yearLabel').textContent = label;
            document.getElementById('hotspotCount').textContent = fmtNum(filtered.length) + ' hotspots';
            document.getElementById('hotspotArea').textContent = fmtNum(Math.round(totalArea)) + ' ha';
            var eudrEl = document.getElementById('eudrCount');
            if (eudrEl) eudrEl.textContent = post2020Count > 0 ? post2020Count + ' post-2020 (EUDR risk)' : '';
        }}

        // Build slider UI
        var sliderDiv = document.createElement('div');
        sliderDiv.innerHTML = '<div style="position:fixed;bottom:20px;left:50%;transform:translateX(-50%);z-index:1000;background:white;padding:16px 24px;border-radius:12px;box-shadow:0 4px 20px rgba(0,0,0,0.25);font-family:sans-serif;min-width:500px;max-width:90vw">' +
            '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">' +
            '<span id="yearLabel" style="font-size:20px;font-weight:700;color:#6F4E37">2001 — 2024</span>' +
            '<div><span id="hotspotCount" style="font-size:14px;font-weight:600;color:#C1292E"></span>' +
            '<span style="color:#888;font-size:12px"> | </span>' +
            '<span id="hotspotArea" style="font-size:14px;font-weight:600;color:#2D5016"></span>' +
            '<br><span id="eudrCount" style="font-size:11px;font-weight:600;color:#FF0000"></span></div></div>' +
            '<input type="range" id="yearSlider" min="2001" max="2024" value="2024" step="1" style="width:100%;height:8px;accent-color:#C1292E;cursor:pointer">' +
            '<div style="display:flex;justify-content:space-between;font-size:10px;color:#888;margin-top:2px"><span>2001</span><span>2005</span><span>2010</span><span>2015</span><span>2020</span><span>2024</span></div>' +
            '<div style="display:flex;gap:8px;margin-top:10px;align-items:center">' +
            '<button id="playBtn" style="background:#6F4E37;color:white;border:none;padding:6px 14px;border-radius:6px;cursor:pointer;font-size:12px;font-weight:600">&#9654; Play</button>' +
            '<label style="font-size:12px;cursor:pointer"><input type="radio" name="mode" value="cumulative" checked> Cumulative</label>' +
            '<label style="font-size:12px;cursor:pointer"><input type="radio" name="mode" value="single"> Single Year</label>' +
            '<div style="flex:1"></div>' +
            '<div style="display:flex;align-items:center;gap:4px;font-size:10px;color:#666"><div style="width:80px;height:10px;background:linear-gradient(to right,#FFEDA0,#FEB24C,#FC4E2A,#800026);border-radius:3px"></div><span>Old &#8594; Recent</span></div>' +
            '</div></div>';
        document.body.appendChild(sliderDiv);

        document.getElementById('yearSlider').addEventListener('input', function(e) {{
            currentYear = parseInt(e.target.value);
            updateHotspots();
        }});

        document.querySelectorAll('input[name="mode"]').forEach(function(radio) {{
            radio.addEventListener('change', function(e) {{
                currentMode = e.target.value;
                updateHotspots();
            }});
        }});

        document.getElementById('playBtn').addEventListener('click', function() {{
            if (isPlaying) {{
                clearInterval(playInterval);
                isPlaying = false;
                this.innerHTML = '&#9654; Play';
                return;
            }}
            isPlaying = true;
            this.innerHTML = '&#9724; Stop';
            var slider = document.getElementById('yearSlider');
            slider.value = 2001;
            currentYear = 2001;
            updateHotspots();
            playInterval = setInterval(function() {{
                currentYear++;
                if (currentYear > 2024) {{
                    clearInterval(playInterval);
                    isPlaying = false;
                    document.getElementById('playBtn').innerHTML = '&#9654; Play';
                    return;
                }}
                slider.value = currentYear;
                updateHotspots();
            }}, 400);
        }});

        // Initial render
        updateHotspots();
    }})();
    </script>
    """

    # Inject into the map's own script section so it runs AFTER the map is initialized.
    # folium's get_root().script renders inside the same <script> block as the map.
    from branca.element import Element

    el = Element(slider_js)
    m.get_root().html.add_child(el)


# Backward-compatible function
def create_aoi_map(aoi: AOIConfig, hotspot_geojson_path: Path | None = None) -> folium.Map:
    """Create an interactive folium map for an AOI (legacy interface)."""
    return create_rich_map(aoi, hotspot_geojson_path)


def save_map(m: folium.Map, aoi: AOIConfig, output_dir: Path | None = None) -> Path:
    """Save a folium map to HTML."""
    if output_dir is None:
        output_dir = PROJECT_ROOT / "outputs" / "maps"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"map_{aoi.id}.html"
    m.save(str(output_path))
    logger.info(f"Saved map to {output_path}")
    return output_path
