"""Build global coffee deforestation map from pre-downloaded points."""

import json
import sys
from pathlib import Path

import folium
from folium.plugins import MiniMap

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

POINTS_PATH = REPO_ROOT / "outputs" / "cache" / "global_coffee_points_150k.json"
OUTPUT_PATH = REPO_ROOT / "outputs" / "maps" / "global_coffee_deforestation.html"


def build():
    with open(POINTS_PATH) as f:
        points = json.load(f)
    print(f"Loaded {len(points)} points")

    m = folium.Map(location=[5, 20], zoom_start=3, tiles="CartoDB positron")
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri", name="Satellite", overlay=False,
    ).add_to(m)

    folium.LayerControl(collapsed=True).add_to(m)
    MiniMap(toggle_display=True, position="bottomleft", zoom_level_offset=-6).add_to(m)

    map_name = m.get_name()
    points_json = json.dumps(points)

    # Build the JS as a plain string (no f-string to avoid brace issues)
    js_parts = []
    js_parts.append("<script>")
    js_parts.append("window.addEventListener('load', function() {")
    js_parts.append(f"  console.log('load event fired, map_name={map_name}, exists:', typeof {map_name});")
    js_parts.append(f"  var map = {map_name};")
    js_parts.append(f"  var allPoints = {points_json};")
    js_parts.append("""
  var markersLayer = null;
  var currentYear = 2024;
  var isPlaying = false;
  var playInterval = null;

  function yc(y) {
    var t = Math.max(0, Math.min(1, (y - 2001) / 23));
    return 'rgb(' + Math.round(255-t*127) + ',' + Math.round(237-t*237) + ',' + Math.round(160-t*122) + ')';
  }

  // Use canvas renderer for performance with 1M+ points
  var canvasRenderer = L.canvas({ padding: 0.5 });

  function update(year) {
    currentYear = year;
    if (markersLayer) map.removeLayer(markersLayer);
    var filtered = allPoints.filter(function(p) { return p.year <= year; });
    markersLayer = L.layerGroup();
    for (var i = 0; i < filtered.length; i++) {
      var p = filtered[i];
      L.circleMarker([p.lat, p.lon], {
        radius: 3,
        color: yc(p.year),
        fillColor: yc(p.year),
        fillOpacity: 0.8,
        weight: 0,
        renderer: canvasRenderer
      }).addTo(markersLayer);
    }
    markersLayer.addTo(map);
    var el = document.getElementById('gl');
    if (el) el.textContent = '2001 \\u2014 ' + year;
    var ec = document.getElementById('gc');
    if (ec) ec.textContent = filtered.length.toLocaleString() + ' detections';
  }

  var d = document.createElement('div');
  d.innerHTML = '<div style="position:fixed;bottom:20px;left:50%;transform:translateX(-50%);z-index:1000;background:rgba(255,255,255,0.97);padding:14px 24px;border-radius:12px;box-shadow:0 4px 20px rgba(0,0,0,0.3);font-family:sans-serif;min-width:480px;max-width:90vw">' +
    '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">' +
    '<span id="gl" style="font-size:20px;font-weight:700;color:#6F4E37">2001 \\u2014 2024</span>' +
    '<span id="gc" style="font-size:14px;font-weight:600;color:#C1292E"></span></div>' +
    '<input type="range" id="gs" min="2001" max="2024" value="2024" step="1" style="width:100%;accent-color:#C1292E;cursor:pointer">' +
    '<div style="display:flex;justify-content:space-between;font-size:10px;color:#888;margin-top:2px"><span>2001</span><span>2010</span><span>2020</span><span>2024</span></div>' +
    '<div style="display:flex;gap:8px;margin-top:8px;align-items:center">' +
    '<button id="gp" style="background:#6F4E37;color:white;border:none;padding:5px 12px;border-radius:6px;cursor:pointer;font-weight:600">\\u25b6 Play</button>' +
    '<span style="font-size:11px;color:#888">Global coffee-linked deforestation (~5km)</span>' +
    '<div style="flex:1"></div>' +
    '<div style="display:flex;align-items:center;gap:3px;font-size:10px;color:#666">' +
    '<div style="width:60px;height:8px;background:linear-gradient(to right,#FFEDA0,#FEB24C,#FC4E2A,#800026);border-radius:3px"></div>' +
    '<span>Old\\u2192Recent</span></div></div></div>';
  document.body.appendChild(d);

  document.getElementById('gs').addEventListener('input', function(e) { update(parseInt(e.target.value)); });
  document.getElementById('gp').addEventListener('click', function() {
    if (isPlaying) { clearInterval(playInterval); isPlaying=false; this.innerHTML='\\u25b6 Play'; return; }
    isPlaying=true; this.innerHTML='\\u25a0 Stop';
    var yr=2001; update(yr);
    playInterval = setInterval(function() {
      yr++; if(yr>2024){clearInterval(playInterval);isPlaying=false;document.getElementById('gp').innerHTML='\\u25b6 Play';return;}
      document.getElementById('gs').value=yr; update(yr);
    }, 400);
  });

  update(2024);
  console.log('Global slider initialized: ' + allPoints.length + ' points');
""")
    js_parts.append("});")
    js_parts.append("</script>")

    js = "\n".join(js_parts)

    # Title
    title = """<div style="position:fixed;top:10px;left:50%;transform:translateX(-50%);z-index:1000;
        background:rgba(111,78,55,0.95);padding:8px 20px;border-radius:8px;
        box-shadow:0 2px 8px rgba(0,0,0,0.3);font-family:sans-serif">
        <span style="color:white;font-size:15px;font-weight:700">Global Coffee-Linked Deforestation</span>
        <span style="color:#E8A33D;font-size:11px;margin-left:10px">Hansen GFC &times; FDP Coffee | 2001-2024</span>
    </div>"""

    m.get_root().html.add_child(folium.Element(title))
    m.get_root().html.add_child(folium.Element(js))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(OUTPUT_PATH))
    print(f"Saved: {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size // 1024} KB)")
    return OUTPUT_PATH


if __name__ == "__main__":
    path = build()
    import webbrowser
    webbrowser.open(f"file://{path}")
