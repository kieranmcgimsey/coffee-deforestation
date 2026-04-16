"""Visual theme: locked palette, matplotlib rcParams, and shared helpers.

What: Defines the single visual identity for all figures and maps.
Why: Visual consistency across all outputs signals professional quality.
Assumes: matplotlib is available.
Produces: Color constants, matplotlib configuration, and helper functions.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from coffee_deforestation.config import BBox

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

# --- Locked palette ---

COLORS = {
    "coffee": "#6F4E37",
    "forest_stable": "#2D5016",
    "forest_loss": "#C1292E",
    "coffee_on_former_forest": "#E8A33D",
    "non_coffee_cropland": "#F4D35E",
    "built_bare": "#9E9E9E",
    "water": "#4A7BA6",
    "background": "#F5F1EB",
}

# Classification colormap (indexed by class code)
CLASS_COLORS = [
    COLORS["coffee"],             # 0: coffee
    COLORS["forest_stable"],      # 1: forest
    COLORS["non_coffee_cropland"],# 2: non-coffee cropland
    COLORS["built_bare"],         # 3: built/bare
    COLORS["water"],              # 4: water
]

CLASS_NAMES = ["Coffee", "Forest", "Cropland (non-coffee)", "Built/Bare", "Water"]

CLASS_CMAP = ListedColormap(CLASS_COLORS, name="coffee_classes")

# Continuous colormaps
NDVI_CMAP = LinearSegmentedColormap.from_list(
    "ndvi", ["#8B4513", "#F4D35E", "#90EE90", "#2D5016"], N=256
)

COFFEE_PROB_CMAP = LinearSegmentedColormap.from_list(
    "coffee_prob", [COLORS["background"], COLORS["coffee"]], N=256
)

LOSS_YEAR_CMAP = LinearSegmentedColormap.from_list(
    "loss_year", ["#FFEDA0", "#C1292E"], N=256
)


def apply_theme() -> None:
    """Apply the locked visual theme to matplotlib."""
    plt.style.use("seaborn-v0_8-whitegrid")

    mpl.rcParams.update({
        "figure.facecolor": COLORS["background"],
        "axes.facecolor": "white",
        "font.family": ["Helvetica Neue", "Arial", "sans-serif"],
        "font.size": 10,
        "axes.titlesize": 14,
        "axes.labelsize": 11,
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.facecolor": COLORS["background"],
    })


def add_attribution(ax: plt.Axes, extra: str = "") -> None:
    """Add standard attribution footer to a figure axis."""
    date = datetime.now().strftime("%Y-%m-%d")
    text = f"Source: Sentinel-1/-2, Hansen GFC, FDP. Generated {date}."
    if extra:
        text = f"{text} {extra}"
    ax.annotate(
        text,
        xy=(0.5, -0.08),
        xycoords="axes fraction",
        ha="center",
        fontsize=7,
        color="#888888",
    )


def figure_with_title(
    title: str,
    subtitle: str = "",
    figsize: tuple[float, float] = (10, 8),
) -> tuple[plt.Figure, plt.Axes]:
    """Create a figure with a styled title and optional subtitle."""
    apply_theme()
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    if subtitle:
        ax.text(
            0.5, 1.02, subtitle,
            transform=ax.transAxes, ha="center", fontsize=9, color="#666666",
        )
    return fig, ax


def save_figure(fig: plt.Figure, path: str | None = None, **kwargs) -> None:  # type: ignore[type-arg]
    """Save a figure and close it to free memory."""
    if path:
        from pathlib import Path

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, **kwargs)
    plt.close(fig)


# --- Replacement land cover colors (for deforestation attribution) ---

REPLACEMENT_COLORS = {
    "coffee": COLORS["coffee"],
    "other_crops": "#F4D35E",
    "built_industrial": "#9E9E9E",
    "bare_degraded": "#D4A574",
    "water": COLORS["water"],
    "regrowth": "#90EE90",
}

REPLACEMENT_NAMES = {
    "coffee": "Coffee",
    "other_crops": "Other Crops",
    "built_industrial": "Built / Industrial",
    "bare_degraded": "Bare / Degraded",
    "water": "Water",
    "regrowth": "Forest Regrowth",
}

# NDVI change diverging colormap
NDVI_CHANGE_CMAP = LinearSegmentedColormap.from_list(
    "ndvi_change", ["#C1292E", "#FFFFFF", "#2D5016"], N=256
)


def format_coordinate_axes(ax: plt.Axes, bbox: BBox) -> None:
    """Set up longitude/latitude tick labels and axis formatting on a map axes.

    Args:
        ax: matplotlib axes
        bbox: BBox-like object with west, south, east, north attributes
    """
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Generate ~5 tick positions
    n_ticks = 5
    x_ticks = np.linspace(0, 1, n_ticks)
    y_ticks = np.linspace(0, 1, n_ticks)

    lons = np.linspace(bbox.west, bbox.east, n_ticks)  # type: ignore[union-attr]
    lats = np.linspace(bbox.south, bbox.north, n_ticks)  # type: ignore[union-attr]

    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{lon:.2f}" for lon in lons], fontsize=7)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{lat:.2f}" for lat in lats], fontsize=7)
    ax.set_xlabel("Longitude", fontsize=8)
    ax.set_ylabel("Latitude", fontsize=8)
    ax.tick_params(axis="both", which="both", length=3, width=0.5)


def add_scale_bar(ax: plt.Axes, bbox: BBox, length_km: float = 10) -> None:
    """Add a scale bar to a map figure.

    Calculates the correct bar width from the bbox extent and target length.
    """
    # Approximate degrees per km at this latitude
    mid_lat = (bbox.south + bbox.north) / 2  # type: ignore[union-attr]
    deg_per_km_lon = 1 / (111.32 * np.cos(np.radians(mid_lat)))
    bbox_width_km = (bbox.east - bbox.west) / deg_per_km_lon  # type: ignore[union-attr]

    bar_width_frac = length_km / bbox_width_km

    # Position at bottom-right
    x0 = 0.95 - bar_width_frac
    y0 = 0.04

    ax.plot([x0, x0 + bar_width_frac], [y0, y0], "k-", linewidth=2,
            transform=ax.transAxes, clip_on=False)
    ax.plot([x0, x0], [y0 - 0.005, y0 + 0.005], "k-", linewidth=1.5,
            transform=ax.transAxes, clip_on=False)
    ax.plot([x0 + bar_width_frac, x0 + bar_width_frac], [y0 - 0.005, y0 + 0.005],
            "k-", linewidth=1.5, transform=ax.transAxes, clip_on=False)
    ax.text(x0 + bar_width_frac / 2, y0 + 0.015, f"{length_km:.0f} km",
            ha="center", va="bottom", fontsize=7, fontweight="bold",
            transform=ax.transAxes)
