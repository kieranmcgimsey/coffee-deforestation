"""Static diagnostic figures using matplotlib.

What: Generates diagnostic PNG figures for each pipeline stage — S2 RGB composites,
cloud masks, NDVI, S1 VV, feature correlations, FDP coffee, Hansen loss, hotspots.
Why: Visual verification at every stage catches errors that metrics miss.
Assumes: Raster data is available as numpy arrays or GeoTIFFs. Theme is applied.
Produces: PNG figures saved to outputs/figures/.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from coffee_deforestation.config import BBox
import numpy as np
from loguru import logger

from coffee_deforestation.config import AOIConfig, PROJECT_ROOT
from coffee_deforestation.viz.theme import (
    COFFEE_PROB_CMAP,
    COLORS,
    LOSS_YEAR_CMAP,
    NDVI_CMAP,
    add_attribution,
    apply_theme,
    figure_with_title,
    save_figure,
)


def _output_dir(aoi_id: str) -> Path:
    path = PROJECT_ROOT / "outputs" / "figures" / aoi_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_rgb_composite(
    red: np.ndarray,
    green: np.ndarray,
    blue: np.ndarray,
    aoi: AOIConfig,
    year: int,
    output_path: str | None = None,
) -> str:
    """Plot an S2 true-color RGB composite."""
    apply_theme()
    fig, ax = figure_with_title(
        f"Sentinel-2 RGB Composite",
        f"{aoi.name} ({aoi.country}) — {year} dry season",
    )

    # Stack and clip to 0-1
    rgb = np.dstack([
        np.clip(red, 0, 0.3) / 0.3,
        np.clip(green, 0, 0.3) / 0.3,
        np.clip(blue, 0, 0.3) / 0.3,
    ])
    ax.imshow(rgb)
    ax.axis("off")
    add_attribution(ax)

    if output_path is None:
        output_path = str(_output_dir(aoi.id) / f"s2_rgb_{year}.png")
    save_figure(fig, output_path)
    logger.debug(f"Saved RGB composite: {output_path}")
    return output_path


def plot_ndvi(
    ndvi: np.ndarray,
    aoi: AOIConfig,
    year: int,
    output_path: str | None = None,
) -> str:
    """Plot NDVI map."""
    fig, ax = figure_with_title(
        "NDVI",
        f"{aoi.name} ({aoi.country}) — {year} dry season",
    )
    im = ax.imshow(ndvi, cmap=NDVI_CMAP, vmin=-0.2, vmax=0.9)
    plt.colorbar(im, ax=ax, label="NDVI", shrink=0.8)
    ax.axis("off")
    add_attribution(ax)

    if output_path is None:
        output_path = str(_output_dir(aoi.id) / f"ndvi_{year}.png")
    save_figure(fig, output_path)
    logger.debug(f"Saved NDVI: {output_path}")
    return output_path


def plot_s1_vv(
    vv: np.ndarray,
    aoi: AOIConfig,
    year: int,
    output_path: str | None = None,
) -> str:
    """Plot S1 VV backscatter."""
    fig, ax = figure_with_title(
        "Sentinel-1 VV Backscatter",
        f"{aoi.name} ({aoi.country}) — {year} dry season",
    )
    im = ax.imshow(vv, cmap="gray", vmin=-25, vmax=0)
    plt.colorbar(im, ax=ax, label="VV (dB)", shrink=0.8)
    ax.axis("off")
    add_attribution(ax)

    if output_path is None:
        output_path = str(_output_dir(aoi.id) / f"s1_vv_{year}.png")
    save_figure(fig, output_path)
    logger.debug(f"Saved S1 VV: {output_path}")
    return output_path


def plot_coffee_probability(
    coffee_prob: np.ndarray,
    aoi: AOIConfig,
    output_path: str | None = None,
) -> str:
    """Plot FDP coffee probability map."""
    fig, ax = figure_with_title(
        "FDP Coffee Probability",
        f"{aoi.name} ({aoi.country})",
    )
    im = ax.imshow(coffee_prob, cmap=COFFEE_PROB_CMAP, vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Coffee probability", shrink=0.8)
    ax.axis("off")
    add_attribution(ax)

    if output_path is None:
        output_path = str(_output_dir(aoi.id) / "fdp_coffee_prob.png")
    save_figure(fig, output_path)
    logger.debug(f"Saved coffee probability: {output_path}")
    return output_path


def plot_hansen_loss(
    loss_year: np.ndarray,
    aoi: AOIConfig,
    output_path: str | None = None,
) -> str:
    """Plot Hansen forest loss year map."""
    fig, ax = figure_with_title(
        "Hansen Forest Loss Year",
        f"{aoi.name} ({aoi.country}) — 2001–2023",
    )
    masked = np.ma.masked_equal(loss_year, 0)
    im = ax.imshow(masked, cmap=LOSS_YEAR_CMAP, vmin=1, vmax=23)
    cbar = plt.colorbar(im, ax=ax, label="Loss year", shrink=0.8)
    cbar.set_ticks([1, 5, 10, 15, 20, 23])
    cbar.set_ticklabels(["2001", "2005", "2010", "2015", "2020", "2023"])
    ax.axis("off")
    add_attribution(ax)

    if output_path is None:
        output_path = str(_output_dir(aoi.id) / "hansen_loss.png")
    save_figure(fig, output_path)
    logger.debug(f"Saved Hansen loss: {output_path}")
    return output_path


def plot_hotspots_overlay(
    background: np.ndarray,
    hotspot_mask: np.ndarray,
    aoi: AOIConfig,
    output_path: str | None = None,
) -> str:
    """Plot hotspot polygons overlaid on a background image."""
    fig, ax = figure_with_title(
        "Coffee-Deforestation Hotspots",
        f"{aoi.name} ({aoi.country})",
    )

    # Show background (e.g., NDVI or RGB)
    if background.ndim == 3:
        ax.imshow(background)
    else:
        ax.imshow(background, cmap=NDVI_CMAP, vmin=-0.2, vmax=0.9)

    # Overlay hotspots
    hotspot_overlay = np.ma.masked_equal(hotspot_mask, 0)
    ax.imshow(hotspot_overlay, cmap="Reds", alpha=0.6, vmin=0, vmax=1)
    ax.axis("off")
    add_attribution(ax)

    if output_path is None:
        output_path = str(_output_dir(aoi.id) / "hotspots_overlay.png")
    save_figure(fig, output_path)
    logger.debug(f"Saved hotspots overlay: {output_path}")
    return output_path


def plot_feature_correlation(
    feature_array: np.ndarray,
    feature_names: list[str],
    aoi: AOIConfig,
    output_path: str | None = None,
) -> str:
    """Plot correlation matrix of the feature stack."""
    apply_theme()
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_title(
        "Feature Stack Correlation Matrix",
        fontsize=14, fontweight="bold", pad=15,
    )

    # Compute correlation
    valid_mask = ~np.any(np.isnan(feature_array), axis=1)
    valid = feature_array[valid_mask]

    if len(valid) > 10000:
        indices = np.random.choice(len(valid), 10000, replace=False)
        valid = valid[indices]

    corr = np.corrcoef(valid.T)

    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_xticks(range(len(feature_names)))
    ax.set_yticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(feature_names, fontsize=8)

    # Add correlation values
    for i in range(len(feature_names)):
        for j in range(len(feature_names)):
            color = "white" if abs(corr[i, j]) > 0.5 else "black"
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                    fontsize=6, color=color)

    add_attribution(ax, f"AOI: {aoi.name}")

    if output_path is None:
        output_path = str(_output_dir(aoi.id) / "feature_correlation.png")
    save_figure(fig, output_path)
    logger.debug(f"Saved feature correlation: {output_path}")
    return output_path


def plot_cloud_mask(
    rgb: np.ndarray,
    cloud_mask: np.ndarray,
    aoi: AOIConfig,
    year: int,
    output_path: str | None = None,
) -> str:
    """Plot S2 RGB with cloud mask overlay."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    axes[0].set_title("S2 RGB (pre-mask)")
    axes[0].imshow(np.clip(rgb, 0, 1))
    axes[0].axis("off")

    axes[1].set_title("Cloud mask overlay")
    axes[1].imshow(np.clip(rgb, 0, 1))
    cloud_overlay = np.ma.masked_equal(cloud_mask, 0)
    axes[1].imshow(cloud_overlay, cmap="Reds", alpha=0.5)
    axes[1].axis("off")

    fig.suptitle(
        f"Cloud Masking — {aoi.name} ({aoi.country}) — {year}",
        fontsize=14, fontweight="bold",
    )
    add_attribution(axes[0])

    if output_path is None:
        output_path = str(_output_dir(aoi.id) / f"cloud_mask_{year}.png")
    save_figure(fig, output_path)
    logger.debug(f"Saved cloud mask: {output_path}")
    return output_path


# --- Stage 2 figures ---


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    model_name: str,
    aoi: AOIConfig | None = None,
    output_path: str | None = None,
) -> str:
    """Plot a confusion matrix heatmap."""
    apply_theme()
    fig, ax = plt.subplots(figsize=(8, 7))

    # Normalize
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, shrink=0.8)

    n_classes = len(class_names)
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    # Add text annotations
    for i in range(n_classes):
        for j in range(n_classes):
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]:.0%})",
                    ha="center", va="center", fontsize=8, color=color)

    subtitle = f"Model: {model_name}"
    if aoi:
        subtitle += f" | AOI: {aoi.name}"
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold", pad=15)
    ax.text(0.5, 1.02, subtitle, transform=ax.transAxes, ha="center", fontsize=9, color="#666")
    add_attribution(ax)

    if output_path is None:
        aoi_id = aoi.id if aoi else "global"
        output_path = str(_output_dir(aoi_id) / f"confusion_matrix_{model_name}.png")
    save_figure(fig, output_path)
    logger.debug(f"Saved confusion matrix: {output_path}")
    return output_path


def plot_ablation_bar_chart(
    ablation_results: dict[str, dict],
    aoi: AOIConfig | None = None,
    output_path: str | None = None,
) -> str:
    """Plot S1-only / S2-only / S1+S2 ablation bar chart."""
    apply_theme()
    fig, ax = plt.subplots(figsize=(8, 5))

    names = list(ablation_results.keys())
    f1_scores = [ablation_results[n]["f1_coffee"] for n in names]
    display_names = {"s1_only": "S1 Only", "s2_only": "S2 Only", "s1_s2": "S1 + S2"}

    colors_list = [COLORS["coffee"], "#4A7BA6", COLORS["forest_stable"]]
    bars = ax.bar(
        [display_names.get(n, n) for n in names],
        f1_scores,
        color=colors_list[:len(names)],
        edgecolor="white",
        linewidth=0.5,
    )

    for bar, score in zip(bars, f1_scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{score:.3f}",
            ha="center", fontsize=10, fontweight="bold",
        )

    ax.set_ylabel("Coffee Class F1 Score")
    ax.set_ylim(0, 1.0)
    ax.set_title("Sensor Ablation Study", fontsize=14, fontweight="bold", pad=15)
    if aoi:
        ax.text(0.5, 1.02, f"AOI: {aoi.name}", transform=ax.transAxes,
                ha="center", fontsize=9, color="#666")
    add_attribution(ax)

    if output_path is None:
        aoi_id = aoi.id if aoi else "global"
        output_path = str(_output_dir(aoi_id) / "ablation_chart.png")
    save_figure(fig, output_path)
    logger.debug(f"Saved ablation chart: {output_path}")
    return output_path


def plot_historical_lookback(
    loss_year_before_coffee: np.ndarray,
    aoi: AOIConfig,
    output_path: str | None = None,
) -> str:
    """Plot historical look-back: year forest was lost before coffee."""
    apply_theme()
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: spatial map
    masked = np.ma.masked_equal(loss_year_before_coffee, 0)
    im = axes[0].imshow(masked, cmap=LOSS_YEAR_CMAP, vmin=1, vmax=23)
    cbar = plt.colorbar(im, ax=axes[0], shrink=0.8)
    cbar.set_ticks([1, 5, 10, 15, 20, 23])
    cbar.set_ticklabels(["2001", "2005", "2010", "2015", "2020", "2023"])
    axes[0].set_title("Forest Loss Year (Coffee Pixels)", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    # Right: histogram of loss years
    valid = loss_year_before_coffee[loss_year_before_coffee > 0]
    if len(valid) > 0:
        axes[1].hist(
            valid + 2000,
            bins=range(2001, 2025),
            color=COLORS["forest_loss"],
            edgecolor="white",
            linewidth=0.5,
        )
    axes[1].set_xlabel("Year of Forest Loss")
    axes[1].set_ylabel("Number of Coffee Pixels")
    axes[1].set_title("Distribution of Loss Years", fontsize=11, fontweight="bold")

    fig.suptitle(
        f"Coffee on Former Forest — {aoi.name} ({aoi.country})",
        fontsize=14, fontweight="bold",
    )
    add_attribution(axes[0])

    if output_path is None:
        output_path = str(_output_dir(aoi.id) / "historical_lookback.png")
    save_figure(fig, output_path)
    logger.debug(f"Saved historical lookback: {output_path}")
    return output_path


def plot_replacement_classes(
    class_distribution: dict[str, float],
    aoi: AOIConfig,
    output_path: str | None = None,
) -> str:
    """Plot pie chart of replacement land cover classes on former forest."""
    apply_theme()
    fig, ax = plt.subplots(figsize=(8, 8))

    labels = list(class_distribution.keys())
    sizes = list(class_distribution.values())

    color_map = {
        "coffee": COLORS["coffee"],
        "forest": COLORS["forest_stable"],
        "cropland": COLORS["non_coffee_cropland"],
        "built_bare": COLORS["built_bare"],
        "water": COLORS["water"],
    }
    colors_list = [color_map.get(l, "#999999") for l in labels]

    # Filter out zero-size slices
    non_zero = [(l, s, c) for l, s, c in zip(labels, sizes, colors_list) if s > 0.001]
    if non_zero:
        labels, sizes, colors_list = zip(*non_zero)

    ax.pie(
        sizes, labels=labels, colors=colors_list,
        autopct="%1.1f%%", startangle=90,
        textprops={"fontsize": 10},
    )
    ax.set_title(
        f"Post-Deforestation Land Cover — {aoi.name}",
        fontsize=14, fontweight="bold", pad=20,
    )

    if output_path is None:
        output_path = str(_output_dir(aoi.id) / "replacement_classes.png")
    save_figure(fig, output_path)
    logger.debug(f"Saved replacement classes: {output_path}")
    return output_path


def plot_classification_map(
    class_raster: np.ndarray,
    aoi: AOIConfig,
    class_names: list[str] | None = None,
    output_path: str | None = None,
) -> str:
    """Plot ML coffee classification map with legend.

    Args:
        class_raster: 2D array of class codes (0=coffee, 1=forest, etc., -1=nodata)
        aoi: AOI configuration
        class_names: list of class labels (default: standard 5-class set)
        output_path: optional output path
    """
    from matplotlib.colors import ListedColormap

    if class_names is None:
        class_names = ["Coffee", "Forest", "Cropland", "Built/Bare", "Water"]

    apply_theme()
    fig, ax = figure_with_title(
        "ML Coffee Classification",
        f"{aoi.name} ({aoi.country}) — reduced resolution (~300m)",
    )

    colors_list = [
        COLORS.get("coffee", "#6F4E37"),
        COLORS.get("forest_stable", "#2D5016"),
        COLORS.get("non_coffee_cropland", "#F4D35E"),
        COLORS.get("built_bare", "#9E9E9E"),
        COLORS.get("water", "#4A7BA6"),
    ]

    # Mask nodata
    masked = np.ma.masked_equal(class_raster, -1)
    unique_classes = sorted(set(class_raster.flat) - {-1})
    n_classes = max(len(unique_classes), 1)
    cmap = ListedColormap(colors_list[:max(n_classes, len(colors_list))])

    ax.imshow(masked, cmap=cmap, vmin=0, vmax=len(colors_list) - 1, interpolation="nearest")
    ax.axis("off")

    # Legend
    patches = [
        plt.Rectangle((0, 0), 1, 1, fc=colors_list[i])
        for i in range(min(len(class_names), len(colors_list)))
    ]
    ax.legend(patches, class_names[:len(patches)], loc="lower right", fontsize=8,
              framealpha=0.9)

    add_attribution(ax, "ML classification (Random Forest, reduced resolution)")

    if output_path is None:
        output_path = str(_output_dir(aoi.id) / "classification_map.png")
    save_figure(fig, output_path)
    logger.debug(f"Saved classification map: {output_path}")
    return output_path


def plot_area_by_year(
    area_ha_by_loss_year: dict[str, float],
    aoi: AOIConfig,
    output_path: str | None = None,
) -> str:
    """Bar chart of deforestation area per loss year."""
    apply_theme()
    fig, ax = figure_with_title(
        "Deforestation Area by Loss Year",
        f"{aoi.name} ({aoi.country}) — coffee-linked hotspot area (ha)",
    )

    if not area_ha_by_loss_year:
        ax.text(0.5, 0.5, "No area-by-year data", transform=ax.transAxes,
                ha="center", va="center", fontsize=12, color="#888")
    else:
        years = sorted(area_ha_by_loss_year.keys(), key=int)
        areas = [area_ha_by_loss_year[y] for y in years]
        bars = ax.bar(years, areas, color=COLORS.get("forest_loss", "#C1292E"), width=0.7)
        ax.set_xlabel("Loss Year (Hansen GFC)")
        ax.set_ylabel("Area (hectares)")
        ax.tick_params(axis="x", rotation=45)
        # Add value labels on bars
        for bar, area in zip(bars, areas):
            if area > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{area:.0f}", ha="center", va="bottom", fontsize=7)

    add_attribution(ax, "Rule-based detection (Hansen loss + FDP coffee > 50%)")

    if output_path is None:
        output_path = str(_output_dir(aoi.id) / "area_by_year.png")
    save_figure(fig, output_path)
    logger.debug(f"Saved area-by-year chart: {output_path}")
    return output_path


def plot_ndvi_trajectory(
    ndvi_by_year: dict[int, float],
    aoi: AOIConfig,
    output_path: str | None = None,
) -> str:
    """Line chart of AOI-wide NDVI over time."""
    apply_theme()
    fig, ax = figure_with_title(
        "NDVI Temporal Trajectory",
        f"{aoi.name} ({aoi.country}) — AOI-wide mean NDVI by year",
    )

    if not ndvi_by_year:
        ax.text(0.5, 0.5, "No per-year NDVI data", transform=ax.transAxes,
                ha="center", va="center", fontsize=12, color="#888")
    else:
        years = sorted(ndvi_by_year.keys())
        values = [ndvi_by_year[y] for y in years]
        ax.plot(years, values, "o-", color=COLORS.get("forest_stable", "#2D5016"),
                linewidth=2, markersize=6)
        ax.fill_between(years, values, alpha=0.15, color=COLORS.get("forest_stable", "#2D5016"))
        ax.set_xlabel("Year")
        ax.set_ylabel("Mean NDVI")
        ax.set_ylim(0, 1)
        for yr, val in zip(years, values):
            ax.annotate(f"{val:.3f}", (yr, val), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=8)

    add_attribution(ax, "From Sentinel-2 annual dry-season composites")

    if output_path is None:
        output_path = str(_output_dir(aoi.id) / "ndvi_trajectory.png")
    save_figure(fig, output_path)
    logger.debug(f"Saved NDVI trajectory: {output_path}")
    return output_path


def plot_attribution_pie(
    attribution: dict[str, float],
    aoi: AOIConfig,
    output_path: str | None = None,
) -> str:
    """Pie chart: 'What replaced the forest?' across all loss pixels."""
    from coffee_deforestation.viz.theme import REPLACEMENT_COLORS, REPLACEMENT_NAMES

    apply_theme()
    fig, ax = figure_with_title(
        "What Replaced the Forest?",
        f"{aoi.name} ({aoi.country}) — all deforestation, not just coffee-linked",
    )

    total = attribution.get("total_loss_ha", 0)
    categories = ["coffee", "other_crops", "built_industrial", "bare_degraded", "water", "regrowth"]
    sizes = [attribution.get(f"{c}_pct", 0) for c in categories]
    labels = [REPLACEMENT_NAMES.get(c, c) for c in categories]
    colors_list = [REPLACEMENT_COLORS.get(c, "#CCC") for c in categories]

    # Filter out zero categories
    nonzero = [(s, l, c) for s, l, c in zip(sizes, labels, colors_list) if s > 0.5]
    if nonzero:
        sizes_f, labels_f, colors_f = zip(*nonzero)
        wedges, texts, autotexts = ax.pie(
            sizes_f, labels=labels_f, colors=colors_f,
            autopct="%1.1f%%", startangle=90,
            textprops={"fontsize": 9},
            pctdistance=0.75,
        )
        for t in autotexts:
            t.set_fontsize(8)
            t.set_fontweight("bold")
    else:
        ax.text(0.5, 0.5, "No loss data", ha="center", va="center", fontsize=12, color="#888")

    ax.set_title(
        f"What Replaced the Forest?\n{aoi.name} — {total:,.0f} ha total loss",
        fontsize=13, fontweight="bold", pad=15,
    )

    if output_path is None:
        output_path = str(_output_dir(aoi.id) / "attribution_pie.png")
    save_figure(fig, output_path)
    logger.debug(f"Saved attribution pie: {output_path}")
    return output_path


def plot_attribution_stacked_bar(
    yearly_attribution: dict[int, dict[str, float]],
    aoi: AOIConfig,
    output_path: str | None = None,
) -> str:
    """Stacked bar chart: replacement land cover by loss year."""
    from coffee_deforestation.viz.theme import REPLACEMENT_COLORS

    apply_theme()
    fig, ax = figure_with_title(
        "Deforestation Drivers by Year",
        f"{aoi.name} ({aoi.country}) — what replaced forest each year",
        figsize=(12, 6),
    )

    categories = ["coffee", "other_crops", "built_industrial", "bare_degraded", "regrowth"]
    cat_labels = ["Coffee", "Other Crops", "Built/Industrial", "Bare/Degraded", "Regrowth"]

    if not yearly_attribution:
        ax.text(0.5, 0.5, "No per-year attribution data", ha="center", va="center",
                fontsize=12, color="#888")
    else:
        years = sorted(yearly_attribution.keys())
        x = np.arange(len(years))
        width = 0.7
        bottoms = np.zeros(len(years))

        for cat, label in zip(categories, cat_labels):
            values = [yearly_attribution[y].get(f"{cat}_pct", 0) for y in years]
            color = REPLACEMENT_COLORS.get(cat, "#CCC")
            ax.bar(x, values, width, bottom=bottoms, label=label, color=color)
            bottoms += np.array(values)

        ax.set_xticks(x)
        ax.set_xticklabels([str(y) for y in years], rotation=45, fontsize=8)
        ax.set_ylabel("Percentage of Loss (%)")
        ax.set_xlabel("Loss Year (Hansen GFC)")
        ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
        ax.set_ylim(0, 105)

    add_attribution(ax, "Hansen GFC + WorldCover + FDP")

    if output_path is None:
        output_path = str(_output_dir(aoi.id) / "attribution_stacked_bar.png")
    save_figure(fig, output_path)
    logger.debug(f"Saved attribution stacked bar: {output_path}")
    return output_path


def plot_before_after(
    rgb_before: np.ndarray,
    rgb_after: np.ndarray,
    aoi: AOIConfig,
    year_before: int,
    year_after: int,
    bbox: BBox | None = None,
    output_path: str | None = None,
) -> str:
    """Side-by-side RGB composites showing landscape change."""
    from coffee_deforestation.viz.theme import add_scale_bar, format_coordinate_axes

    apply_theme()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(
        f"Before / After — {aoi.name} ({aoi.country})",
        fontsize=14, fontweight="bold", y=0.98,
    )

    # Clip to visual range
    for ax, rgb, year, label in [
        (ax1, rgb_before, year_before, "Before"),
        (ax2, rgb_after, year_after, "After"),
    ]:
        clipped = np.clip(rgb, 0, 0.3) / 0.3
        ax.imshow(clipped)
        ax.set_title(f"{label}: {year} dry season", fontsize=11, fontweight="bold")
        if bbox:
            format_coordinate_axes(ax, bbox)
            add_scale_bar(ax, bbox)
        else:
            ax.axis("off")

    fig.tight_layout()
    add_attribution(ax2, "Sentinel-2 annual composites")

    if output_path is None:
        output_path = str(_output_dir(aoi.id) / f"before_after_{year_before}_{year_after}.png")
    save_figure(fig, output_path)
    logger.debug(f"Saved before/after: {output_path}")
    return output_path


def plot_ndvi_change(
    delta: np.ndarray,
    aoi: AOIConfig,
    year_before: int,
    year_after: int,
    bbox: BBox | None = None,
    output_path: str | None = None,
) -> str:
    """NDVI difference map with diverging colormap: red=loss, green=gain."""
    from coffee_deforestation.viz.theme import (
        NDVI_CHANGE_CMAP,
        add_scale_bar,
        format_coordinate_axes,
    )

    apply_theme()
    fig, ax = figure_with_title(
        f"NDVI Change ({year_before} to {year_after})",
        f"{aoi.name} — red = vegetation loss, green = recovery",
    )

    vmax = max(abs(np.nanmin(delta)), abs(np.nanmax(delta)), 0.3)
    im = ax.imshow(delta, cmap=NDVI_CHANGE_CMAP, vmin=-vmax, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("NDVI Change", fontsize=9)

    if bbox:
        format_coordinate_axes(ax, bbox)
        add_scale_bar(ax, bbox)
    else:
        ax.axis("off")

    # Add summary stats
    mean_delta = np.nanmean(delta)
    pct_loss = np.nanmean(delta < -0.05) * 100
    ax.text(0.02, 0.02, f"Mean: {mean_delta:+.3f} | Pixels declining: {pct_loss:.1f}%",
            transform=ax.transAxes, fontsize=8, color="white",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7))

    add_attribution(ax, f"NDVI delta: {year_after} minus {year_before}")

    if output_path is None:
        output_path = str(_output_dir(aoi.id) / f"ndvi_change_{year_before}_{year_after}.png")
    save_figure(fig, output_path)
    logger.debug(f"Saved NDVI change: {output_path}")
    return output_path


def plot_yearly_loss_comparison(
    yearly_stats: dict[int, dict[str, float]],
    aoi: AOIConfig,
    output_path: str | None = None,
) -> str:
    """Bar chart comparing total loss vs coffee-linked loss per year."""
    apply_theme()
    fig, ax = figure_with_title(
        "Annual Deforestation: Total vs Coffee-Linked",
        f"{aoi.name} ({aoi.country}) — hectares per Hansen loss year",
        figsize=(12, 6),
    )

    if not yearly_stats:
        ax.text(0.5, 0.5, "No per-year data", ha="center", va="center", fontsize=12, color="#888")
    else:
        years = sorted(yearly_stats.keys())
        x = np.arange(len(years))
        width = 0.35

        total_ha = [yearly_stats[y].get("total_loss_ha", 0) for y in years]
        coffee_ha = [yearly_stats[y].get("coffee_loss_ha", 0) for y in years]

        ax.bar(x - width / 2, total_ha, width, label="All Forest Loss",
               color=COLORS.get("forest_loss", "#C1292E"), alpha=0.7)
        ax.bar(x + width / 2, coffee_ha, width, label="Coffee-Linked Loss",
               color=COLORS.get("coffee", "#6F4E37"))

        ax.set_xticks(x)
        ax.set_xticklabels([str(y) for y in years], rotation=45, fontsize=8)
        ax.set_ylabel("Area (hectares)")
        ax.set_xlabel("Loss Year")
        ax.legend(fontsize=9)

    add_attribution(ax, "Hansen GFC + FDP coffee probability")

    if output_path is None:
        output_path = str(_output_dir(aoi.id) / "yearly_loss_comparison.png")
    save_figure(fig, output_path)
    logger.debug(f"Saved yearly loss comparison: {output_path}")
    return output_path


def plot_region_overview(
    aoi: AOIConfig,
    patch_results: list[dict],
    output_path: str | None = None,
) -> str:
    """Country-scale overview map showing all patches with hotspot counts.

    Args:
        aoi: main AOI config (with region_bbox and patches)
        patch_results: list of dicts with {name, bbox, hotspot_count, total_area_ha}
        output_path: optional output path
    """
    from coffee_deforestation.viz.theme import format_coordinate_axes

    apply_theme()
    bbox = aoi.region_bbox or aoi.bbox
    fig, ax = figure_with_title(
        f"Regional Overview — {aoi.name}",
        f"{aoi.country} — {len(patch_results)} sample patches across the coffee belt",
        figsize=(12, 10),
    )

    # Background: simple lat/lon grid
    ax.set_xlim(bbox.west, bbox.east)
    ax.set_ylim(bbox.south, bbox.north)
    ax.set_aspect("equal")
    ax.set_facecolor("#E8F4FD")  # light blue for "ocean/background"

    # Draw each patch as a colored rectangle
    from matplotlib.patches import Rectangle

    for pr in patch_results:
        pb = pr["bbox"]
        hotspots = pr.get("hotspot_count", 0)

        # Color by hotspot intensity
        intensity = min(hotspots / 500, 1.0)  # normalize to 0-1
        color = plt.cm.YlOrRd(intensity)  # type: ignore[attr-defined]

        rect = Rectangle(
            (pb.west, pb.south), pb.width_deg, pb.height_deg,
            linewidth=1.5, edgecolor="black", facecolor=color, alpha=0.7,
        )
        ax.add_patch(rect)

        # Label
        cx = (pb.west + pb.east) / 2
        cy = (pb.south + pb.north) / 2
        ax.text(cx, cy, f"{pr['name']}\n{hotspots} hotspots\n{pr.get('total_area_ha', 0):.0f} ha",
                ha="center", va="center", fontsize=7, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, alpha=0.3)

    # Colorbar for hotspot density
    sm = plt.cm.ScalarMappable(cmap="YlOrRd", norm=plt.Normalize(0, 500))  # type: ignore[attr-defined]
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Hotspot Count", fontsize=9)

    add_attribution(ax, "Patches selected across the regional coffee belt")

    if output_path is None:
        output_path = str(_output_dir(aoi.id) / "region_overview.png")
    save_figure(fig, output_path)
    logger.debug(f"Saved region overview: {output_path}")
    return output_path
