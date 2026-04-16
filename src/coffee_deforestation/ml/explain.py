"""Feature importance and model explanation.

What: Extracts and visualizes feature importance from trained RF and XGBoost models.
Why: Interpretability is critical — reviewers need to understand what drives predictions.
Assumes: Trained models are available. Feature names match the feature stack.
Produces: Feature importance bar charts saved to outputs/figures/.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from coffee_deforestation.config import AOIConfig, PROJECT_ROOT
from coffee_deforestation.features.stack import get_feature_names
from coffee_deforestation.viz.theme import COLORS, add_attribution, apply_theme, save_figure


def get_feature_importance(
    model: Any,
    feature_names: list[str] | None = None,
) -> dict[str, float]:
    """Extract feature importances from a trained model.

    Works with both RandomForest and XGBoost (both expose .feature_importances_).
    """
    if feature_names is None:
        feature_names = get_feature_names()

    importances = model.feature_importances_  # type: ignore[union-attr]

    importance_dict = dict(zip(feature_names, importances.tolist()))
    # Sort by importance
    importance_dict = dict(
        sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    )

    return importance_dict


def plot_feature_importance(
    importances: dict[str, float],
    model_name: str,
    aoi: AOIConfig | None = None,
    output_path: str | None = None,
) -> str:
    """Plot a horizontal bar chart of feature importances."""
    apply_theme()

    names = list(importances.keys())
    values = list(importances.values())

    fig, ax = plt.subplots(figsize=(10, 8))

    # Color bars by feature category
    bar_colors = []
    for name in names:
        if name in ("ndvi", "evi", "ndwi", "nbr", "savi", "B4", "B8", "B11", "B12"):
            bar_colors.append("#4A7BA6")  # S2 features — blue
        elif name in ("vv_median", "vh_median", "vv_vh_ratio", "vv_stddev", "vh_stddev"):
            bar_colors.append(COLORS["coffee"])  # S1 features — brown
        else:
            bar_colors.append(COLORS["built_bare"])  # contextual — gray

    y_pos = np.arange(len(names))
    ax.barh(y_pos, values, color=bar_colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Importance")

    subtitle = f"Model: {model_name}"
    if aoi:
        subtitle += f" | AOI: {aoi.name}"
    ax.set_title("Feature Importance", fontsize=14, fontweight="bold", pad=15)
    ax.text(0.5, 1.02, subtitle, transform=ax.transAxes, ha="center", fontsize=9, color="#666")

    # Legend for feature categories
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#4A7BA6", label="Optical (S2)"),
        Patch(facecolor=COLORS["coffee"], label="SAR (S1)"),
        Patch(facecolor=COLORS["built_bare"], label="Contextual"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

    add_attribution(ax)

    if output_path is None:
        fig_dir = PROJECT_ROOT / "outputs" / "figures"
        if aoi:
            fig_dir = fig_dir / aoi.id
        fig_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(fig_dir / f"feature_importance_{model_name}.png")

    save_figure(fig, output_path)
    logger.info(f"Saved feature importance plot: {output_path}")
    return output_path
