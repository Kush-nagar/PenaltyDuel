"""
Reusable goal-mouth visualizations for penalty placement EDA.

StatsBomb shot zones are named from the goalkeeper's point of view:
{low, high} x {left, center, right}.
"""

from __future__ import annotations

from collections.abc import Mapping

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle


ZONE_GRID = [
    ["high_left", "high_center", "high_right"],
    ["low_left", "low_center", "low_right"],
]

ZONE_ORDER = [zone for row in ZONE_GRID for zone in row]

ZONE_LABELS = {
    "high_left": "High left",
    "high_center": "High center",
    "high_right": "High right",
    "low_left": "Low left",
    "low_center": "Low center",
    "low_right": "Low right",
}


def _zone_value(values: Mapping[str, float], zone: str) -> float:
    value = values.get(zone, 0.0)
    if value is None:
        return 0.0
    return float(value)


def plot_goalmouth_heatmap(
    values: Mapping[str, float],
    *,
    ax: Axes | None = None,
    title: str = "",
    cmap: str = "Blues",
    value_format: str = "{:.0f}",
    cbar_label: str | None = None,
) -> Axes:
    """
    Draw a 2x3 goal-mouth heatmap for the six canonical shot zones.

    Parameters
    ----------
    values:
        Mapping from zone name to numeric value. Missing zones are shown as 0.
    ax:
        Optional Matplotlib axes. A new axes is created when omitted.
    title:
        Plot title.
    cmap:
        Matplotlib colormap name.
    value_format:
        Format string used for zone annotations.
    cbar_label:
        Optional colorbar label.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    zone_values = {zone: _zone_value(values, zone) for zone in ZONE_ORDER}
    max_value = max(zone_values.values()) if zone_values else 0.0
    norm = Normalize(vmin=0.0, vmax=max(max_value, 1e-9))
    color_map = plt.get_cmap(cmap)

    for row_idx, row in enumerate(ZONE_GRID):
        for col_idx, zone in enumerate(row):
            value = zone_values[zone]
            patch = Rectangle(
                (col_idx, row_idx),
                width=1,
                height=1,
                facecolor=color_map(norm(value)),
                edgecolor="black",
                linewidth=1.2,
            )
            ax.add_patch(patch)
            ax.text(
                col_idx + 0.5,
                row_idx + 0.5,
                f"{ZONE_LABELS[zone]}\n{value_format.format(value)}",
                ha="center",
                va="center",
                fontsize=10,
                color="black",
            )

    ax.set_xlim(0, 3)
    ax.set_ylim(2, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    ax.set_xlabel("Keeper perspective")

    if cbar_label:
        scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=color_map)
        plt.colorbar(scalar_mappable, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)

    return ax
