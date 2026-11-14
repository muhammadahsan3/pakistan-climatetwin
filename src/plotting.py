"""
Shared figure style so every chart in the notebooks, paper and slides looks consistent.

Colour rules used throughout:
  * categorical (provinces, seasons, models): fixed palette, always in the same order,
    so "Punjab" or "XGBoost" has the same colour in every figure;
  * sequential (magnitude, e.g. rainfall): one hue, light -> dark;
  * diverging (trends / anomalies with a meaningful zero): blue (cooling / drier)
    <-> neutral grey (zero) <-> red (warming), always symmetric around zero.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from src import config as C

CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

PROVINCE_ORDER = ["Punjab", "Sindh", "Khyber Pakhtunkhwa", "Balochistan",
                  "Gilgit-Baltistan", "Azad Kashmir", "Islamabad Capital Territory"]
PROVINCE_COLORS = dict(zip(PROVINCE_ORDER, CATEGORICAL))

SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281", "#0d366b"])
SEQ_ORANGE = LinearSegmentedColormap.from_list(
    "seq_orange", ["#fde8dd", "#f6b393", "#eb6834", "#c24a1a", "#8a320f"])
DIVERGING = LinearSegmentedColormap.from_list(
    "div_blue_red", ["#104281", "#2a78d6", "#9ec5f4", "#f0efec", "#f2a3a2", "#e34948", "#9c2121"])


def set_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelcolor": INK_2, "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
        "legend.frameon": False, "lines.linewidth": 2,
    })


def symmetric_norm(values, vmax: float | None = None) -> TwoSlopeNorm:
    """Diverging colour scale centred exactly on zero."""
    v = vmax or float(np.nanmax(np.abs(values)))
    return TwoSlopeNorm(vmin=-v, vcenter=0, vmax=v)


def save(fig, name: str) -> None:
    C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(C.FIGURES_DIR / f"{name}.png")


def map_axes(ax) -> None:
    ax.set_axis_off()
    ax.set_aspect("equal")
