# plot_style.py
"""Shared figure style for the ZERO figures. Import and call set_style() before plotting."""
from __future__ import annotations

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Helvetica is not installed everywhere: Windows ships Arial, Linux the URW / TeX Gyre clones.
# Asking for it directly makes matplotlib warn once per text object and fall back to DejaVu Sans,
# so the figures silently change font. Take the first metric-compatible family actually present.
FONT_STACK = ["Helvetica", "Arial", "Nimbus Sans", "TeX Gyre Heros", "Liberation Sans", "DejaVu Sans"]

FONT_BIG, FONT_MED = 20, 16
C_ZERO, C_OTHER = plt.cm.viridis(0.25), "#B8B8B8"                                  # 0-facts vs the rest
C_DIR = [plt.cm.viridis(0.2), plt.cm.viridis(0.6)]                                 # 0×n vs n×0
C_TYPES = [plt.cm.viridis(0.2), plt.cm.viridis(0.45), plt.cm.viridis(0.68), "#B8B8B8"]   # error types


def resolve_font(stack: list[str] = FONT_STACK) -> str:
    available = {f.name for f in fm.fontManager.ttflist}
    return next((f for f in stack if f in available), "sans-serif")


FONT = resolve_font()


def set_style() -> None:
    sns.set_theme(font_scale=1.4, style="ticks", font=FONT)
    plt.rcParams.update({"pdf.fonttype": 42, "svg.fonttype": "none"})   # keep text editable in PDF/SVG


def paint_it_black(axes) -> None:
    for ax in np.ravel(axes):
        for spine in ax.spines.values():
            spine.set_color("black"); spine.set_linewidth(1)
        ax.tick_params(colors="black", width=1)
        for lab in ax.get_xticklabels() + ax.get_yticklabels():
            lab.set_color("black")
        ax.xaxis.label.set_color("black"); ax.yaxis.label.set_color("black")
