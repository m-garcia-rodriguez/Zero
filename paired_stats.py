# paired_stats.py
"""Paired within-student tests shared by the RT notebooks (DECISIONS #19)."""
from __future__ import annotations

import numpy as np
from scipy.stats import binom, rankdata, wilcoxon


def median_ci(x, alpha: float = 0.05) -> tuple[float, float]:
    """Distribution-free CI of the median, from the order statistics."""
    x = np.sort(np.asarray(x)); k = int(binom.ppf(alpha / 2, len(x), 0.5))
    return (x[k - 1], x[len(x) - k]) if k >= 1 else (np.nan, np.nan)


def signed_rank(dif) -> dict:
    """Wilcoxon signed-rank on a vector of per-student differences, with effect size and median CI."""
    dif = np.asarray(dif); nz = dif[dif != 0]; r = rankdata(np.abs(nz))
    w_pos, w_neg = r[nz > 0].sum(), r[nz < 0].sum()
    lo, hi = median_ci(dif)
    return {"students": len(dif), "med_diff": np.median(dif), "ci_lo": lo, "ci_hi": hi,
            "pct_slower": 100 * (dif > 0).mean(), "r_rb": (w_pos - w_neg) / (w_pos + w_neg),
            "p": wilcoxon(nz, alternative="two-sided").pvalue}


def holm(p) -> np.ndarray:
    """Holm family-wise correction over a family of p values."""
    p = np.asarray(p); order = np.argsort(p); m = len(p)
    out = np.empty(m)
    out[order] = np.maximum.accumulate((m - np.arange(m)) * p[order]).clip(max=1)
    return out


def stars(p) -> str:
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "ns"
