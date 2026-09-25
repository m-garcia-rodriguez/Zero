# binary_stats.py
"""2 x 2 association measures for the sequential notebooks (DECISIONS #38)."""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import multivariate_normal, norm


def wilson(k: float, n: float, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval for a proportion. Behaves at the extremes, unlike the Wald interval."""
    z = norm.ppf(1 - alpha / 2); ph = k / n; d = 1 + z * z / n
    centre = (ph + z * z / (2 * n)) / d
    half = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return centre - half, centre + half


def contrast_2x2(prev, nxt) -> dict:
    """P(next correct | previous correct) against P(next correct | previous error).

    Wilson CI on each probability, Newcombe hybrid-score CI on their difference (it inherits the
    Wilson behaviour near 0 and 1, where a Wald CI on a difference of ~93% vs ~52% misbehaves),
    risk ratio, and odds ratio with the Woolf log CI. Percentages, not proportions.
    """
    prev, nxt = np.asarray(prev, float), np.asarray(nxt, float)
    k1, n1 = nxt[prev == 1].sum(), float((prev == 1).sum())
    k0, n0 = nxt[prev == 0].sum(), float((prev == 0).sum())
    if min(n1, n0) == 0:
        return {}
    p1, p0 = k1 / n1, k0 / n0
    l1, u1 = wilson(k1, n1); l0, u0 = wilson(k0, n0)
    a, b, c, d = k1, n1 - k1, k0, n0 - k0
    orr = (a * d) / (b * c) if b * c else np.nan
    se = np.sqrt(1 / max(a, .5) + 1 / max(b, .5) + 1 / max(c, .5) + 1 / max(d, .5))
    return dict(students=n1 + n0, n_C1=n1, n_E1=n0,
                pC2C1=100 * p1, C1lo=100 * l1, C1hi=100 * u1,
                pC2E1=100 * p0, E1lo=100 * l0, E1hi=100 * u0,
                diff=100 * (p1 - p0),
                dlo=100 * (p1 - p0 - np.hypot(p1 - l1, u0 - p0)),
                dhi=100 * (p1 - p0 + np.hypot(u1 - p1, p0 - l0)),
                RR=p1 / p0 if p0 else np.nan,
                OR=orr, ORlo=orr * np.exp(-1.96 * se), ORhi=orr * np.exp(1.96 * se))


def tetrachoric(prev, nxt) -> float:
    """Tetrachoric correlation of two binary variables.

    The odds ratio is not comparable across tables with different marginals, and the problem types
    compared here differ a lot in how often a trial is correct (×1 is near ceiling, retrieval is
    not). The tetrachoric r assumes a latent bivariate normal behind the two dichotomies and is
    invariant to where the thresholds fall, so it is the measure that answers "is the association
    stronger for ×0" without the base rate doing the talking.
    """
    prev, nxt = np.asarray(prev, float), np.asarray(nxt, float)
    n = len(prev)
    both = float(((prev == 1) & (nxt == 1)).sum()) / n
    h1, h2 = norm.ppf((prev == 0).mean()), norm.ppf((nxt == 0).mean())
    if not np.isfinite(h1) or not np.isfinite(h2):
        return np.nan
    f = lambda r: multivariate_normal(mean=[0, 0], cov=[[1, r], [r, 1]]).cdf([-h1, -h2]) - both
    try:
        return brentq(f, -0.999, 0.999, xtol=1e-5)
    except ValueError:
        return np.nan


def tetrachoric_ci(prev, nxt, rng, n_boot: int = 200, alpha: float = 0.05) -> tuple[float, float, float]:
    """Tetrachoric r with a percentile bootstrap CI over the units (one row = one student)."""
    prev, nxt = np.asarray(prev, float), np.asarray(nxt, float)
    r = tetrachoric(prev, nxt)
    idx = rng.integers(0, len(prev), (n_boot, len(prev)))
    boot = np.array([tetrachoric(prev[i], nxt[i]) for i in idx])
    return r, np.nanpercentile(boot, 100 * alpha / 2), np.nanpercentile(boot, 100 * (1 - alpha / 2))
