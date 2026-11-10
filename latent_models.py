# latent_models.py
"""Discrete types against a continuum, on categorical answers (sequential_zero_consistency.ipynb, section 15b).

Both models are finite mixtures over "components" and share one engine:

  LCA  -- K latent classes. Class k answers category y with logit alpha[k, y]; the class shares differ by age group.
  IRT  -- nominal-response model with one continuous ability theta ~ N(mu_g, sigma_g) per age group, integrated
          by Gauss-Hermite quadrature: logit = a[y] * theta + c[y]. Category 0 (correct) is the reference, so a[y]
          says how fast answer y disappears as ability rises.

Both get the SAME trial-level terms X @ B (fact and order dummies, one coefficient per non-reference category,
shared across classes / ability levels), so neither model wins by having item structure the other lacks.
Answers are independent given the component (local independence) -- checked afterwards, not assumed away.

Estimation: marginal maximum likelihood, analytic gradient, L-BFGS. Trials are processed in chunks so memory stays
bounded whatever the number of quadrature nodes. A tiny ridge (RIDGE) keeps logits finite when a class never gives
some answer; it is excluded from the reported log-likelihood.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import log_softmax, logsumexp, softmax

RIDGE = 1e-2


class LatentData:
    """y (T,) category codes 0..C-1, 0 = correct; unit (T,) 0..S-1; g_unit (S,) age group 0..G-1; X (T, p).

    The likelihood only depends on which (age group, X row) CELL a trial falls in and on its answer, so the data are
    stored as a sparse count matrix N (units x cell*answer) and every evaluation is a couple of sparse products."""

    def __init__(self, y, unit, g_unit, X, C, G=None):
        self.y = np.asarray(y, np.int64); self.unit = np.asarray(unit, np.int64)
        self.g_unit = np.asarray(g_unit, np.int64); self.X = np.asarray(X, float)
        self.C, self.T, self.S, self.p = C, len(self.y), len(self.g_unit), self.X.shape[1]
        self.G = int(G if G is not None else self.g_unit.max() + 1)
        self.g_trial = self.g_unit[self.unit]
        self.n_trials = np.bincount(self.unit, minlength=self.S)
        Xu, xr = np.unique(self.X, axis=0, return_inverse=True)
        nx = len(Xu)
        self.cell_t = self.g_trial * nx + xr.ravel()                    # (T,) cell of each trial
        self.n_cell = self.G * nx
        self.g_cell = np.repeat(np.arange(self.G), nx)
        self.X_cell = np.tile(Xu, (self.G, 1))
        self.N = sparse.csr_matrix((np.ones(self.T), (self.unit, self.cell_t * C + self.y)),
                                   shape=(self.S, self.n_cell * C))
        self.Ug = sparse.csr_matrix((np.ones(self.S), (self.g_unit, np.arange(self.S))), shape=(self.G, self.S))

    def subset(self, mask):
        """Keep the trials in mask; every unit stays (a unit with no trial left contributes its prior only)."""
        return LatentData(self.y[mask], self.unit[mask], self.g_unit, self.X[mask], self.C, self.G)


# ---------------------------------------------------------------------------------------------------- engine
def _logp(A, B, d):
    """log P(answer | component) per cell: (n_cell, M, C)."""
    Z = A[d.g_cell] + (d.X_cell @ B)[:, None, :]
    return Z - logsumexp(Z, axis=2, keepdims=True)


def _posterior(A, logw, B, d):
    lp = _logp(A, B, d)
    post = logw[d.g_unit] + d.N @ lp.transpose(0, 2, 1).reshape(-1, lp.shape[1])
    lse = logsumexp(post, axis=1)
    return lse.sum(), np.exp(post - lse[:, None]), lp


def _mix(A, logw, B, d):
    """Marginal log-likelihood, posterior R (S, M) and the gradients wrt A (G, M, C), B (p, C), logw-softmax (G, M)."""
    LL, R, lp = _posterior(A, logw, B, d)
    M, C = R.shape[1], d.C
    W = np.asarray(d.N.T @ R).reshape(d.n_cell, C, M).transpose(0, 2, 1)   # posterior-weighted answer counts
    dZ = W - W.sum(2, keepdims=True) * np.exp(lp)
    dA = np.zeros_like(A); np.add.at(dA, d.g_cell, dZ)
    dB = d.X_cell.T @ dZ.sum(1)
    dW = d.Ug @ R - np.asarray(d.Ug.sum(1)) * np.exp(logw)
    return LL, R, dA, dB, dW


class Fit:
    """A fitted model. components(d) -> (A, logw, B); everything else is derived from those."""

    def __init__(self, kind, M, theta, LL, d, unpack, n_par):
        self.kind, self.M, self.theta, self.LL, self.unpack, self.n_par = kind, M, theta, LL, unpack, n_par
        self.S = d.S
        self.bic = -2 * LL + n_par * np.log(d.S)
        self.aic = -2 * LL + 2 * n_par
        self.name = f"{M} class{'es' if M > 1 else ''}" if kind == "lca" else "IRT (continuous)"

    def components(self):
        return self.unpack(self.theta)

    def posterior(self, d):
        A, logw, B = self.components()
        return _posterior(A, logw, B, d)[1]

    # -- diagnostics ------------------------------------------------------------------------------------------
    def class_shares(self):
        """(G, K) model-implied class shares per age group (LCA only)."""
        return np.exp(self.components()[1])

    def profiles(self, d, R=None):
        """(K, C) answer-type probabilities of each class, averaged over the trials the class actually answered."""
        A, _, B = self.components(); R = self.posterior(d) if R is None else R
        n = np.asarray(d.N.T @ R).reshape(d.n_cell, d.C, -1).sum(1)          # (n_cell, M) weighted trials
        num = np.einsum("cm,cmy->my", n, np.exp(_logp(A, B, d)))
        return num / n.sum(0)[:, None]

    def trial_probs(self, d):
        """(T, M, C) answer-type probabilities of every trial under every component."""
        A, _, B = self.components()
        return np.exp(_logp(A, B, d))[d.cell_t]

    def separation(self, d, R=None):
        """Relative entropy (1 = every unit certain of its class) and AvePP per modal class (LCA only)."""
        R = self.posterior(d) if R is None else R
        K = R.shape[1]
        if K == 1:
            return 1.0, np.array([1.0])
        ent = 1 + (R * np.log(np.clip(R, 1e-300, None))).sum() / (len(R) * np.log(K))
        modal = R.argmax(1)
        avepp = np.array([R[modal == k, k].mean() if (modal == k).any() else np.nan for k in range(K)])
        return ent, avepp

    def simulate(self, d, rng):
        """Answers drawn from the fitted model for d's own trials: a component per unit from its age group's
        distribution (a continuous theta for IRT), then each answer independently."""
        A, logw, B = self.components()
        if self.kind == "lca":
            comp = _draw(np.exp(logw[d.g_unit]), rng)
            Z = A[d.g_trial, comp[d.unit]] + d.X @ B
        else:
            a, c, mu, sig = self.ability
            th = mu[d.g_unit] + sig[d.g_unit] * rng.standard_normal(d.S)
            Z = np.concatenate([np.zeros((d.T, 1)), th[d.unit, None] * a + c], 1) + d.X @ B
        return _draw(softmax(Z, axis=1), rng)

    def heldout_logscore(self, d_train, d_test):
        """Mean log P(held-out answer | the unit's other answers) -- the out-of-sample comparison of the models."""
        A, logw, B = self.components()
        R = _posterior(A, logw, B, d_train)[1]
        ll = _logp(A, B, d_test)[d_test.cell_t, :, d_test.y]
        return logsumexp(ll + np.log(np.clip(R[d_test.unit], 1e-300, None)), axis=1).mean()


def _draw(P, rng):
    return (P.cumsum(1) > rng.random(len(P))[:, None]).argmax(1)


def _optimise(f, x0, maxiter):
    r = minimize(f, x0, jac=True, method="L-BFGS-B", options={"maxiter": maxiter, "ftol": 1e-10, "gtol": 1e-6})
    return r.x, -r.fun


# ------------------------------------------------------------------------------------------------------- LCA
def fit_lca(d, K, n_starts=5, rng=None, maxiter=3000, init_B=None):
    rng = np.random.default_rng(0) if rng is None else rng
    C, G, p = d.C, d.G, d.p
    na, ne = K * (C - 1), G * (K - 1)

    def unpack(th):
        al = np.concatenate([np.zeros((K, 1)), th[:na].reshape(K, C - 1)], 1)
        eta = np.concatenate([np.zeros((G, 1)), th[na:na + ne].reshape(G, K - 1)], 1)
        B = np.concatenate([np.zeros((p, 1)), th[na + ne:].reshape(p, C - 1)], 1)
        return np.broadcast_to(al, (G, K, C)), log_softmax(eta, axis=1), B

    def f(th):
        A, logw, B = unpack(th)
        LL, _, dA, dB, dW = _mix(A, logw, B, d)
        g = np.concatenate([dA.sum(0)[:, 1:].ravel(), dW[:, 1:].ravel(), dB[:, 1:].ravel()])
        pen = th.copy(); pen[na:na + ne] = 0
        return -(LL - .5 * RIDGE * pen @ pen), -(g - RIDGE * pen)

    # the observed answer mix, so every start sits near the data
    base = np.log(np.bincount(d.y, minlength=C) + .5); base = base[1:] - base[0]
    B0 = np.zeros(p * (C - 1)) if init_B is None else init_B
    best = None
    for s in range(n_starts if K > 1 else 1):
        al = np.tile(base, K) + (rng.normal(0, 1.5, na) if K > 1 else 0)
        x, LL = _optimise(f, np.concatenate([al, np.zeros(ne), B0]), maxiter)
        if best is None or LL > best[1]:
            best = (x, LL)
    x = best[0]
    A, logw, B = unpack(x)
    LL = _posterior(A, logw, B, d)[0]
    # order the classes by P(correct), highest first, so class 1 is always the most accurate
    pc = softmax(A[0], axis=1)[:, 0]; o = np.argsort(-pc)
    al = x[:na].reshape(K, C - 1)[o]; eta = np.concatenate([np.zeros((G, 1)), x[na:na + ne].reshape(G, K - 1)], 1)[:, o]
    x = np.concatenate([al.ravel(), (eta[:, 1:] - eta[:, :1]).ravel(), x[na + ne:]])
    fit = Fit("lca", K, x, LL, d, unpack, na + ne + p * (C - 1))
    return fit


# ------------------------------------------------------------------------------------------------------- IRT
def fit_irt(d, Q=15, ref_group=0, maxiter=3000, init_B=None):
    C, G, p = d.C, d.G, d.p
    xq, wq = np.polynomial.hermite_e.hermegauss(Q); wq = wq / wq.sum()
    others = [g for g in range(G) if g != ref_group]

    def split(th):
        a, c = th[:C - 1], th[C - 1:2 * (C - 1)]
        mu, ls = np.zeros(G), np.zeros(G)
        mu[others] = th[2 * (C - 1):2 * (C - 1) + G - 1]; ls[others] = th[2 * (C - 1) + G - 1:2 * (C - 1) + 2 * (G - 1)]
        B = np.concatenate([np.zeros((p, 1)), th[2 * (C - 1) + 2 * (G - 1):].reshape(p, C - 1)], 1)
        return a, c, mu, np.exp(ls), B

    def unpack(th):
        a, c, mu, sig, B = split(th)
        theta = mu[:, None] + sig[:, None] * xq[None, :]                        # (G, Q)
        A = np.concatenate([np.zeros((G, Q, 1)), theta[..., None] * a + c], 2)
        return A, np.broadcast_to(np.log(wq), (G, Q)), B

    def f(th):
        a, c, mu, sig, B = split(th)
        A, logw, _ = unpack(th)
        LL, _, dA, dB, _ = _mix(A, logw, B, d)
        theta = mu[:, None] + sig[:, None] * xq[None, :]
        dAy = dA[..., 1:]
        dth = dAy @ a                                                          # (G, Q)
        g = np.concatenate([np.einsum("gqy,gq->y", dAy, theta), dAy.sum((0, 1)),
                            dth.sum(1)[others], (dth * xq * sig[:, None]).sum(1)[others], dB[:, 1:].ravel()])
        pen = th.copy(); pen[2 * (C - 1):2 * (C - 1) + 2 * (G - 1)] = 0
        return -(LL - .5 * RIDGE * pen @ pen), -(g - RIDGE * pen)

    base = np.log(np.bincount(d.y, minlength=C) + .5); base = base[1:] - base[0]
    B0 = np.zeros(p * (C - 1)) if init_B is None else init_B
    x, _ = _optimise(f, np.concatenate([-np.ones(C - 1), base, np.zeros(2 * (G - 1)), B0]), maxiter)
    A, logw, B = unpack(x)
    fit = Fit("irt", Q, x, _posterior(A, logw, B, d)[0], d, unpack, 2 * (C - 1) + 2 * (G - 1) + p * (C - 1))
    fit.ability = split(x)[:4]
    return fit
