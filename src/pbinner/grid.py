"""Discrete grid over phi = fraction of a motif's sites that are methylated.

The grid is uniform in logit space so the regions near 0 (motif absent) and 1 (motif present),
where nearly all hosts sit, are resolved finely. Densities are represented as probability mass
per grid cell, computed from CDF differences, so the grid spacing does not bias anything.
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit, logit
from scipy.stats import beta


class Grid:
    def __init__(self, size: int = 64, lo: float = 1e-3):
        z = np.linspace(logit(lo), logit(1 - lo), size)
        self.x = expit(z)
        self.edges = np.concatenate([[0.0], expit((z[:-1] + z[1:]) / 2), [1.0]])
        self.logx = np.log(self.x)
        self.log1mx = np.log1p(-self.x)
        self.size = size
        self._trans_cache: dict[float, np.ndarray] = {}

    def beta_mass(self, a, b) -> np.ndarray:
        m = np.maximum(np.diff(beta.cdf(self.edges, a, b)), 1e-300)
        return m / m.sum()

    def transition(self, tau: float | None, delta: float = 1.0) -> np.ndarray:
        """T[i, j] = P(phi_unit in cell j | phi_host = x_i): phi_unit ~ Beta(d x_i tau, (1 - d x_i) tau).

        delta scales the host level (plasmid under-methylation). tau = None/inf means no
        between-unit variation (point mass at delta * x_i).
        """
        key = (tau, delta)
        if key not in self._trans_cache:
            mean = delta * self.x
            if tau is None or not np.isfinite(tau):
                T = np.zeros((self.size, self.size))
                T[np.arange(self.size), np.clip(np.searchsorted(self.edges, mean) - 1, 0, self.size - 1)] = 1.0
            else:
                T = np.stack([self.beta_mass(m * tau, (1 - m) * tau) for m in mean])
            self._trans_cache[key] = T
        return self._trans_cache[key]

    def summarize(self, logp: np.ndarray):
        """Mean, 2.5% and 97.5% quantiles, and P(phi > 0.5) of densities given as log cell masses (last axis)."""
        p = np.exp(logp - logp.max(-1, keepdims=True))
        p /= p.sum(-1, keepdims=True)
        mean = (p * self.x).sum(-1)
        cdf = np.cumsum(p, -1)
        lo = self.x[np.clip((cdf < 0.025).sum(-1), 0, self.size - 1)]
        hi = self.x[np.clip((cdf < 0.975).sum(-1), 0, self.size - 1)]
        p_present = p[..., self.x > 0.5].sum(-1)
        return mean, lo, hi, p_present
