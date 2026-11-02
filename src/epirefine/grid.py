"""Discrete grid over phi = fraction of a motif's sites that are methylated.

The grid is uniform in logit space so the regions near 0 (motif absent) and 1 (motif present),
where nearly all hosts sit, are resolved finely. Densities are represented as probability mass
per grid cell, computed from CDF differences, so the grid spacing does not bias anything.
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit, logit
from scipy.stats import beta, norm


class Grid:
    def __init__(self, size: int = 64, lo: float = 1e-3, kernel: str = "beta"):
        z = np.linspace(logit(lo), logit(1 - lo), size)
        self.x = expit(z)
        self.edges = np.concatenate([[0.0], expit((z[:-1] + z[1:]) / 2), [1.0]])
        self.logx = np.log(self.x)
        self.log1mx = np.log1p(-self.x)
        self.size = size
        self._trans_cache: dict[float, np.ndarray] = {}
        if kernel not in ("beta", "logit"):
            raise ValueError(f"unknown variation kernel {kernel!r}")
        self.kernel = kernel
        self.no_variation = float("inf") if kernel == "beta" else 0.0  # the parameter meaning "unit = host"

    def beta_mass(self, a, b) -> np.ndarray:
        m = np.maximum(np.diff(beta.cdf(self.edges, a, b)), 1e-300)
        return m / m.sum()

    def is_point(self, param) -> bool:
        """True when the variation parameter means no between-unit variation."""
        if param is None:
            return True
        return (not np.isfinite(param)) if self.kernel == "beta" else param <= 0

    def transition(self, param: float | None, delta: float = 1.0) -> np.ndarray:
        """T[i, j] = P(phi_unit in cell j | phi_host = x_i), with the host level scaled by delta
        (plasmid under-methylation).

        kernel "beta":  phi_unit ~ Beta(d x_i tau, (1 - d x_i) tau); param = tau, inf = no variation.
        kernel "logit": logit(phi_unit) ~ Normal(logit(d x_i), sigma^2); param = sigma, 0 = no
                        variation. Variation is on the log-odds scale, so a level near 0 or 1 barely
                        moves while intermediate levels vary most (dev log BACKLOG item 1).
        """
        key = (param, delta)
        if key not in self._trans_cache:
            mean = delta * self.x
            if self.is_point(param):
                T = np.zeros((self.size, self.size))
                T[np.arange(self.size), np.clip(np.searchsorted(self.edges, mean) - 1, 0, self.size - 1)] = 1.0
            elif self.kernel == "beta":
                T = np.stack([self.beta_mass(m * param, (1 - m) * param) for m in mean])
            else:
                with np.errstate(divide="ignore"):
                    ze = logit(self.edges)  # -inf .. inf
                mz = logit(np.clip(mean, 1e-12, 1 - 1e-12))
                cdf = norm.cdf((ze[None, :] - mz[:, None]) / param)
                T = np.maximum(np.diff(cdf, axis=1), 1e-300)
                T /= T.sum(1, keepdims=True)
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
