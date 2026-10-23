"""Micro-benchmarks of the proposed speed-ups (docs/scalability.md section 3). Run: python dev/speedup_prototypes.py"""
import time, numpy as np
from scipy.special import logsumexp
rng = np.random.default_rng(0)
def T(label, f, n=1):
    t = time.time()
    for _ in range(n): r = f()
    dt = (time.time() - t) / n
    print(f"{label:<55} {dt:9.3f} s", flush=True); return dt

# 1. motif grouping: one pairwise intersection of two motifs' record sets (5e6 each, as on 1.7 Gb)
a = np.unique(rng.integers(0, 4e9, 5_000_000)); b = np.unique(rng.integers(0, 4e9, 5_000_000))
t_int = T("intersect1d 5M x 5M (one motif pair)", lambda: np.intersect1d(a, b, assume_unique=True))
print(f"  -> ~250k same-mod-type pairs at M=1000: {t_int*250_000/3600:.1f} h")

# 2. population prior EM: current loop vs vectorised across groups
M, H, G = 800, 400, 64
ell = -rng.gamma(2, 50, (H, M, G)); ell -= ell.max(-1, keepdims=True)
def current(iters=300, mlim=40):
    for m in range(mlim):
        pi = np.full(G, 1 / G)
        for _ in range(iters):
            lp = np.log(pi)[None] + ell[:, m]
            post = np.exp(lp - logsumexp(lp, axis=1, keepdims=True)); pi = np.maximum(post.mean(0), 1e-300)
t_cur = T("prior EM current (40 of 800 groups, 300 it)", current) * M / 40
print(f"  -> current, all {M} groups: {t_cur:.0f} s")
lik = np.exp(ell).transpose(1, 0, 2)  # M,H,G likelihood (rows max 1)
def vect(iters=300):
    pi = np.full((M, G), 1 / G)
    for _ in range(iters):
        denom = np.einsum("mhg,mg->mh", lik, pi)
        pi = pi * np.einsum("mhg,mh->mg", lik, 1 / denom) / H
    return pi
t_vec = T("prior EM vectorised over groups (300 it)", vect)
t_vec50 = T("prior EM vectorised, 50 it (convergence-checked)", lambda: vect(50))

# 3. motif search: regex vs 2-bit shape matching, on 100 Mb random seq, 20 contiguous 5-mer motifs of one shape
import re
seq = "".join(rng.choice(list("ACGT"), 100_000_000))
mot = ["".join(rng.choice(list("ACGT"), 5)) for _ in range(20)]
t_re = T("regex, 1 motif, 100 Mb, both strands", lambda: [m.start() for p in (mot[0], mot[0][::-1]) for m in re.finditer("(?=" + p + ")", seq)])
codes = np.frombuffer(seq.encode(), np.uint8); lut = np.zeros(256, np.uint8); lut[[65, 67, 71, 84]] = [0, 1, 2, 3]
enc = lut[codes]
def shape_match():
    k = np.zeros(len(enc) - 4, np.uint32)
    for j in range(5): k = (k << 2) | enc[j:len(enc) - 4 + j]
    want = np.array([sum(lut[ord(c)] << (2 * (4 - i)) for i, c in enumerate(m)) for m in mot], np.uint32)
    return np.flatnonzero(np.isin(k, want))
t_shape = T("2-bit shape match, 20 motifs at once, 100 Mb (+ strand)", shape_match)
print(f"  -> per motif-strand: regex {t_re/2:.2f}s vs shape {t_shape/20:.3f}s  ({(t_re/2)/(t_shape/20):.0f}x)")

# 4. unit likelihood per site: logaddexp on rows x G (current) vs lookup-table gather
n = 2_000_000; x = np.linspace(0.001, 0.999, G); lx, l1x = np.log(x), np.log1p(-x)
la, lb = rng.normal(-5, 2, n), rng.normal(-5, 2, n)
t_lae = T("logaddexp per site x 64 (2M sites, float64)", lambda: np.logaddexp(la[:, None] + lx, lb[:, None] + l1x))
tab = rng.normal(size=(5000, G)).astype(np.float32); idx = rng.integers(0, 5000, n)
t_gat = T("table gather per site x 64 (2M sites, float32)", lambda: tab[idx])
