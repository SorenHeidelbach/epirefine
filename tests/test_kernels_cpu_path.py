"""The CPU path of the model kernels: reported by cpu_path() and features(), forced to "generic" by
EPIREFINE_RS_NO_SIMD, and the same bits on both paths (the AVX2+FMA bodies do the arithmetic of the generic ones).
Each path runs in its own process, because the path is fixed at the first kernel call."""
import os
import subprocess
import sys

import numpy as np
import pytest

rs = pytest.importorskip("epirefine._kernels", reason="no compiled kernels in this install")

pytestmark = pytest.mark.skipif(not rs.supports("model_kernels"), reason="build without model kernels")

SCRIPT = r"""
import sys
import numpy as np
from epirefine import _kernels as rs

rng = np.random.default_rng(7)
U, M, G, H = 90, 7, 64, 5
lp = -rng.gamma(2.0, 40.0, (U, M, G))
lp[rng.random((U, M)) < 0.2] = 0.0
lp = lp.astype(np.float32)
cb = rng.integers(0, H, U)
w = rng.random(U)
log_pi = np.log(rng.dirichlet(np.ones(G), M))
ell = rs.bin_loglik(lp, cb, H, w, threads=2)
lik = np.exp(rng.normal(-3, 1, (M, H, G)))
pi = rng.dirichlet(np.ones(G), M)
e = rng.random((U, M, G))
eb = rng.random((M, G, H))
am, bm, le0 = rng.normal(size=(U, M)), rng.normal(size=(M, H)), rng.normal(size=(U, M))
s, r = rs.score_sums(e, am, eb, bm, le1=-0.5, le0=le0, threads=2)
out = dict(ell=ell, comm=rs.community_terms(lp, log_pi, threads=2),
           lco=rs.leave_chunk_out_terms(lp, cb, w, ell, log_pi, threads=2),
           npmle=rs.npmle(lik, pi, iters=50, tol=1e-9, threads=2), s=s, r=r,
           cells=rs.cell_sums(rng.random((40, G)), rng.integers(0, 40, 300), rng.random(300), np.array([0, 10, 150, 299])))
np.savez(sys.argv[1], path=np.array(rs.cpu_path()), features=np.array(rs.features()), **out)
"""


def run(tmp_path, name, no_simd):
    env = dict(os.environ)
    env.pop("EPIREFINE_RS_NO_SIMD", None)
    if no_simd is not None:
        env["EPIREFINE_RS_NO_SIMD"] = no_simd
    f = tmp_path / f"{name}.npz"
    subprocess.run([sys.executable, "-c", SCRIPT, str(f)], env=env, check=True)
    return dict(np.load(f))


def test_cpu_path_reported():
    path = rs.cpu_path()
    assert path in ("avx2_fma", "generic")
    assert f"cpu:{path}" in rs.features() and rs.supports(f"cpu:{path}")


def test_forced_generic_path_gives_the_same_bits(tmp_path):
    default = run(tmp_path, "default", None)
    generic = run(tmp_path, "generic", "1")
    assert str(generic["path"]) == "generic" and "cpu:generic" in list(generic["features"])
    assert str(run(tmp_path, "off", "0")["path"]) == str(default["path"])  # "0" does not force anything
    for k in ("ell", "comm", "lco", "npmle", "s", "r", "cells"):
        assert default[k].dtype == generic[k].dtype and np.array_equal(default[k].view(np.uint64), generic[k].view(np.uint64)), k
