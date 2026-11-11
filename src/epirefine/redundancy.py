"""Sequence redundancy between a contig and a bin, from FracMinHash sketches of canonical k-mers.

Methylation can't tell a genome from a strain variant of it (same MTases, same methylome), so an
unbinned contig that duplicates sequence already in a bin (a strain variant, a haplotig) looks like
a perfect member. Including it adds duplicated marker genes, i.e. contamination by CheckM's
measure, without adding completeness (dev log D21). The containment of the contig's k-mers in the
bin's k-mers measures that: about 0.81 at 99% identity, 0.53 at 97% and 0.34 at 95% for k = 21.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

K = 21
SCALE = 50  # keep k-mers whose hash falls in the lowest 1/SCALE of the range
_MASK = (1 << (2 * K)) - 1
_MAXHASH = np.uint64((1 << 64) // SCALE)


def _hash(x: np.ndarray) -> np.ndarray:
    """splitmix64 finaliser: a cheap, well-mixed 64-bit hash."""
    x = x.astype(np.uint64)
    with np.errstate(over="ignore"):
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        x = x ^ (x >> np.uint64(31))
    return x


def sketch(codes: np.ndarray) -> np.ndarray:
    """Sorted unique hashes of the sampled canonical k-mers of an encoded sequence (codes 0-3, 4 = N)."""
    n = len(codes) - K + 1
    if n <= 0:
        return np.empty(0, np.uint64)
    c = codes.astype(np.uint64)
    fwd = np.zeros(n, np.uint64)
    rev = np.zeros(n, np.uint64)
    bad = np.zeros(n, bool)
    for j in range(K):
        x = c[j:j + n]
        fwd = (fwd << np.uint64(2)) | (x & np.uint64(3))
        rev |= (np.uint64(3) - (x & np.uint64(3))) << np.uint64(2 * j)  # reverse complement, read backwards
        bad |= codes[j:j + n] == 4
    h = _hash(np.minimum(fwd, rev))
    return np.unique(h[(h < _MAXHASH) & ~bad])


def containment(query: np.ndarray, target: np.ndarray) -> float:
    """Share of the query's sampled k-mers that occur in the target sketch."""
    if len(query) == 0:
        return 0.0
    return float(np.isin(query, target, assume_unique=True).mean())


def inclusion_redundancy(asm, candidates: list[tuple[int, int]], members: dict[int, list[int]],
                         priority: np.ndarray | None = None, max_redundancy: float = 1.0) -> np.ndarray:
    """Containment of each (contig id, bin) candidate in that bin's current members' sequence.

    priority[i] (higher first) and max_redundancy < 1: the candidates of a bin are taken in that order and
    each one accepted (containment below max_redundancy) joins the bin's k-mers before the next is
    compared, so two unbinned copies of one sequence (haplotigs, a plasmid assembled twice) are not both
    included. Without priority every candidate is compared with the given members only.
    """
    out = np.zeros(len(candidates))
    by_bin: dict[int, list[int]] = {}
    for i, (_, b) in enumerate(candidates):
        by_bin.setdefault(b, []).append(i)

    def work(item):  # one bin at a time; numpy releases the GIL, so bins run on threads
        b, idx = item
        bin_sk = np.unique(np.concatenate([sketch(np.asarray(asm.seq(c))) for c in members.get(b, [])] or
                                          [np.empty(0, np.uint64)]))
        if priority is None:
            for i in idx:
                out[i] = containment(sketch(np.asarray(asm.seq(candidates[i][0]))), bin_sk)
            return
        for i in sorted(idx, key=lambda i: (-priority[i], candidates[i][0])):
            sk = sketch(np.asarray(asm.seq(candidates[i][0])))
            out[i] = containment(sk, bin_sk)
            if out[i] < max_redundancy:
                bin_sk = np.union1d(bin_sk, sk)

    with ThreadPoolExecutor(min(16, os.cpu_count() or 1)) as ex:
        list(ex.map(work, by_bin.items()))
    return out
