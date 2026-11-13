"""The compiled kernels of EpiRefine: the pileup reader, the motif scan and the model kernels.

The extension `_core` is built from `rust/` into the platform wheels (D73); the pure wheel and a source tree
without a build have no `_core`, and importing this package then raises ImportError. EpiRefine imports it only
through `epirefine._rs`, which falls back to the numpy code and says why. Everything here takes and returns plain
Python values and numpy arrays and imports nothing else of EpiRefine. EpiRefine's own Python code is the
reference: for the same input the kernels return the same rows (and the model kernels the same values to the last
bits of exp, log and sums).

    PileupReader     one range of a modkit pileup -> site rows (contig, group, pos, n, k)
    MotifScanner     the records of epirefine.motifs.MotifMatcher
    site_records     the deduplicated (key, group, site) records of epirefine.sites._Worker.records
    supports(name)   whether this build provides a feature
"""
from __future__ import annotations

import os
from typing import Iterator, Sequence

import numpy as np

from . import _core
from ._core import __commit__  # the commit the extension was built from (None: built from an sdist)

__all__ = ["PileupReader", "MotifScanner", "site_records", "bin_loglik", "cell_sums", "score_sums", "npmle", "community_terms", "leave_chunk_out_terms", "supports", "features", "cpu_path", "SITE_COLUMNS", "API", "__commit__"]

#: columns of a batch, in order, with their dtypes (EpiRefine's SITE_SCHEMA)
SITE_COLUMNS = (("contig", np.int32), ("group", np.int32), ("pos", np.int64), ("n", np.int32), ("k", np.int32))

#: interface version of this module; raised when a function changes meaning, so that EpiRefine's Python code can
#: require what it was written for (supports("api>=1")); a stale build in an editable install fails that check
API = 1


def features() -> list[str]:
    """Names accepted by supports()."""
    return sorted(_core.features())


def cpu_path() -> str:
    """The CPU path of the model kernels in this process: "avx2_fma" (x86-64 CPUs with AVX2 and FMA) or "generic"
    (the build for the platform's baseline instruction set). Both do the same arithmetic and give the same bits;
    EPIREFINE_RS_NO_SIMD=1, set before the first kernel call, forces "generic". features() lists it as "cpu:<path>"."""
    return _core.cpu_path()


def supports(feature: str) -> bool:
    """True if this build provides `feature`.

    Features: "pileup_reader" (motif matching in the kernel), "pileup_reader_keys" (site keys given
    by the caller), "motif_scan", "site_records", the inputs "plain", "bgzf", "gzip", and
    "libdeflate" (BGZF blocks inflated by libdeflate rather than the pure-Rust inflater), the model
    kernels "model_kernels", "cell_sums", "score_sums", and the CPU path "cpu:avx2_fma" or "cpu:generic".
    "api>=N" is true when the interface version is at least N.
    """
    if feature.startswith("api>="):
        try:
            return API >= int(feature[5:])
        except ValueError:
            return False
    return feature in _core.features()


def _motifs(motifs) -> list[tuple[str, int, str]]:
    out = []
    for m in motifs:
        if isinstance(m, (tuple, list)):
            seq, mod_pos, mod_type = m
        else:  # an object with the fields of epirefine.motifs.Motif
            seq, mod_pos, mod_type = m.seq, m.mod_pos, m.mod_type
        out.append((str(seq), int(mod_pos), str(mod_type)))
    return out


def _i64(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.int64)


def _codes(codes) -> np.ndarray:
    if not isinstance(codes, np.ndarray) or codes.dtype != np.uint8 or codes.ndim != 1 or not codes.flags.c_contiguous:
        codes = np.ascontiguousarray(codes, dtype=np.uint8)
    return codes


class MotifScanner:
    """IUPAC motif matching on a 2-bit encoded sequence (A, C, G, T = 0..3, anything else 4), both
    strands; the same records, in the same order, as epirefine.motifs.MotifMatcher.

    motifs: (sequence, modified position, modification type) tuples, or objects with the fields
    seq, mod_pos, mod_type.
    """

    def __init__(self, motifs):
        self._s = _core.MotifScanner(_motifs(motifs))

    @property
    def n_shapes(self) -> int:
        return self._s.n_shapes

    def find(self, codes) -> dict[str, np.ndarray]:
        """Records in one encoded sequence, as MotifMatcher.find: pos (int64), strand (int8, 0 '+', 1 '-'),
        motif (int32), occ_motif (int32), occ (int64)."""
        pos, strand, motif, occ_motif, occ = self._s.find(_codes(codes))
        return {"pos": pos, "strand": strand, "motif": motif, "occ_motif": occ_motif, "occ": occ}

    def find_contigs(self, codes, offsets, lengths, contig_ids) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Records of the contigs `contig_ids` of an encoded assembly (contig i is
        codes[offsets[i]:offsets[i] + lengths[i]]), in contig coordinates.

        Returns (cidx, records): cidx is the index into contig_ids of each record. The records and their
        order are those MotifMatcher.find gives on the contigs joined by separators, mapped back to contig
        coordinates (as MotifMatcher.find_many and sites._Worker.records do).
        """
        cidx, (pos, strand, motif, occ_motif, occ) = self._s.find_contigs(
            _codes(codes), _i64(offsets), _i64(lengths), _i64(contig_ids))
        return cidx, {"pos": pos, "strand": strand, "motif": motif, "occ_motif": occ_motif, "occ": occ}


def site_records(codes, offsets, lengths, contig_ids, motifs, group_of, motif_rank) -> dict[str, np.ndarray]:
    """Deduplicated site records of the contigs `contig_ids`, as epirefine.sites._Worker.records:
    key (int64: index into contig_ids << 36 | pos << 3 | strand << 2 | modification code), group (int32)
    and site (int64: index << 50 | canonical motif << 36 | occurrence), ascending by key.

    A key matched by motifs of several groups goes to the group with the fewest records
    (sum of motif_rank), and within a group to the motif with the highest rank.
    """
    key, group, site = _core.site_records(_codes(codes), _i64(offsets), _i64(lengths), _i64(contig_ids),
                                          _motifs(motifs), [int(g) for g in group_of], [int(r) for r in motif_rank])
    return {"key": key, "group": group, "site": site}


class PileupReader:
    """The pileup pass of one range: site rows (contig id, group, pos, n, k) of a modkit pileup.

    path, start, end: as in EpiRefine's sites._run_range. A plain file is read from byte start to byte
    end; a ``.gz`` file with end >= 0 is BGZF and start, end are virtual offsets (as sites.split_pileup
    returns them); end < 0 reads the whole file (plain, BGZF or any gzip). A range must hold whole contigs.

    names, offsets, lengths, codes: the encoded assembly (contig i is codes[offsets[i]:offsets[i] +
    lengths[i]], one byte per base: A, C, G, T = 0..3, anything else 4). `codes` may be a numpy memmap;
    it is not copied. The contig id of a row is the index into `names`.

    motifs, group_of, motif_rank: the motifs (tuples or objects with seq, mod_pos, mod_type), the group of
    each and its record count; wanted: contig ids to keep (None = all).

    check_eof: a BGZF file read to its end must finish with the empty end-of-file block that bgzip writes;
    without it the file is taken to be truncated and OSError is raised.

    Rows: exactly those of sites._Worker._sites for the contigs of the range (see docs/INTEGRATION.md for
    the few malformed inputs where the kernel raises and the Python code does not, or the reverse).
    Errors: OSError for unreadable, truncated or corrupt input; ValueError for malformed rows and for
    contigs whose rows are not contiguous; OverflowError for counts beyond 32 bits. After an error the
    reader is unusable. The GIL is released while reading; one reader uses one thread.
    """

    def __init__(self, path, start: int, end: int, *, names: Sequence[str], offsets, lengths, codes, motifs,
                 group_of, motif_rank, wanted=None, check_eof: bool = True):
        self._r = _core.PileupReader(os.fspath(path), int(start), int(end), list(names), _i64(offsets), _i64(lengths),
                                     _codes(codes), _motifs(motifs), [int(g) for g in group_of],
                                     [int(r) for r in motif_rank],
                                     None if wanted is None else sorted(int(i) for i in wanted), bool(check_eof))

    @classmethod
    def with_keys(cls, path, start: int, end: int, *, names: Sequence[str], key_ptr, keys, key_group, key_site,
                  check_eof: bool = True) -> "PileupReader":
        """Reader with the motif-site keys computed by the caller (motif matching stays in Python).

        The keys of contig i are keys[key_ptr[i]:key_ptr[i + 1]], strictly ascending, each
        pos << 3 | strand << 2 | modification code (a=0, m=1, 21839=2, h=3); key_group and key_site give
        the group and a site id (any integer, equal for the keys of one site) of each key. Contigs without
        keys give no rows. All keys are held in memory for the life of the reader.
        """
        self = cls.__new__(cls)
        self._r = _core.PileupReader.with_keys(os.fspath(path), int(start), int(end), list(names), _i64(key_ptr),
                                               _i64(keys), np.ascontiguousarray(key_group, dtype=np.int32),
                                               _i64(key_site), bool(check_eof))
        return self

    def next_batch(self, max_rows: int = 20_000_000) -> dict[str, np.ndarray] | None:
        """The next rows as a dict of arrays (SITE_COLUMNS), or None when the range is exhausted. A batch
        holds whole contigs and at least max_rows rows unless the range ends first; rows are in file order
        of the contigs, unordered within a contig."""
        b = self._r.next_batch(int(max_rows))
        if b is None:
            return None
        return {name: a for (name, _), a in zip(SITE_COLUMNS, b)}

    def batches(self, max_rows: int = 20_000_000) -> Iterator[dict[str, np.ndarray]]:
        while True:
            b = self.next_batch(max_rows)
            if b is None:
                return
            yield b

    def read_all(self) -> dict[str, np.ndarray]:
        """All rows of the range at once (memory grows with the range; for tests and small inputs)."""
        parts = list(self.batches(1 << 62))
        return {name: (np.concatenate([p[name] for p in parts]) if parts else np.empty(0, dt))
                for name, dt in SITE_COLUMNS}

    @property
    def finished(self) -> bool:
        return self._r.finished

    @property
    def contigs(self) -> list[str]:
        """Names of the contigs with rows read so far, in file order (complete once finished)."""
        return self._r.contigs()

    @property
    def stats(self) -> dict:
        """Counters: input kind, text bytes, lines, rows on sites, contigs keyed, keys, seconds spent
        reading/inflating and building keys."""
        return self._r.stats()


# ---- model kernels (feature "model_kernels"): float32 [units, groups, grid] tables, float64 arithmetic


def _table(lp) -> np.ndarray:
    lp = np.asarray(lp)
    if lp.dtype != np.float32 or lp.ndim != 3:
        raise ValueError("the table must be a float32 array [units, groups, grid]")
    return np.ascontiguousarray(lp)


def _threads(threads) -> int:
    return int(threads) if threads else (os.cpu_count() or 1)


def bin_loglik(lp, chunk_bin, n_bins: int, w, threads: int | None = None) -> np.ndarray:
    """ell[h] = sum over the chunks k of bin h of w[k] * lp[k] (float64 [n_bins, groups, grid]); the kernel of
    epirefine.model.bin_loglik. Chunks are added in ascending order, so the result does not depend on `threads`."""
    lp = _table(lp)
    out = _core.bin_loglik(lp, _i64(chunk_bin), np.ascontiguousarray(w, dtype=np.float64), int(n_bins), _threads(threads))
    return out.reshape((int(n_bins),) + lp.shape[1:])


def community_terms(lp, log_pi, threads: int | None = None) -> np.ndarray:
    """Per unit: sum over groups of logsumexp(lp[u, m] + log_pi[m]) (float64 [units])."""
    return _core.community_terms(_table(lp), np.ascontiguousarray(log_pi, dtype=np.float64), _threads(threads))


def leave_chunk_out_terms(lp, chunk_bin, w, ell, log_pi, threads: int | None = None) -> np.ndarray:
    """Per chunk k of bin h: sum over groups of logsumexp(lp[k, m] + prof[m]), where prof is the bin's profile
    without the chunk: a = ell[h] - w[k] * lp[k] + log_pi, prof = a - logsumexp(a) (float64 [chunks])."""
    return _core.leave_chunk_out_terms(_table(lp), _i64(chunk_bin), np.ascontiguousarray(w, dtype=np.float64),
                                       np.ascontiguousarray(ell, dtype=np.float64),
                                       np.ascontiguousarray(log_pi, dtype=np.float64), _threads(threads))


def npmle(lik, pi, iters: int = 300, tol: float = 1e-6, threads: int | None = None) -> np.ndarray:
    """Grid NPMLE by EM: lik[m, b, g] (likelihood of bin b's data in group m at grid point g, any scale per
    (m, b)), pi[m, g] the start. Returns pi[m, g] after the first iteration in which no cell of any group moved
    by `tol` or more (at most `iters`): the loop of epirefine.model.npmle, groups on threads."""
    lik = np.ascontiguousarray(lik, dtype=np.float64)
    pi = np.ascontiguousarray(pi, dtype=np.float64)
    return _core.npmle(lik, pi, int(iters), float(tol), _threads(threads)).reshape(pi.shape)


def cell_sums(table, r, w, starts, threads: int | None = 1) -> np.ndarray:
    """out[c] = sum over the rows i of cell c of w[i] * table[r[i]], cells given by the row number each starts at
    (ascending; a cell ends where the next starts, the last at len(r)). float64 [cells, G]: the bits of
    `np.add.reduceat(table[r] * w[:, None], starts, axis=0)` without the [rows, G] array (feature "cell_sums")."""
    table = np.ascontiguousarray(table, dtype=np.float64)
    out = _core.cell_sums(table, _i64(r), np.ascontiguousarray(w, dtype=np.float64), _i64(starts), _threads(threads))
    return out.reshape(-1, table.shape[1])


def score_sums(e, am, eb, bm, le1: float = 0.0, le0=None, threads: int | None = None):
    """S[u, h] = sum over groups m of A[u, h, m], A = log(max(e[u, m] . eb[m, :, h], 1e-300)) + am[u, m] + bm[m, h],
    for e [units, groups, G], am [units, groups], eb [groups, G, bins], bm [groups, bins] (float64); with le0
    [units, groups] also R[u, h] = sum_m logaddexp(A + le1, le0[u, m]) (else R is None). The sums of
    epirefine.model.summed_scores, the exponentials being the caller's (feature "score_sums")."""
    f = lambda a: np.ascontiguousarray(a, dtype=np.float64)
    e, eb = f(e), f(eb)
    s, r = _core.score_sums(e, f(am), eb, f(bm), _threads(threads), float(le1), None if le0 is None else f(le0))
    shape = (e.shape[0], eb.shape[2])
    return s.reshape(shape), (r.reshape(shape) if le0 is not None else None)
