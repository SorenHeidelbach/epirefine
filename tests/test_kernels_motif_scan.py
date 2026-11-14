"""The motif scan and the site records equal EpiRefine's MotifMatcher and _Worker.records."""
import random

import numpy as np
import pytest

epirefine_rs = pytest.importorskip("epirefine._kernels", reason="no compiled kernels in this install")
from epirefine import sites
from epirefine.motifs import Motif, MotifMatcher, encode

from kernels_reference import MOTIF_SETS, make_assembly, make_cfg, random_seq

DTYPES = {"pos": np.int64, "strand": np.int8, "motif": np.int32, "occ_motif": np.int32, "occ": np.int64}


def same_records(a, b):
    assert list(a) == list(b) == list(DTYPES)
    for k, dt in DTYPES.items():
        assert a[k].dtype == b[k].dtype == dt, k
        assert np.array_equal(a[k], b[k]), k


@pytest.mark.parametrize("name", list(MOTIF_SETS))
def test_find_equals_motifmatcher(name):
    rng = random.Random(name)
    motifs = [Motif(*m) for m in MOTIF_SETS[name]]
    py, rs = MotifMatcher(motifs), epirefine_rs.MotifScanner(motifs)
    assert rs.n_shapes == len(py.shapes)
    plant = [m.seq for m in motifs]
    total = 0
    for n in (0, 1, 3, 4, 5, 11, 12, 13, 22, 100, 4095, 4096, 4097, 4099, 8192 + 5, 50_000):
        seq = random_seq(rng, n, plant)
        codes = encode(seq) if n else np.empty(0, np.uint8)
        a, b = py.find(codes), rs.find(codes)
        same_records(a, b)
        total += len(a["pos"])
    assert total > 100


def test_find_overlapping_and_ambiguous_bases():
    motifs = [Motif("AAA", 1, "a"), Motif("ANA", 0, "a"), Motif("CCC", 0, "m"), Motif("GGG", 2, "m")]
    py, rs = MotifMatcher(motifs), epirefine_rs.MotifScanner(motifs)
    for seq in ("AAAAAAA", "AANAAAnAAAA", "CCCCCGGGGG", "aaaAAAcccGGG", "NNNN", "AAA", "AA"):
        same_records(py.find(encode(seq)), rs.find(encode(seq)))
    assert len(rs.find(encode("AAAAAAA"))["pos"]) == 10  # 5 overlapping starts of AAA and of ANA, on +


def test_find_contigs_equals_concatenation():
    rng = random.Random(5)
    motifs = [Motif(*m) for m in MOTIF_SETS["degenerate"]]
    contigs = make_assembly(rng, MOTIF_SETS["degenerate"], n_contigs=40)
    asm_codes = np.concatenate([encode(s) for s in contigs.values()])
    lengths = np.array([len(s) for s in contigs.values()], np.int64)
    offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    py, rs = MotifMatcher(motifs), epirefine_rs.MotifScanner(motifs)
    ids = [7, 0, 39, 12, 13, 3]
    # Python: as _Worker.records and find_many do, on the contigs joined by separators
    sep = max(len(m.seq) for m in motifs)
    seqs = [asm_codes[offsets[i]:offsets[i] + lengths[i]] for i in ids]
    offs = np.cumsum([0] + [len(s) + sep for s in seqs])[:-1]
    cat = np.full(int(offs[-1] + len(seqs[-1])), 4, np.uint8)
    for o, s in zip(offs, seqs):
        cat[o:o + len(s)] = s
    rec = py.find(cat)
    cidx = np.searchsorted(offs, rec["pos"], side="right") - 1
    rec["pos"] = rec["pos"] - offs[cidx]
    rec["occ"] = rec["occ"] - 2 * offs[cidx]
    got_cidx, got = rs.find_contigs(asm_codes, offsets, lengths, ids)
    assert np.array_equal(got_cidx, cidx) and len(cidx) > 100
    same_records(rec, got)


@pytest.mark.parametrize("name", list(MOTIF_SETS))
@pytest.mark.parametrize("grouping", ["epirefine", "random"])
def test_site_records_equal_worker_records(tmp_path, name, grouping):
    rng = random.Random(name + grouping)
    contigs = make_assembly(rng, MOTIF_SETS[name], n_contigs=25)
    cfg = make_cfg(contigs, MOTIF_SETS[name], tmp_path / "asm", random_groups=rng if grouping == "random" else None)
    w = sites._Worker(cfg)
    ids = list(range(len(contigs)))
    rng.shuffle(ids)
    py = w.records(ids).sort("key")
    rs = epirefine_rs.site_records(w.asm.codes, w.asm.offsets, w.asm.lengths, ids, cfg["motifs"], cfg["group_of"],
                                   cfg["motif_rank"])
    assert py.height > 0
    assert rs["key"].dtype == np.int64 and rs["group"].dtype == np.int32 and rs["site"].dtype == np.int64
    assert np.array_equal(py["key"].to_numpy(), rs["key"])
    assert np.array_equal(py["group"].to_numpy(), rs["group"])
    assert np.array_equal(py["site"].to_numpy(), rs["site"])


def test_motif_errors():
    with pytest.raises(ValueError):
        epirefine_rs.MotifScanner([("GAXC", 1, "a")])
    with pytest.raises(ValueError):
        epirefine_rs.MotifScanner([("N" * 0, 0, "a")])
    with pytest.raises(ValueError):  # as MotifMatcher: too many concrete sequences
        epirefine_rs.MotifScanner([("NNNNNNNNN".replace("N", "B") + "BB", 0, "a")])
    with pytest.raises(ValueError):
        MotifMatcher([Motif("BBBBBBBBBBB", 0, "a")])
