"""Truncated, corrupt or malformed input raises; partial rows are never returned."""
import gzip
import random

import pytest

epirefine_rs = pytest.importorskip("epirefine._kernels", reason="no compiled kernels in this install")
from epirefine import sites
from epirefine.sites import split_pileup

from kernels_reference import MOTIF_SETS, bgzf, make_assembly, make_cfg, pileup_text


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("err")
    rng = random.Random(4)
    contigs = make_assembly(rng, MOTIF_SETS["degenerate"], n_contigs=20)
    cfg = make_cfg(contigs, MOTIF_SETS["degenerate"], tmp / "asm")
    text = pileup_text(rng, contigs)
    asm = sites.EncodedAssembly(cfg["asm_dir"])
    kw = dict(names=asm.names, offsets=asm.offsets, lengths=asm.lengths, codes=asm.codes, motifs=cfg["motifs"],
              group_of=cfg["group_of"], motif_rank=cfg["motif_rank"])
    return tmp, text, kw


def read(path, start, end, kw, **more):
    return epirefine_rs.PileupReader(path, start, end, **kw, **more).read_all()


def n_rows(t):
    return len(t["contig"])


def test_bgzf_truncated_or_corrupt(data):
    tmp, text, kw = data
    good = bgzf(text, 4096)
    f = tmp / "good.bed.gz"
    f.write_bytes(good)
    ref = read(f, 0, -1, kw)
    assert n_rows(ref) > 100
    size = len(good)
    block2 = int.from_bytes(good[16:18], "little") + 1      # offset of the second block
    cases = {
        "cut inside a block": good[: size // 2],
        "cut inside the last data block": good[: size - 40],
        "cut inside a header": good[: block2 + 7],
        "EOF block missing": good[: size - 28],
        "payload byte changed": good[: block2 + 40] + bytes([good[block2 + 40] ^ 0x55]) + good[block2 + 41:],
        "CRC changed": good[: block2 - 8] + bytes([good[block2 - 8] ^ 1]) + good[block2 - 7:],
        "ISIZE changed": good[: block2 - 4] + bytes([good[block2 - 4] ^ 1]) + good[block2 - 3:],
        "header damaged": good[: block2 + 12] + b"XX" + good[block2 + 14:],
        "garbage appended": good + b"some trailing bytes that are not a block",
    }
    for name, blob in cases.items():
        bad = tmp / "bad.bed.gz"
        bad.write_bytes(blob)
        for start, end in ((0, -1), (0, len(blob) << 16)):
            r = epirefine_rs.PileupReader(bad, start, end, **kw)
            with pytest.raises(OSError):
                r.read_all()
            with pytest.raises(RuntimeError):   # a failed reader stays failed
                r.next_batch()
            with pytest.raises(OSError):        # also when the rows are taken in small batches
                for _ in epirefine_rs.PileupReader(bad, start, end, **kw).batches(1):
                    pass
    # a file without the EOF block is read on request (some writers omit it)
    bad.write_bytes(good[: size - 28])
    got = read(bad, 0, -1, kw, check_eof=False)
    assert all((got[k] == ref[k]).all() for k in ref)


def test_bgzf_range_errors(data):
    tmp, text, kw = data
    blob = bgzf(text, 1000)
    f = tmp / "r.bed.gz"
    f.write_bytes(blob)
    ranges = split_pileup(f, 4, min_size=0)
    assert len(ranges) > 2
    a, b = ranges[1]
    assert n_rows(read(f, a, b, kw)) > 0
    with pytest.raises(OSError):        # the range points into the middle of a block
        read(f, ((a >> 16) + 3) << 16, b, kw)
    with pytest.raises((OSError, ValueError)):   # the end is not a block start
        read(f, a, ((b >> 16) + 5) << 16, kw)
    with pytest.raises(OSError):        # the file is shorter than the range says
        f2 = tmp / "r2.bed.gz"
        f2.write_bytes(blob[: (b >> 16) - 10])
        read(f2, a, b, kw)
    with pytest.raises(OSError):
        read(tmp / "missing.bed.gz", 0, -1, kw)


def test_gzip_stream_truncated_or_corrupt(data):
    tmp, text, kw = data
    blob = gzip.compress(text)
    f = tmp / "s.bed.gz"
    f.write_bytes(blob)
    assert n_rows(read(f, 0, -1, kw)) > 100
    for bad in (blob[: len(blob) // 2], blob[:-4], blob[:-8] + bytes([blob[-8] ^ 1]) + blob[-7:]):
        f.write_bytes(bad)
        with pytest.raises(OSError):
            read(f, 0, -1, kw)


def test_plain_range_beyond_file(data):
    tmp, text, kw = data
    f = tmp / "p.bed"
    f.write_bytes(text)
    assert n_rows(read(f, 0, len(text), kw)) > 100
    with pytest.raises(OSError):
        read(f, 0, len(text) + 10, kw)


def row(c, p, nv="5", nm="1", pos=None):
    return f"{c}\t{p if pos is None else pos}\t{p + 1}\ta\t{nv}\t+\t{p}\t{p + 1}\t0\t{nv}\t0.5\t{nm}\t0\t0\t0\t0\t0\t0\n"


def test_malformed_rows_and_contiguity(tmp_path):
    contigs = {"c1": "ACGATCGATCGGATCAAGATCTT" * 3, "c2": "TTGATCAAGATCGG" * 3, "c3": "GATC"}
    cfg = make_cfg(contigs, [("GATC", 1, "a")], tmp_path / "asm")
    asm = sites.EncodedAssembly(cfg["asm_dir"])
    kw = dict(names=asm.names, offsets=asm.offsets, lengths=asm.lengths, codes=asm.codes, motifs=cfg["motifs"],
              group_of=cfg["group_of"], motif_rank=cfg["motif_rank"])
    f = tmp_path / "p.bed"

    def get(text):
        f.write_text(text)
        return read(f, 0, -1, kw)

    assert n_rows(get(row("c1", 3) + row("c2", 3))) == 2
    # rows of a contig in two places: the Python pass raises when the two runs fall into different
    # blocks; the kernel always does
    with pytest.raises(ValueError, match="not contiguous"):
        get(row("c1", 3) + row("c2", 3) + row("c1", 7))
    with pytest.raises(ValueError, match="not contiguous"):
        get(row("zz", 3) + row("c2", 3) + row("zz", 7))
    with pytest.raises(ValueError, match="not contiguous"):
        f.write_text(("".join(row("c1", p) for p in range(40)) + row("c3", 1)) * 2)
        sites._run_range(cfg, str(f), 0, -1, str(tmp_path / "py"), 600, 10 ** 9)
    # counts that are not integers, on a site row
    for bad in (row("c1", 3, nv="x"), row("c1", 3, nv="+4"), row("c1", 3, nm="1e3"), row("c1", 3, pos="3.0")):
        with pytest.raises(ValueError, match="not an integer"):
            get(bad)
    with pytest.raises(OverflowError):
        get(row("c1", 3, nv=str(2 ** 31)))
    with pytest.raises(OverflowError):
        get(row("c1", 3, nv=str(2 ** 31 - 1)) * 2)
    with pytest.raises(ValueError):
        get("\t3\t4\ta\n")
    # negative or missing values drop the row, as in the Python pass
    assert n_rows(get(row("c1", 3, nv="-4") + row("c1", 7, nv="") + row("c1", 3, pos="-1"))) == 0


def test_argument_errors(data):
    tmp, text, kw = data
    f = tmp / "p.bed"
    f.write_bytes(text)
    for change in ({"group_of": kw["group_of"][:-1]}, {"motif_rank": []}, {"lengths": kw["lengths"][:-1]},
                   {"motifs": [("GAXC", 1, "a")] * len(kw["motifs"])}, {"motifs": [("GATC", 9, "a")] * len(kw["motifs"])},
                   {"lengths": kw["lengths"] * 1000}):
        with pytest.raises(ValueError):
            read(f, 0, -1, {**kw, **change})


def test_supports():
    assert epirefine_rs.supports("pileup_reader") and epirefine_rs.supports("motif_scan") and epirefine_rs.supports("bgzf")
    assert epirefine_rs.supports("api>=1") and not epirefine_rs.supports("api>=99")
    assert not epirefine_rs.supports("no_such_feature")
    assert epirefine_rs.__commit__ is None or len(epirefine_rs.__commit__) >= 40
