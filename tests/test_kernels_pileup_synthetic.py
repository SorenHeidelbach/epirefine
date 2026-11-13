"""Synthetic pileups: the kernel and EpiRefine's Python pass give identical site tables."""
import gzip
import random

import numpy as np
import polars as pl
import pytest

epirefine_rs = pytest.importorskip("epirefine._kernels", reason="no compiled kernels in this install")
from epirefine import sites
from epirefine.sites import SITE_SCHEMA, split_pileup

from kernels_reference import (MOTIF_SETS, bgzf, compare_range, make_assembly, make_cfg, n_diff, pileup_text, run_range_keys,
                       run_range_rs, table)


def case(tmp_path, name, seed=0, layout="tab", n_contigs=30, **kw):
    rng = random.Random(f"{name}{seed}{layout}")
    contigs = make_assembly(rng, MOTIF_SETS[name], n_contigs=n_contigs)
    cfg = make_cfg(contigs, MOTIF_SETS[name], tmp_path / "asm", **kw)
    order = list(contigs)
    rng.shuffle(order)          # pileup order differs from assembly order
    order = order[:-2]          # two assembly contigs have no rows
    text = pileup_text(rng, contigs, order=order, layout=layout)
    return cfg, text


@pytest.mark.parametrize("name", list(MOTIF_SETS))
@pytest.mark.parametrize("layout", ["tab", "space"])
def test_whole_file_plain_gzip_bgzf(tmp_path, name, layout):
    cfg, text = case(tmp_path, name, layout=layout)
    plain = tmp_path / "p.bed"
    plain.write_bytes(text)
    ref, _ = compare_range(cfg, plain, 0, -1, tmp_path, tag="plain")
    assert ref.height > 50
    assert ref["contig"].n_unique() > 5
    gz = tmp_path / "p_stream.bed.gz"
    with gzip.open(gz, "wb") as fh:
        fh.write(text)
    bg = tmp_path / "p_bgzf.bed.gz"
    bg.write_bytes(bgzf(text, 1000))
    for tag, f in (("gz", gz), ("bgzf", bg)):
        t, _ = compare_range(cfg, f, 0, -1, tmp_path, tag=tag)
        assert t.equals(ref)
    # design (b): keys from EpiRefine's own motif matching, reading and aggregation in the kernel
    assert n_diff(run_range_keys(cfg, str(plain), 0, -1), ref) == 0
    assert n_diff(run_range_keys(cfg, str(bg), 0, -1), ref) == 0


@pytest.mark.parametrize("name", ["degenerate", "palindromic", "non_palindromic"])
@pytest.mark.parametrize("layout", ["tab", "space"])
def test_ranges_plain(tmp_path, name, layout):
    cfg, text = case(tmp_path, name, seed=1, layout=layout, n_contigs=50)
    f = tmp_path / "p.bed"
    f.write_bytes(text)
    whole, _ = compare_range(cfg, f, 0, -1, tmp_path, tag="w")
    for n_parts in (2, 3, 7, 64):
        ranges = split_pileup(f, n_parts, min_size=0)
        assert n_parts == 2 or len(ranges) > 2
        got = [compare_range(cfg, f, a, b, tmp_path, flush_rows=300, tag=f"{n_parts}_{i}")[1]
               for i, (a, b) in enumerate(ranges)]
        assert pl.concat(got).sort(whole.columns).equals(whole)   # the ranges together give the whole file


@pytest.mark.parametrize("block", [97, 1000, 4096, 65280])
@pytest.mark.parametrize("layout", ["tab", "space"])
def test_ranges_bgzf(tmp_path, block, layout):
    # blocks of 97 bytes of text: most lines span two or three blocks; 65280: htslib's block size
    cfg, text = case(tmp_path, "degenerate", seed=block, layout=layout, n_contigs=40 if block > 97 else 12)
    plain = tmp_path / "p.bed"
    plain.write_bytes(text)
    f = tmp_path / "p.bed.gz"
    f.write_bytes(bgzf(text, block))
    whole = table(sites._run_range(cfg, str(plain), 0, -1, str(tmp_path / "ref"), 1 << 24, 10 ** 9)["parts"])
    for n_parts in (2, 7, 64):
        ranges = split_pileup(f, n_parts, min_size=0)
        got = [compare_range(cfg, f, a, b, tmp_path, flush_rows=300, tag=f"{n_parts}_{i}")[1]
               for i, (a, b) in enumerate(ranges)]
        assert pl.concat(got).sort(whole.columns).equals(whole)
        assert len(ranges) >= 2


def test_wanted_and_random_groups(tmp_path):
    rng = random.Random(11)
    for i, wanted in enumerate([None, {0, 3, 4, 17, 29}, set()]):
        cfg, text = case(tmp_path / f"c{i}", "degenerate", seed=i, wanted=wanted, random_groups=rng)
        f = tmp_path / f"p{i}.bed"
        f.write_bytes(text)
        t, _ = compare_range(cfg, f, 0, -1, tmp_path, tag=f"w{i}")
        if wanted is not None:
            assert set(t["contig"].unique().to_list()) <= wanted
        assert (t.height > 0) == (wanted != set())


def test_batches_hold_whole_contigs_and_flush(tmp_path):
    cfg, text = case(tmp_path, "degenerate", seed=3, n_contigs=60)
    f = tmp_path / "p.bed"
    f.write_bytes(text)
    asm = sites.EncodedAssembly(cfg["asm_dir"])
    kw = dict(names=asm.names, offsets=asm.offsets, lengths=asm.lengths, codes=asm.codes, motifs=cfg["motifs"],
              group_of=cfg["group_of"], motif_rank=cfg["motif_rank"])
    everything = epirefine_rs.PileupReader(f, 0, -1, **kw).read_all()
    for name, dt in epirefine_rs.SITE_COLUMNS:
        assert everything[name].dtype == dt
    r = epirefine_rs.PileupReader(f, 0, -1, **kw)
    batches = list(r.batches(50))
    assert len(batches) >= 2 and r.finished and r.next_batch(50) is None
    seen = set()
    for b in batches:
        c = set(b["contig"].tolist())
        assert not (c & seen)
        seen |= c
    a = pl.DataFrame({k: np.concatenate([b[k] for b in batches]) for k in everything}, schema=SITE_SCHEMA)
    assert n_diff(a, pl.DataFrame(everything, schema=SITE_SCHEMA)) == 0
    st = r.stats
    assert st["input"] == "plain" and st["text_bytes"] == len(text) and st["rows_hit"] > 0


def rows(*specs, sep="\t"):
    out = []
    for c, p, code, strand, nv, nm in specs:
        tail = sep.join(map(str, [nv, 0.5, nm, 0, 0, 0, 0, 0, 0]))
        out.append(f"{c}\t{p}\t{p + 1}\t{code}\t{nv}\t{strand}\t{p}\t{p + 1}\t255,0,0\t{tail}\n")
    return "".join(out)


@pytest.fixture
def small(tmp_path):
    contigs = {"c1": "ACGATCGATCGGATCAAGATCTT" * 3, "c2": "TTGATCAAGATCGG" * 3, "c3": "GATC", "c4": "A"}
    cfg = make_cfg(contigs, [("GATC", 1, "a"), ("GATC", 3, "m")], tmp_path / "asm")
    return cfg, tmp_path


def both(cfg, tmp, text, tag="s", **kw):
    f = tmp / f"{tag}.bed"
    f.write_bytes(text if isinstance(text, bytes) else text.encode())
    return compare_range(cfg, f, 0, -1, tmp, tag=tag, **kw)[0]


def test_row_rules(small):
    cfg, tmp = small
    # GATC at 2: the A at 3 on +, the A at 4 on -; one site (palindrome), pos = lowest position with rows
    t = both(cfg, tmp, rows(("c1", 3, "a", "+", 10, 3), ("c1", 4, "a", "-", 7, 2)), "strands")
    assert t.rows() == [(0, 0, 3, 17, 5)]
    t = both(cfg, tmp, rows(("c1", 4, "a", "-", 7, 2)), "minus_only")
    assert t.rows() == [(0, 0, 4, 7, 2)]
    # wrong strand, wrong code, unknown code, zero coverage: no rows
    t = both(cfg, tmp, rows(("c1", 3, "a", "-", 10, 3), ("c1", 3, "m", "+", 10, 3), ("c1", 3, "17802", "+", 10, 3),
                            ("c1", 3, "a", "+", 0, 0), ("c1", 7, "a", "+", 4, 4)), "filters")
    assert t.rows() == [(0, 0, 7, 4, 4)]
    # a strand other than '-' counts as '+'; repeated rows are summed
    t = both(cfg, tmp, rows(("c1", 3, "a", ".", 10, 3), ("c1", 3, "a", "+", 1, 1)), "dot")
    assert t.rows() == [(0, 0, 3, 11, 4)]
    # single-row contigs, a contig absent from the assembly, the last contig of the range
    t = both(cfg, tmp, rows(("zz", 5, "a", "+", 3, 1), ("c3", 1, "a", "+", 9, 9), ("c4", 0, "a", "+", 2, 1),
                            ("c2", 3, "a", "+", 5, 0)), "single")
    assert t.rows() == [(1, 0, 3, 5, 0), (2, 0, 1, 9, 9)]


def test_odd_but_valid_text(small):
    cfg, tmp = small
    base = rows(("c1", 3, "a", "+", 10, 3), ("c1", 7, "a", "+", 6, 1), ("c2", 3, "a", "+", 5, 0))
    ref = both(cfg, tmp, base, "base")
    assert ref.height == 3
    assert both(cfg, tmp, base[:-1], "no_final_newline").equals(ref)
    assert both(cfg, tmp, base.replace("\n", "\r\n"), "crlf").equals(ref)
    assert both(cfg, tmp, "#chrom\tstart\n" + base.replace("c2", "# x y\nc2", 1), "comments").equals(ref)
    assert both(cfg, tmp, rows(("c1", 3, "a", "+", 10, 3), ("c1", 7, "a", "+", 6, 1), ("c2", 3, "a", "+", 5, 0), sep=" "),
                "space").equals(ref)
    # short rows: no n_valid -> dropped; n_valid without n_mod -> k counts 0
    t = both(cfg, tmp, "c1\t3\t4\ta\n" + "c1\t7\t8\ta\t5\t+\t7\t8\t0\t9\t0.1\n" + rows(("c1", 12, "a", "+", 2, 1)), "short")
    assert t.rows() == [(0, 0, 7, 9, 0), (0, 0, 12, 2, 1)]
    t = both(cfg, tmp, rows(("c1", 3, "a", "+", 10, ""), ("c1", 12, "a", "+", 2, 1)), "empty_nmod")
    assert t.rows() == [(0, 0, 3, 10, 0), (0, 0, 12, 2, 1)]
    # rows out of position order inside a contig
    lines = rows(*[("c1", p, "a", s, 3, 1) for p in range(60) for s in "+-"]).splitlines(keepends=True)
    ordered = both(cfg, tmp, "".join(lines), "ordered")
    random.Random(2).shuffle(lines)
    assert both(cfg, tmp, "".join(lines), "shuffled").equals(ordered) and ordered.height > 5
    # empty file
    assert both(cfg, tmp, "", "empty").height == 0


def test_blocks_of_the_python_pass_do_not_matter(small):
    cfg, tmp = small
    text = rows(*[(c, p, "a", s, 3, 1) for c, n in (("c1", 69), ("zz", 30), ("c2", 42), ("c3", 4)) for p in range(n)
                  for s in "+-"])
    ref = both(cfg, tmp, text, "b0", block=1 << 20)
    for block in (64, 100, 1000):
        assert both(cfg, tmp, text, f"b{block}", block=block).equals(ref)
