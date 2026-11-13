"""Shared by the tests: the Python reference (EpiRefine's sites._run_range), the same pass through the
kernel, synthetic inputs, and the comparison."""
from __future__ import annotations

import os
import random
import zlib

import numpy as np
import polars as pl

import pytest

epirefine_rs = pytest.importorskip("epirefine._kernels", reason="no compiled kernels in this install")
from epirefine import sites
from epirefine.motifs import Motif, MotifMatcher, build_records, group_motifs, revcomp
from epirefine.sites import SITE_SCHEMA, EncodedAssembly

os.environ.setdefault("POLARS_MAX_THREADS", "1")
ALL = ["contig", "group", "pos", "n", "k"]


def bgzf(data: bytes, block: int, eof: bool = True) -> bytes:
    """BGZF writer of EpiRefine's tests/test_bgzf.py (blocks of `block` bytes of text + the EOF block)."""
    out = b""
    for i in list(range(0, len(data), block)) + ([None] if eof else []):
        chunk = b"" if i is None else data[i:i + block]
        c = zlib.compressobj(6, zlib.DEFLATED, -15)
        body = c.compress(chunk) + c.flush()
        size = len(body) + 26
        out += (b"\x1f\x8b\x08\x04\x00\x00\x00\x00\x00\xff\x06\x00BC\x02\x00" + (size - 1).to_bytes(2, "little") + body
                + zlib.crc32(chunk).to_bytes(4, "little") + len(chunk).to_bytes(4, "little"))
    return out


def run_range_rs(cfg: dict, path: str, start: int, end: int, out_prefix: str, block: int, flush_rows: int) -> dict:
    """sites._run_range through the kernel: the function of docs/epirefine_integration.diff."""
    asm = EncodedAssembly(cfg["asm_dir"])
    reader = epirefine_rs.PileupReader(path, start, end, names=asm.names, offsets=asm.offsets, lengths=asm.lengths,
                                       codes=asm.codes, motifs=cfg["motifs"], group_of=cfg["group_of"],
                                       motif_rank=cfg["motif_rank"], wanted=cfg["wanted"])
    parts = []
    for batch in reader.batches(flush_rows):
        f = f"{out_prefix}_{len(parts):04d}.parquet"
        pl.DataFrame(batch, schema=SITE_SCHEMA).sort("contig", "pos").write_parquet(f, compression="zstd")
        parts.append(f)
    names = reader.contigs
    return {"parts": parts, "contigs": sorted(names), "first": names[0] if names else None,
            "last": names[-1] if names else None}


def run_range_keys(cfg: dict, path: str, start: int, end: int) -> pl.DataFrame:
    """Design (b): motif matching and deduplication by EpiRefine (_Worker.records), reading and
    aggregation by the kernel."""
    w = sites._Worker(cfg)
    n = len(w.asm.names)
    cids = [i for i in range(n) if w.wanted is None or i in w.wanted]
    ptr = np.zeros(n + 1, np.int64)
    keys, groups, site = [np.empty(0, np.int64)], [np.empty(0, np.int32)], [np.empty(0, np.int64)]
    if cids:
        rec = w.records(cids).sort("key")
        key = rec["key"].to_numpy()
        cidx = np.asarray(cids)[key >> 36]
        order = np.argsort(cidx, kind="stable")
        counts = np.bincount(cidx, minlength=n)
        ptr[1:] = np.cumsum(counts)
        keys, groups, site = [(key & ((1 << 36) - 1))[order]], [rec["group"].to_numpy()[order]], [rec["site"].to_numpy()[order]]
    r = epirefine_rs.PileupReader.with_keys(path, start, end, names=w.asm.names, key_ptr=ptr, keys=keys[0],
                                            key_group=groups[0], key_site=site[0])
    return pl.DataFrame(r.read_all(), schema=SITE_SCHEMA)


def table(parts) -> pl.DataFrame:
    if not parts:
        return pl.DataFrame(schema=SITE_SCHEMA)
    return pl.concat([pl.read_parquet(p) for p in parts]).sort(ALL)


def same_table(a: pl.DataFrame, b: pl.DataFrame) -> bool:
    return a.schema == b.schema and a.shape == b.shape and a.sort(ALL).equals(b.sort(ALL))


def n_diff(a: pl.DataFrame, b: pl.DataFrame) -> int:
    """Rows present in one table and not the other (with multiplicity)."""
    ca = a.group_by(ALL).len()
    cb = b.group_by(ALL).len()
    j = ca.join(cb, on=ALL, how="full", suffix="_b").with_columns(pl.col("len").fill_null(0), pl.col("len_b").fill_null(0))
    return int((j["len"].cast(pl.Int64) - j["len_b"].cast(pl.Int64)).abs().sum())


def compare_range(cfg, path, start, end, tmp, block=1 << 24, flush_rows=5000, tag="x"):
    """Runs both paths on one range; returns (python table, kernel table) after asserting equality of the
    tables and of the contig bookkeeping."""
    py = sites._run_range(cfg, str(path), start, end, str(tmp / f"py_{tag}"), block, flush_rows)
    rs = run_range_rs(cfg, str(path), start, end, str(tmp / f"rs_{tag}"), block, flush_rows)
    tp, tr = table(py["parts"]), table(rs["parts"])
    assert tr.schema == tp.schema == pl.Schema(SITE_SCHEMA)
    assert n_diff(tp, tr) == 0 and tr.equals(tp), f"range {start}-{end}: tables differ"
    assert rs["contigs"] == py["contigs"] and rs["first"] == py["first"] and rs["last"] == py["last"]
    # every part holds whole contigs, as in the Python pass
    seen = set()
    for p in rs["parts"]:
        c = set(pl.read_parquet(p)["contig"].unique().to_list())
        assert not (c & seen)
        seen |= c
    return tp, tr


# ---------------------------------------------------------------------------------------------
# synthetic inputs

MOTIF_SETS = {
    "palindromic": [("GATC", 1, "a"), ("CCGG", 1, "m"), ("GGCC", 2, "21839")],
    "non_palindromic": [("GAAGA", 4, "a"), ("ACCGAG", 3, "a"), ("CTCGGT", 0, "m")],
    "degenerate": [("RGATCY", 2, "a"), ("GATC", 1, "a"), ("GANTC", 1, "a"), ("CCWGG", 1, "m"), ("CCNGG", 1, "m"),
                   ("CCNGG", 1, "21839"), ("GCANNNNNNTGC", 2, "a"), ("AGCNNNNNCTTC", 0, "a"), ("GAAGNNNNNGCT", 2, "a"),
                   ("TCGA", 3, "a"), ("YGGCCR", 3, "m"), ("GGCC", 2, "h")],
    # spans up to and beyond the 32-base window of the scan, and 32 fixed bases (a 64-bit code)
    "long": [("GATCGATCGATCGA", 1, "a"), ("ACGTNNACGTNNACGTNNACGT", 0, "a"), ("CCWGGNNNNNNNNCCWGG", 1, "m"), ("GATC", 1, "a"),
             ("GATC" + "N" * 30 + "GATC", 1, "a"), ("ACGTACGTACGTACGTACGTACGTACGTACGA", 0, "a"),
             ("TCGTACGTACGTACGTNACGTACGTACGTACGA", 32, "a"), ("GANNNNNNNNNNNNNNNNNNNNNNNNNNNNTC", 1, "a"),
             ("CANNNNNNNNNNNNNNNNNNNNNNNNNNNNNTG", 1, "a")],
    "single": [("GATC", 1, "a")],
}


def random_seq(rng: random.Random, n: int, plant=()) -> str:
    s = rng.choices("ACGT", k=n)
    for _ in range(n // 25):  # plant motif instances so that sites are common
        if not plant:
            break
        m = rng.choice(plant)
        inst = "".join(rng.choice({"R": "AG", "Y": "CT", "W": "AT", "S": "CG", "N": "ACGT"}.get(c, c)) for c in m)
        p = rng.randrange(0, max(1, n - len(inst) + 1))
        s[p:p + len(inst)] = inst[: n - p]
    if n > 50 and rng.random() < 0.3:  # an N run and lower case
        p = rng.randrange(n - 10)
        s[p:p + rng.randrange(1, 10)] = "N" * 9
    return "".join(s)[:n]


def make_assembly(rng: random.Random, motifs, n_contigs=30, sizes=(1, 3, 40, 400, 3000)) -> dict[str, str]:
    plant = [m[0] for m in motifs]
    return {f"contig_{i}": random_seq(rng, rng.choice(sizes), plant) for i in range(n_contigs)}


def pileup_text(rng: random.Random, contigs: dict[str, str], order=None, layout="tab", extra_contigs=("absent_1",),
                keep=0.9, comments=True) -> bytes:
    """modkit-like rows for every (position, strand, modification): tab layout (18 tab-separated columns) or
    the older one (the last 9 space separated). Includes zero-coverage rows, unknown codes and rows of
    contigs that are not in the assembly."""
    comp = {"A": "T", "C": "G", "G": "C", "T": "A", "N": "N"}
    codes = {"A": ["a"], "C": ["m", "h", "21839", "17802"]}
    names = list(order if order is not None else contigs)
    for i, x in enumerate(extra_contigs):
        names.insert(min(len(names), 1 + 2 * i), x)
    sep = "\t" if layout == "tab" else " "
    out = []
    if comments:
        # the Python pass decides the layout of a block from its first line, so in the space-separated
        # layout a comment line must contain a space as well (see docs/INTEGRATION.md, differences)
        out.append("#chrom\tstart\tend\tcode\n" if layout == "tab" else "#chrom start end code\n")
    for c in names:
        seq = contigs.get(c, "ACGTACGT")
        rows = 0
        for p in range(len(seq)):
            for strand in "+-":
                base = seq[p] if strand == "+" else comp.get(seq[p], "N")
                for code in codes.get(base, []):
                    # the first row of a contig is always kept and has a letter code: the Python pass infers the
                    # type of the code column from the first rows of each block, and a block of numeric codes
                    # only (21839, 17802) makes it fail (see docs/INTEGRATION.md, differences)
                    if rows and rng.random() > keep:
                        continue
                    nv = rng.choice([0, 0, 1, 2, 5, 20, 60])
                    nm = rng.randrange(nv + 1)
                    tail = sep.join(map(str, [nv, round(100 * nm / max(nv, 1), 2), nm, nv - nm, 0, 1, 2, 0, 3]))
                    out.append(f"{c}\t{p}\t{p + 1}\t{code}\t{nv}\t{strand}\t{p}\t{p + 1}\t255,0,0\t{tail}\n")
                    rows += 1
        if not rows or out[-1].split("\t")[3].isdigit():
            # every contig of the pileup has at least one row, and its last row has a letter code too (the
            # last piece of a BGZF range is parsed as a block of its own by the Python pass)
            p = len(seq) if rows else 0
            out.append(f"{c}\t{p}\t{p + 1}\ta\t0\t+\t{p}\t{p + 1}\t255,0,0\t"
                       + sep.join(["0", "0.0", "0", "0", "0", "0", "0", "0", "0"]) + "\n")
        if comments and rng.random() < 0.2:
            out.append("# a comment line\n")
    return "".join(out).encode()


def make_cfg(contigs: dict[str, str], motifs, asm_dir, wanted=None, random_groups: random.Random | None = None) -> dict:
    """The worker configuration of sites.extract_sites: groups and ranks from EpiRefine's own grouping, or
    random groups and ranks (reverse-complement pairs kept together, as group_motifs guarantees)."""
    asm = EncodedAssembly.write(contigs, asm_dir)
    ms = [Motif(*m) for m in motifs]
    if random_groups is None:
        try:
            records = build_records(contigs, ms)
            group_of = group_motifs(records, ms)
            rank = dict(records.group_by("motif").len().iter_rows())
        except ValueError:  # no occurrence at all
            group_of, rank = list(range(len(ms))), {}
        motif_rank = [int(rank.get(i, 0)) for i in range(len(ms))]
    else:
        r = random_groups
        group_of = [r.randrange(max(1, len(ms) // 2)) for _ in ms]
        by_key = {(m.seq, m.mod_type): i for i, m in enumerate(ms)}
        for i, m in enumerate(ms):
            j = by_key.get((revcomp(m.seq), m.mod_type))
            if j is not None:
                group_of[i] = group_of[j] = min(group_of[i], group_of[j])
        remap = {g: k for k, g in enumerate(sorted(set(group_of)))}
        group_of = [remap[g] for g in group_of]
        motif_rank = [r.choice([0, 1, 5, 5, 100]) for _ in ms]
    return {"asm_dir": asm.directory, "motifs": [(m.seq, m.mod_pos, m.mod_type) for m in ms], "group_of": list(group_of),
            "motif_rank": motif_rank, "wanted": wanted}
