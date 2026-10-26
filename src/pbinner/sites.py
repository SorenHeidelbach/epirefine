"""Streaming extraction of motif sites from a modkit pileup, in parallel and in bounded memory.

The assembly is 2-bit encoded once into a memory-mapped file. The pileup is split into byte
ranges that start at contig boundaries, and one worker per range:

  1. parses its range in blocks (6 of the 18 columns),
  2. finds motif records for the contigs in the block (MotifMatcher on the shared encoded assembly),
  3. deduplicates them into (group, site) records (same rules as motifs.assign_sites),
  4. joins pileup rows to records on integer keys and sums counts per site,
  5. writes compact site rows (contig id, group, pos, n, k) to Parquet part files.

All motif records of the assembly are never held at once, so memory does not grow with
assembly size x number of motifs. Downstream stages read the part files one at a time; each part
holds whole contigs.

The pileup must have all rows of a contig together (modkit output, or any file sorted by contig);
this is checked and violations raise an error. Gzipped pileups are read by a single worker.
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import polars as pl

from .motifs import Motif, MotifMatcher, _CODE, group_sizes

MOD_CODE = {"a": 0, "m": 1, "21839": 2, "h": 3}
SITE_SCHEMA = {"contig": pl.Int32, "group": pl.Int32, "pos": pl.Int64, "n": pl.Int32, "k": pl.Int32}
_PARSE_COLS = [0, 1, 3, 5, 9, 11]


class EncodedAssembly:
    """Contig names, lengths and a memory-mapped 2-bit code array (one byte per base)."""

    def __init__(self, directory):
        d = Path(directory)
        meta = json.loads((d / "assembly.json").read_text())
        self.names: list[str] = meta["names"]
        self.offsets = np.array(meta["offsets"], np.int64)
        self.lengths = np.array(meta["lengths"], np.int64)
        self.index = {c: i for i, c in enumerate(self.names)}
        self.directory = str(d)
        self.codes = np.memmap(d / "assembly.u8", np.uint8, "r")

    def seq(self, i: int) -> np.ndarray:
        return self.codes[self.offsets[i]:self.offsets[i] + self.lengths[i]]

    @staticmethod
    def write(contigs: dict[str, str], directory) -> "EncodedAssembly":
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        names, offsets, lengths, off = [], [], [], 0
        with open(d / "assembly.u8", "wb") as fh:
            for name, seq in contigs.items():
                fh.write(_CODE[np.frombuffer(seq.encode("ascii"), np.uint8)].tobytes())
                names.append(name); offsets.append(off); lengths.append(len(seq)); off += len(seq)
        (d / "assembly.json").write_text(json.dumps({"names": names, "offsets": offsets, "lengths": lengths}))
        return EncodedAssembly(d)


def _contig_at(fh, offset: int) -> tuple[int, bytes]:
    """(start offset of the first full line at/after offset, its contig name)."""
    fh.seek(offset)
    if offset:
        fh.readline()
    start = fh.tell()
    line = fh.readline()
    return start, line.split(b"\t", 1)[0]


def split_pileup(path, n_parts: int) -> list[tuple[int, int]]:
    """Byte ranges covering the file, each starting at the first row of a contig."""
    size = os.path.getsize(path)
    if str(path).endswith(".gz") or n_parts <= 1 or size < (64 << 20):
        return [(0, -1)]
    cuts = [0]
    with open(path, "rb") as fh:
        for i in range(1, n_parts):
            start, name = _contig_at(fh, size * i // n_parts)
            if not name or start <= cuts[-1]:
                continue
            # rows of a contig are contiguous, so "first full line after x is this contig" is true and
            # then false as x grows: binary search for where the contig ends
            lo, hi = start, size
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if _contig_at(fh, mid)[1] == name:
                    lo = mid
                else:
                    hi = mid
            pos = _contig_at(fh, hi)[0] if hi < size else size
            if cuts[-1] < pos < size:
                cuts.append(pos)
    cuts.append(size)
    return [(a, b) for a, b in zip(cuts[:-1], cuts[1:]) if b > a]


def _blocks(path, start: int, end: int, block: int):
    import gzip
    fh = gzip.open(path, "rb") if str(path).endswith(".gz") else open(path, "rb")
    with fh:
        if end < 0:
            tail = b""
            while True:
                buf = fh.read(block)
                if not buf:
                    if tail:
                        yield tail
                    return
                buf = tail + buf
                cut = buf.rfind(b"\n") + 1
                tail = buf[cut:]
                if cut:
                    yield buf[:cut]
        fh.seek(start)
        pos = start
        while pos < end:
            buf = fh.read(min(block, end - pos))
            if not buf:
                return
            if pos + len(buf) < end:
                cut = buf.rfind(b"\n") + 1
                fh.seek(pos + cut)
                buf = buf[:cut]
            pos += len(buf)
            if buf:
                yield buf


def _parse(block: bytes) -> pl.DataFrame:
    import io
    first = block[: block.find(b"\n")]
    if b" " in first:  # older modkit: columns 10+ space separated
        block = block.replace(b" ", b"\t")
    df = pl.read_csv(io.BytesIO(block), separator="\t", has_header=False, columns=_PARSE_COLS,
                     comment_prefix="#", truncate_ragged_lines=True,
                     schema_overrides={"column_2": pl.Int64, "column_10": pl.Int32, "column_12": pl.Int32})
    df.columns = ["contig", "pos", "mod_type", "strand", "n_valid", "n_mod"]
    return df


class _Worker:
    def __init__(self, cfg: dict):
        self.asm = EncodedAssembly(cfg["asm_dir"])
        self.motifs = [Motif(*m) for m in cfg["motifs"]]
        self.matcher = MotifMatcher(self.motifs)
        self.group = np.array(cfg["group_of"], np.int32)
        self.rank = np.array(cfg["motif_rank"], np.int64)
        self.gsize = np.array(group_sizes(cfg["group_of"], cfg["motif_rank"]), np.int64)
        self.modc = np.array([MOD_CODE.get(m.mod_type, 3) for m in self.motifs], np.int64)
        self.wanted = cfg["wanted"]  # contig ids to keep (None = all)
        self.sep = int(max(len(m.seq) for m in self.motifs))

    def records(self, cids: list[int]) -> pl.DataFrame:
        """Deduplicated site records of these contigs: key (cidx, pos, strand, mod) -> group, site."""
        seqs = [self.asm.seq(i) for i in cids]
        offsets = np.cumsum([0] + [len(s) + self.sep for s in seqs])[:-1]
        codes = np.full(int(offsets[-1] + len(seqs[-1])), 4, np.uint8)
        for o, s in zip(offsets, seqs):
            codes[o:o + len(s)] = s
        rec = self.matcher.find(codes)
        if not len(rec["pos"]):
            return pl.DataFrame(schema={"key": pl.Int64, "group": pl.Int32, "site": pl.Int64})
        cidx = np.searchsorted(offsets, rec["pos"], side="right") - 1
        pos = rec["pos"] - offsets[cidx]
        occ = rec["occ"] - 2 * offsets[cidx]
        mi = rec["motif"]
        key = (cidx.astype(np.int64) << 36) | (pos << 3) | (rec["strand"].astype(np.int64) << 2) | self.modc[mi]
        g = self.group[mi]
        df = pl.DataFrame({"key": key, "group": g, "gsize": self.gsize[g], "rank": self.rank[mi], "motif": mi,
                           "site": (cidx.astype(np.int64) << 50) | (rec["occ_motif"].astype(np.int64) << 36) | occ})
        # same rules as motifs.assign_sites: most specific group, then its most general motif
        return (df.sort(["key", "gsize", "group", "rank", "motif"], descending=[False, False, False, True, False])
                .unique(subset="key", keep="first")
                .select("key", "group", "site"))

    def process(self, pile: pl.DataFrame, seen: set) -> pl.DataFrame | None:
        names = pile["contig"].unique(maintain_order=True).to_list()
        cids = []
        for c in names:
            if c in seen:
                raise ValueError(f"pileup rows of contig {c} are not contiguous; sort the pileup by contig")
            seen.add(c)
            i = self.asm.index.get(c)
            if i is not None and (self.wanted is None or i in self.wanted):
                cids.append(i)
        out = [self._sites(pile, cids[s:s + 4096]) for s in range(0, len(cids), 4096)]
        out = [o for o in out if o is not None]
        return pl.concat(out) if out else None

    def _sites(self, pile: pl.DataFrame, cids: list[int]) -> pl.DataFrame | None:
        local = {self.asm.names[i]: j for j, i in enumerate(cids)}
        pile = (pile.filter(pl.col("contig").is_in(list(local)) & (pl.col("n_valid") > 0))
                .with_columns(pl.col("contig").replace_strict(local, return_dtype=pl.Int64).alias("cidx"),
                              pl.col("mod_type").replace_strict(MOD_CODE, default=7, return_dtype=pl.Int64).alias("m"))
                .filter(pl.col("m") < 7)
                .select((pl.col("cidx") * (1 << 36) + pl.col("pos") * 8
                         + (pl.col("strand") == "-").cast(pl.Int64) * 4 + pl.col("m")).alias("key"),
                        "n_valid", "n_mod"))
        rec = self.records(cids)
        hit = pile.join(rec, on="key", how="inner")
        if not hit.height:
            return None
        agg = hit.group_by("site").agg(pl.col("group").first(), (pl.col("key") // 8).min().alias("pk"),
                                       pl.col("n_valid").sum().alias("n"), pl.col("n_mod").sum().alias("k"))
        cid = np.array(cids, np.int32)[(agg["site"].to_numpy() >> 50)]
        return pl.DataFrame({"contig": cid, "group": agg["group"], "pos": agg["pk"].to_numpy() & ((1 << 33) - 1),
                             "n": agg["n"], "k": agg["k"]}).cast(SITE_SCHEMA)


def _run_range(cfg: dict, path: str, start: int, end: int, out_prefix: str, block: int, flush_rows: int):
    w = _Worker(cfg)
    seen: set = set()
    parts, buf, n_buf, carry = [], [], 0, None
    first_contig = last_contig = None

    def flush():
        nonlocal buf, n_buf
        if buf:
            f = f"{out_prefix}_{len(parts):04d}.parquet"
            pl.concat(buf).sort("contig", "pos").write_parquet(f, compression="zstd")
            parts.append(f)
            buf, n_buf = [], 0

    for blk in _blocks(path, start, end, block):
        df = _parse(blk)
        if not df.height:
            continue
        if carry is not None:
            df = pl.concat([carry, df])
        last = df["contig"][-1]
        if first_contig is None:
            first_contig = df["contig"][0]
        tail = df["contig"] == last
        carry, df = df.filter(tail), df.filter(~tail)  # the last contig may continue in the next block
        if df.height:
            s = w.process(df, seen)
            if s is not None:
                buf.append(s); n_buf += s.height
        if n_buf >= flush_rows:
            flush()
    if carry is not None and carry.height:
        last_contig = carry["contig"][0]
        s = w.process(carry, seen)
        if s is not None:
            buf.append(s)
    flush()
    return {"parts": parts, "contigs": sorted(seen), "first": first_contig, "last": last_contig}


def extract_sites(asm: EncodedAssembly, pileup, motifs: list[Motif], group_of: list[int], motif_rank,
                  out_dir, wanted: set[int] | None = None, workers: int | None = None,
                  block: int = 128 << 20, flush_rows: int = 20_000_000, log=print) -> list[str]:
    """Run the pileup pass. Returns the Parquet part files (each holds whole contigs)."""
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    workers = workers or min(16, os.cpu_count() or 1)
    ranges = split_pileup(pileup, workers * 4)
    cfg = {"asm_dir": asm.directory, "motifs": [(m.seq, m.mod_pos, m.mod_type) for m in motifs],
           "group_of": list(group_of), "motif_rank": list(motif_rank), "wanted": wanted}
    log(f"  pileup pass: {len(ranges)} byte ranges on {min(workers, len(ranges))} workers")
    env_before = os.environ.get("POLARS_MAX_THREADS")
    os.environ["POLARS_MAX_THREADS"] = "1"
    try:
        with ProcessPoolExecutor(min(workers, len(ranges)), mp_context=get_context("spawn")) as ex:
            futs = [ex.submit(_run_range, cfg, str(pileup), a, b, str(out / f"r{i:04d}"), block, flush_rows)
                    for i, (a, b) in enumerate(ranges)]
            results = []
            for i, f in enumerate(futs):
                results.append(f.result())
                if (i + 1) % max(1, len(futs) // 10) == 0:
                    log(f"  pileup pass: {i + 1}/{len(futs)} ranges done")
    finally:
        if env_before is None:
            os.environ.pop("POLARS_MAX_THREADS", None)
        else:
            os.environ["POLARS_MAX_THREADS"] = env_before
    seen: dict[str, int] = {}
    for r in results:
        for c in r["contigs"]:
            seen[c] = seen.get(c, 0) + 1
    dup = [c for c, n in seen.items() if n > 1]
    if dup:
        raise ValueError(f"pileup rows of {len(dup)} contigs (e.g. {dup[0]}) are not contiguous; sort the pileup by contig")
    return [p for r in results for p in r["parts"]]


# ---------------------------------------------------------------------------------------------
# second pass: site cache -> per-unit likelihood tables


def _unit_part(part: str, cfg: dict) -> dict:
    from .assignability import expected_tables
    from .detection import DetectionParams
    from .grid import Grid
    from .model import unit_loglik

    grid = Grid(cfg["grid_size"])
    det = {g: DetectionParams(*p) for g, p in cfg["det"].items()}
    M = cfg["n_groups"]
    df = pl.read_parquet(part).filter(pl.col("n") >= cfg["min_cov"])
    cid, pos = df["contig"].to_numpy(), df["pos"].to_numpy()
    grp, n, k = df["group"].to_numpy(), df["n"].to_numpy(), df["k"].to_numpy()

    def tables(units, rows, with_h):
        ids, local = np.unique(units, return_inverse=True)
        sub = pl.DataFrame({"unit": local, "group": grp[rows], "n": n[rows], "k": k[rows]})
        res = {"ids": ids, "L": unit_loglik(sub, len(ids), M, det, grid, dtype=np.float32),
               "n_sites": np.bincount(local, minlength=len(ids))}
        gs = sub.group_by("unit", "group").len()
        res["n_sites_group"] = (ids[gs["unit"].to_numpy()], gs["group"].to_numpy(), gs["len"].to_numpy())
        if with_h:  # expected-evidence tables: calibration fragments only
            res["H"] = expected_tables(sub.select("unit", "group", "n"), len(ids), M, det, grid, dtype=np.float32)
        return res

    out = {}
    co = cfg["chunk_offset"][cid]
    rows = np.flatnonzero(co >= 0)
    if len(rows):
        out["chunks"] = tables(co[rows] + pos[rows] // cfg["chunk_size"], rows, False)
    to = cfg["target_of"][cid]
    rows = np.flatnonzero(to >= 0)
    if len(rows):
        out["targets"] = tables(to[rows], rows, False)
    fr = cfg["frags"]
    if fr is not None and fr.height:
        # a site can lie in several fragments (of different lengths): one row per (site, fragment)
        f = (pl.DataFrame({"row": np.arange(df.height), "contig": cid, "pos": pos})
             .join(fr, on="contig", how="inner")
             .filter((pl.col("pos") >= pl.col("start")) & (pl.col("pos") < pl.col("end"))))
        if f.height:
            out["frags"] = tables(f["unit"].to_numpy(), f["row"].to_numpy(), True)
    return out


def unit_tables(parts: list[str], *, n_groups: int, det: dict, grid_size: int, min_cov: int,
                chunk_offset: np.ndarray, chunk_size: int, n_chunks: int, target_of: np.ndarray, n_targets: int,
                frags: pl.DataFrame | None, workers: int | None = None, log=print) -> dict:
    """Likelihood tables for host chunks, targets and calibration fragments, in parallel over parts.

    Returns float32 arrays L_chunks [K, M, G], L_t [U, M, G], L_frag, H_frag = (H1, H0), and site
    counts per chunk, per target, per (target, group) and per fragment.
    """
    G = grid_size
    cfg = {"grid_size": G, "n_groups": n_groups, "min_cov": min_cov, "chunk_size": chunk_size,
           "det": {g: (p.mu0, p.kappa0, p.mu1, p.kappa1, p.pi1, p.n_sites) for g, p in det.items()},
           "chunk_offset": chunk_offset, "target_of": target_of,
           "frags": frags.select("unit", "contig", "start", "end") if frags is not None else None}
    n_frags = frags.height if frags is not None else 0
    res = {"L_chunks": np.zeros((n_chunks, n_groups, G), np.float32),
           "chunk_sites": np.zeros(n_chunks, np.int64),
           "L_t": np.zeros((n_targets, n_groups, G), np.float32),
           "n_sites_t": np.zeros(n_targets, np.int64),
           "n_sites_group_t": np.zeros((n_targets, n_groups), np.int32),
           "L_frag": np.zeros((n_frags, n_groups, G), np.float32),
           "H_frag": (np.zeros((n_frags, n_groups, G), np.float32), np.zeros((n_frags, n_groups, G), np.float32)),
           "frag_sites": np.zeros(n_frags, np.int64)}
    workers = workers or min(16, os.cpu_count() or 1)
    env = {k: os.environ.get(k) for k in ("POLARS_MAX_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
    os.environ.update({k: "1" for k in env})
    try:
        with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
            for i, r in enumerate(ex.map(_unit_part, parts, [cfg] * len(parts))):
                if "chunks" in r:
                    c = r["chunks"]
                    res["L_chunks"][c["ids"]] = c["L"]; res["chunk_sites"][c["ids"]] = c["n_sites"]
                if "targets" in r:
                    t = r["targets"]
                    res["L_t"][t["ids"]] = t["L"]; res["n_sites_t"][t["ids"]] = t["n_sites"]
                    u, g, n = t["n_sites_group"]
                    res["n_sites_group_t"][u, g] = n
                if "frags" in r:
                    f = r["frags"]
                    res["L_frag"][f["ids"]] += f["L"]; res["frag_sites"][f["ids"]] += f["n_sites"]
                    res["H_frag"][0][f["ids"]] += f["H"][0]; res["H_frag"][1][f["ids"]] += f["H"][1]
                if (i + 1) % max(1, len(parts) // 5) == 0:
                    log(f"  unit tables: {i + 1}/{len(parts)} parts")
    finally:
        for k, v in env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return res
