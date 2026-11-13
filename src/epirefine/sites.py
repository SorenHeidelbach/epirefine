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
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import polars as pl

from .model import _cores
from .runinfo import note_worker_peaks
from .motifs import Motif, MotifMatcher, _CODE, canonical_index, group_sizes

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
    def write(contigs, directory) -> "EncodedAssembly":
        """contigs: dict name -> sequence, or an iterable of (name, sequence) (streamed, one contig at a time)."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        names, offsets, lengths, off = [], [], [], 0
        seen: set[str] = set()
        with open(d / "assembly.u8", "wb") as fh:
            for name, seq in (contigs.items() if isinstance(contigs, dict) else contigs):
                if name in seen:  # the name -> index map would point both records at the last one
                    raise ValueError(f"contig name {name!r} occurs more than once in the assembly; contig names must be unique")
                seen.add(name)
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


# --- BGZF (bgzip) pileups: block-compressed, so a worker can start at any block. Positions are virtual offsets
# (compressed offset of the block << 16 | offset inside the inflated block), as in htslib.
_BGZF_HEAD = b"\x1f\x8b\x08\x04"


def _bgzf_bsize(head: bytes) -> int:
    """Total size of the block whose first 18 bytes are head; 0 if this is not a BGZF block header."""
    if len(head) < 18 or head[:4] != _BGZF_HEAD or head[10:16] != b"\x06\x00BC\x02\x00":
        return 0
    return int.from_bytes(head[16:18], "little") + 1


def is_bgzf(path) -> bool:
    with open(path, "rb") as fh:
        return _bgzf_bsize(fh.read(18)) > 0


def _bgzf_inflate(fh, coff: int) -> tuple[bytes, int]:
    """(inflated data of the block at coff, offset of the next block)."""
    import zlib
    fh.seek(coff)
    head = fh.read(18)
    size = _bgzf_bsize(head)
    if not size:
        raise ValueError(f"no BGZF block at offset {coff}")
    body = fh.read(size - 18)
    return zlib.decompress(body[:-8], -15), coff + size


def _bgzf_next_block(fh, offset: int, size: int) -> int:
    """Offset of the first block starting at or after offset (size if none)."""
    if offset <= 0:
        return 0
    pos = offset
    while pos < size:
        fh.seek(pos)
        buf = fh.read((1 << 17) + 18)
        i = buf.find(_BGZF_HEAD)
        while 0 <= i <= len(buf) - 18:
            n = _bgzf_bsize(buf[i:i + 18])
            if n:  # a real header is followed by another header (or the end of the file), twice
                p, ok = pos + i + n, True
                for _ in range(2):
                    if p >= size:
                        ok = p == size
                        break
                    fh.seek(p)
                    m = _bgzf_bsize(fh.read(18))
                    if not m:
                        ok = False
                        break
                    p += m
                if ok:
                    return pos + i
            i = buf.find(_BGZF_HEAD, i + 1)
        if len(buf) < 18:
            break
        pos += len(buf) - 17
    return size


def _bgzf_contig_at(fh, offset: int, size: int):
    """(virtual offset, contig) of the first full line in or after the first block at/after offset, or None.
    As in _contig_at, the line the position falls in is skipped (none at the start of the file)."""
    c = _bgzf_next_block(fh, offset, size)
    skip = c > 0
    while c < size:
        data, nxt = _bgzf_inflate(fh, c)
        u = 0
        if skip:
            j = data.find(b"\n")
            if j < 0:
                c = nxt
                continue
            u, skip = j + 1, False
        if u < len(data):
            line, cc = data[u:], nxt
            while b"\t" not in line and b"\n" not in line and cc < size:
                more, cc = _bgzf_inflate(fh, cc)
                line += more
            return (c << 16) | u, line.split(b"\n", 1)[0].split(b"\t", 1)[0]
        c = nxt
    return None


def _bgzf_contig_end(fh, voff: int, name: bytes, size: int):
    """Virtual offset of the first line from voff on that is not a row of contig name (None: none before the end)."""
    c, p = voff >> 16, voff & 0xFFFF
    pend, pend_pos = b"", None
    while c < size:
        data, nxt = _bgzf_inflate(fh, c)
        while p < len(data):
            j = data.find(b"\n", p)
            pos = pend_pos if pend_pos is not None else (c << 16) | p
            if j < 0:
                pend, pend_pos = pend + data[p:], pos
                break
            line = pend + data[p:j + 1]
            pend, pend_pos = b"", None
            if line.split(b"\t", 1)[0] != name:
                return pos
            p = j + 1
        c, p = nxt, 0
    return None


def _split_bgzf(path, n_parts: int, size: int) -> list[tuple[int, int]]:
    end = size << 16
    cuts = [0]
    with open(path, "rb") as fh:
        for i in range(1, n_parts):
            at = _bgzf_contig_at(fh, size * i // n_parts, size)
            if at is None or not at[1] or at[0] <= cuts[-1]:
                continue
            start, name = at
            # as in the plain file: "the first full line after x is this contig" is true and then false as x
            # grows; x is a compressed offset here, so the search ends on two neighbouring blocks, and the
            # first row of the next contig is found by reading on from the last row known to be this contig
            lo, hi = start >> 16, size
            while hi - lo > 1:
                mid = (lo + hi) // 2
                r = _bgzf_contig_at(fh, mid, size)
                if r is not None and r[1] == name:
                    lo = mid
                else:
                    hi = mid
            r = _bgzf_contig_at(fh, lo, size)
            pos = _bgzf_contig_end(fh, r[0] if r is not None and r[1] == name and r[0] >= start else start, name, size)
            if pos is not None and cuts[-1] < pos < end:
                cuts.append(pos)
    cuts.append(end)
    return [(a, b) for a, b in zip(cuts[:-1], cuts[1:]) if b > a]


def split_pileup(path, n_parts: int, min_size: int = 64 << 20) -> list[tuple[int, int]]:
    """Ranges covering the file, each starting at the first row of a contig: byte offsets for a plain file,
    virtual offsets for a bgzipped one. Other gzip files cannot be split: one range (0, -1)."""
    size = os.path.getsize(path)
    if str(path).endswith(".gz"):
        if n_parts <= 1 or size < min_size or not is_bgzf(path):
            return [(0, -1)]
        return _split_bgzf(path, n_parts, size)
    if n_parts <= 1 or size < min_size:
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


def _bgzf_members(fh, c0: int, c1: int, path=""):
    """Inflated data of each BGZF block from compressed offset c0 up to c1 (a block start).

    One raw-deflate call per block and the block's own CRC-32 and length check: 3.5-4x the speed of
    gzip.GzipFile over the same bytes, which decodes through a small-buffer Python reader. Anything
    wrong with a block (no gzip member with a BGZF size field, ends short, does not inflate, wrong
    checksum or length, or the blocks do not end at c1) raises: a truncated or damaged file must not
    pass as a shorter one.
    """
    import zlib
    fh.seek(c0)
    pos = c0
    while pos < c1:
        head = fh.read(12)
        if len(head) < 12 or head[:3] != b"\x1f\x8b\x08" or not head[3] & 4:
            raise ValueError(f"{path}: no BGZF block at offset {pos} (truncated or corrupt file)")
        xlen = int.from_bytes(head[10:12], "little")
        extra, size, i = fh.read(xlen), 0, 0
        while i + 4 <= len(extra):  # subfields: two id bytes, length, data; BGZF's is 'BC' with the block size - 1
            n = int.from_bytes(extra[i + 2:i + 4], "little")
            if extra[i:i + 2] == b"BC" and n == 2:
                size = int.from_bytes(extra[i + 4:i + 6], "little") + 1
            i += 4 + n
        body = fh.read(size - 12 - xlen) if size > 12 + xlen + 8 else b""
        if len(extra) < xlen or not size or len(body) < size - 12 - xlen or pos + size > c1:
            raise ValueError(f"{path}: BGZF block at offset {pos} is incomplete (truncated or corrupt file)")
        try:
            data = zlib.decompress(body[:-8], -15)
        except zlib.error as e:
            raise ValueError(f"{path}: BGZF block at offset {pos} does not inflate ({e})") from None
        if zlib.crc32(data) != int.from_bytes(body[-8:-4], "little") or len(data) != int.from_bytes(body[-4:], "little"):
            raise ValueError(f"{path}: BGZF block at offset {pos} fails its checksum or length (corrupt file)")
        pos += size
        yield data


def _bgzf_blocks(path, start: int, end: int, block: int):
    """Line-complete chunks of the rows between two virtual offsets (both at line starts).

    The inflated stream is cut every `block` bytes before the split at line ends, exactly as reading
    `block` bytes at a time from a gzip stream did: the chunks decide where site-cache parts are
    flushed, so they must not move.
    """
    c0, u0, c1, u1 = start >> 16, start & 0xFFFF, end >> 16, end & 0xFFFF

    def reads(fh):  # the pieces of each read of exactly `block` bytes (the last one shorter)
        got, n = [], 0
        for data in _bgzf_members(fh, c0, c1, path):
            while n + len(data) >= block:
                k = block - n
                got.append(data[:k])
                yield got
                got, n, data = [], 0, data[k:]
            if data:
                got.append(data); n += len(data)
        if got:
            yield got

    with open(path, "rb", buffering=1 << 20) as fh:
        last = next(_bgzf_members(fh, c1, float("inf"), path))[:u1] if u1 else b""
        tail, drop = b"", u0
        for pieces in reads(fh):
            if drop:
                buf = b"".join(pieces)
                buf, drop = buf[drop:], max(0, drop - len(buf))
                pieces = [buf]
            # buf = tail + read; yield buf up to its last line end, keep the rest: done on the pieces, so
            # the chunk is copied once (each copy of a 128 MB chunk costs as much as inflating a third of it)
            pieces.insert(0, tail)
            i = len(pieces) - 1
            while i >= 0 and b"\n" not in pieces[i]:
                i -= 1
            if i < 0:
                tail = b"".join(pieces)
                continue
            j = pieces[i].rfind(b"\n") + 1
            tail = b"".join([pieces[i][j:]] + pieces[i + 1:])
            pieces[i:] = [pieces[i][:j]]
            yield b"".join(pieces)
        rest = tail + last[drop:]
        if rest:
            yield rest


def _blocks(path, start: int, end: int, block: int):
    import gzip
    if end >= 0 and str(path).endswith(".gz"):
        yield from _bgzf_blocks(path, start, end, block)
        return
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


_PARSE_SCHEMA: dict | None = None


def _parse_schema() -> dict:
    """Types of the six columns read, keyed by the names read_csv gives projected headerless columns.

    Contig, mod code and strand are text whatever a block holds: inferred per block, contig names made
    of digits became integers (no contig matched: no sites, no error) and a block with only numeric
    mod codes (21839) failed in the code lookup. The names are taken from a probe, not assumed: the
    overrides written for 'column_2', 'column_10' and 'column_12' never applied, because this polars
    numbers projected columns from 0.
    """
    global _PARSE_SCHEMA
    if _PARSE_SCHEMA is None:
        import io
        probe = pl.read_csv(io.BytesIO(b"c\t1\t2\ta\t0\t+\t1\t2\t0\t1\t1.0\t1\n"), separator="\t", has_header=False,
                            columns=_PARSE_COLS)
        _PARSE_SCHEMA = dict(zip(probe.columns, (pl.Utf8, pl.Int64, pl.Utf8, pl.Utf8, pl.Int64, pl.Int64)))
    return _PARSE_SCHEMA


def _parse(block: bytes) -> pl.DataFrame:
    import io
    first = block[: block.find(b"\n")]
    if b" " in first:  # older modkit: columns 10+ space separated
        block = block.replace(b" ", b"\t")
    df = pl.read_csv(io.BytesIO(block), separator="\t", has_header=False, columns=_PARSE_COLS,
                     comment_prefix="#", truncate_ragged_lines=True, schema_overrides=_parse_schema())
    df.columns = ["contig", "pos", "mod_type", "strand", "n_valid", "n_mod"]
    return df


class _Worker:
    def __init__(self, cfg: dict):
        self.asm = EncodedAssembly(cfg["asm_dir"])
        self.motifs = [Motif(*m) for m in cfg["motifs"]]
        self.matcher = MotifMatcher(self.motifs, cfg.get("pairing", "sequence"))
        # strand_sites: "sum" = one site per occurrence of a palindrome or complement pair (strands summed);
        # "split" = every record its own site; "auto" = summed unless the strands can differ for a known reason:
        # the occurrence belongs to a complement pair (two motifs, possibly two enzymes), or its two records are
        # matched by different numbers of groups (one strand is also another motif's site, e.g. the A of ATCGATC
        # that Dam methylates on one strand only)
        self.strand_sites = cfg.get("strand_sites", "sum")
        canon = self.matcher.canon
        paired = np.bincount(canon, minlength=len(self.motifs)) > 1
        self.pair = paired[canon]  # motif is half of a complement pair
        self.group = np.array(cfg["group_of"], np.int32)
        self.rank = np.array(cfg["motif_rank"], np.int64)
        # order of the groups for shared positions: summed record counts, or the given ranks (motifs.specificity_ranks)
        self.gsize = np.array(cfg.get("group_rank") or group_sizes(cfg["group_of"], cfg["motif_rank"]), np.int64)
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
            return pl.DataFrame(schema={"key": pl.Int64, "group": pl.Int32, "site": pl.Int64, "half": pl.Boolean})
        cidx = np.searchsorted(offsets, rec["pos"], side="right") - 1
        pos = rec["pos"] - offsets[cidx]
        occ = rec["occ"] - 2 * offsets[cidx]
        mi = rec["motif"]
        key = (cidx.astype(np.int64) << 36) | (pos << 3) | (rec["strand"].astype(np.int64) << 2) | self.modc[mi]
        g = self.group[mi]
        # a site is the occurrence of the canonical motif (both strands of a palindrome or of a complement pair), or
        # the record itself (position, strand, modification type; bit 62 keeps the two kinds of id apart)
        occ_site = (cidx.astype(np.int64) << 50) | (rec["occ_motif"].astype(np.int64) << 36) | occ
        rec_site = (1 << 62) | (cidx.astype(np.int64) << 50) | (key & ((1 << 36) - 1))
        df = pl.DataFrame({"key": key, "group": g, "gsize": self.gsize[g], "rank": self.rank[mi], "motif": mi,
                           "site": rec_site if self.strand_sites == "split" else occ_site})
        if self.strand_sites == "auto":
            df = df.with_columns(pl.Series("rec_site", rec_site), pl.Series("pair", self.pair[mi]),
                                 pl.col("group").n_unique().over("key").alias("n_groups"))
        # same rules as motifs.assign_sites: most specific group, then its most general motif
        df = (df.sort(["key", "gsize", "group", "rank", "motif"], descending=[False, False, False, True, False])
              .unique(subset="key", keep="first", maintain_order=True))
        if self.strand_sites == "auto":
            uneven = pl.col("n_groups").min().over("site") != pl.col("n_groups").max().over("site")
            # half: a single-strand site cut out of an occurrence that has a record on each strand
            df = (df.with_columns((pl.col("pair") | uneven).alias("cut"), (pl.len().over("site") > 1).alias("two"))
                  .with_columns(pl.when("cut").then(pl.col("rec_site")).otherwise(pl.col("site")).alias("site"),
                                (pl.col("cut") & pl.col("two")).alias("half")))
        else:
            df = df.with_columns(pl.lit(False).alias("half"))
        return df.select("key", "group", "site", "half")

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
                                       pl.col("n_valid").sum().alias("n"), pl.col("n_mod").sum().alias("k"),
                                       pl.col("half").first())
        cid = np.array(cids, np.int32)[(agg["site"].to_numpy() >> 50) & 0xFFF]
        out = pl.DataFrame({"contig": cid, "group": agg["group"], "pos": agg["pk"].to_numpy() & ((1 << 33) - 1),
                            "n": agg["n"], "k": agg["k"]}).cast(SITE_SCHEMA)
        if self.strand_sites == "auto":
            # one more column in this mode only: the site holds one strand of a two-strand occurrence, so that
            # contig coverage (median reads per site) can still be counted per occurrence
            out = out.with_columns(agg["half"].cast(pl.Int8).alias("half"))
        return out


def same_pairs(motifs, pairing: str) -> bool:
    """True when `pairing` pairs this motif list exactly as "sequence" does (always, unless a sequence is listed
    with several modified positions or a palindrome twice): the sites are then those of 0.10, so its site
    caches and the compiled reader remain valid. motifs: Motif objects or (seq, mod_pos, mod_type) tuples."""
    if pairing == "sequence":
        return True
    ms = [m if isinstance(m, Motif) else Motif(*m) for m in motifs]
    return canonical_index(ms, pairing) == canonical_index(ms, "sequence")


def _rs(cfg: dict | None = None):
    """The optional compiled kernels (package epirefine-rs), or None when they are not installed, are switched
    off (EPIREFINE_RS=0), do not provide the interface this code was written for, or (cfg given) do not
    implement the site definition of the run: the kernel builds the sites of 0.10 (complement pairs by
    sequence, strands summed, shared positions by the motifs' record counts), so --shared-sites motif,
    --strand-sites auto or split, and a --complement-pairs rule that pairs this motif list differently are read
    by the Python pass, which is the reference."""
    if os.environ.get("EPIREFINE_RS", "1").strip().lower() in ("0", "false", "no", "off"):
        return None
    if cfg is not None:
        if cfg.get("group_rank") is not None or cfg.get("strand_sites", "sum") != "sum":
            return None
        if not same_pairs(cfg["motifs"], cfg.get("pairing", "sequence")):
            return None
    try:
        import epirefine_rs
    except ImportError:
        return None
    return epirefine_rs if epirefine_rs.supports("api>=1") and epirefine_rs.supports("pileup_reader") else None


def _run_range_rs(rs, cfg: dict, path: str, start: int, end: int, out_prefix: str, flush_rows: int):
    """_run_range with the compiled reader: the same site rows, part files of whole contigs and bookkeeping."""
    asm = EncodedAssembly(cfg["asm_dir"])
    reader = rs.PileupReader(path, start, end, names=asm.names, offsets=asm.offsets, lengths=asm.lengths,
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


def _run_range(cfg: dict, path: str, start: int, end: int, out_prefix: str, block: int, flush_rows: int):
    rs = _rs(cfg)
    if rs is not None:
        return _run_range_rs(rs, cfg, path, start, end, out_prefix, flush_rows)
    w = _Worker(cfg)
    seen: set = set()
    parts, buf, n_buf = [], [], 0
    # rows of the last contig seen, which may continue in the next block: kept as the pieces each block
    # added and joined once when the contig ends. Joined, compared and filtered again at every block, a
    # contig spanning B blocks was copied B(B+1)/2 times (91 block copies instead of 13 for 10 Mbp).
    carry, carry_name = [], None
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
        last = df["contig"][-1]
        if first_contig is None:
            first_contig = df["contig"][0]
        tail = df["contig"] == last
        rest, cur = (df.clear(), df) if tail.all() else (df.filter(~tail), df.filter(tail))
        if carry and carry_name == last:  # the carried contig goes on (its rows stay ahead of this block's)
            carry.append(cur)
            df = rest
        else:                             # it ended: its rows and this block's other contigs are complete
            df = pl.concat(carry + [rest]) if carry else rest
            carry, carry_name = [cur], last
        if df.height:
            s = w.process(df, seen)
            if s is not None:
                buf.append(s); n_buf += s.height
        if n_buf >= flush_rows:
            flush()
    if carry:
        last_contig = carry_name
        s = w.process(pl.concat(carry), seen)
        if s is not None:
            buf.append(s)
    flush()
    return {"parts": parts, "contigs": sorted(seen), "first": first_contig, "last": last_contig}


def extract_sites(asm: EncodedAssembly, pileup, motifs: list[Motif], group_of: list[int], motif_rank,
                  out_dir, wanted: set[int] | None = None, workers: int | None = None,
                  block: int = 128 << 20, flush_rows: int = 20_000_000, log=print, group_rank=None,
                  pairing: str = "sequence", strand_sites: str = "sum") -> list[str]:
    """Run the pileup pass. Returns the Parquet part files (each holds whole contigs)."""
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    workers = workers or min(16, _cores())
    ranges = split_pileup(pileup, workers * 4)
    cfg = {"asm_dir": asm.directory, "motifs": [(m.seq, m.mod_pos, m.mod_type) for m in motifs],
           "group_of": list(group_of), "motif_rank": list(motif_rank), "wanted": wanted,
           "group_rank": list(group_rank) if group_rank is not None else None,
           "pairing": pairing, "strand_sites": strand_sites}
    rs = _rs(cfg)
    log(f"  pileup pass: {len(ranges)} byte ranges on {min(workers, len(ranges))} workers"
        + (f" (epirefine-rs {rs.__version__} reader)" if rs is not None else ""))
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
            note_worker_peaks()
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


def _fragment_rows(cid: np.ndarray, pos: np.ndarray, fr: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(unit, row) for every site row inside a calibration fragment (contig, start <= pos < end).

    A site can lie in several fragments (of different lengths): one pair per (site, fragment), rows
    ascending and the fragments of a row in the order of `fr`, which is the order of the join it
    replaces. A part is sorted by (contig, pos), so the rows of a fragment are one interval found by
    binary search. Joining sites to fragments on the contig and filtering on position afterwards
    paired every site of a contig with every fragment of it first: 330-360 million rows (14-16 GB)
    for a part of complete genomes with 18 fragments each, of which under 4 million survived.
    """
    fc, fs, fe, fu = (fr[c].to_numpy() for c in ("contig", "start", "end", "unit"))
    if len(cid) > 1 and not (cid[1:] >= cid[:-1]).all():
        lo = hi = None  # not sorted by contig: fall back to the join
    else:
        lo, hi = np.searchsorted(cid, fc, "left"), np.searchsorted(cid, fc, "right")
        for a, b in set(zip(lo.tolist(), hi.tolist())):
            if b - a > 1 and not (pos[a + 1:b] >= pos[a:b - 1]).all():
                lo = None
                break
    if lo is None:
        f = (pl.DataFrame({"row": np.arange(len(cid)), "contig": cid, "pos": pos})
             .join(fr.select("unit", "contig", "start", "end"), on="contig", how="inner")
             .filter((pl.col("pos") >= pl.col("start")) & (pl.col("pos") < pl.col("end"))))
        return f["unit"].to_numpy(), f["row"].to_numpy()
    rows, units = [], []
    for j in np.flatnonzero(hi > lo):
        p = pos[lo[j]:hi[j]]
        a, b = lo[j] + np.searchsorted(p, fs[j], "left"), lo[j] + np.searchsorted(p, fe[j], "left")
        if b > a:
            rows.append(np.arange(a, b)); units.append(np.full(b - a, fu[j], fu.dtype))
    if not rows:
        return np.zeros(0, fu.dtype), np.zeros(0, np.int64)
    rows, units = np.concatenate(rows), np.concatenate(units)
    order = np.argsort(rows, kind="stable")
    return units[order], rows[order]


# (group, n) -> expected per-site log-likelihood rows (assignability.expected_tables), kept for the life of
# a worker process: they depend only on the caller model and the grid, and filling them is a Python loop
# of 0.5 ms per combination that every part repeated for the 110,000-285,000 combinations it holds
_EXPECTED: dict = {}


def _unit_part(part: str, cfg: dict) -> dict:
    from .assignability import expected_tables
    from .detection import DetectionParams
    from .grid import Grid
    from .model import unit_loglik_arrays

    grid = Grid(cfg["grid_size"])
    det = {g: DetectionParams(*p) for g, p in cfg["det"].items()}
    M = cfg["n_groups"]
    # (one block per column, so that the numpy arrays below are views of the table and not copies next to it)
    df = (pl.read_parquet(part).filter((pl.col("n") >= cfg["min_cov"]) & ~pl.col("group").is_in(cfg.get("drop_groups", [])))
          .rechunk())
    cid, pos = df["contig"].to_numpy(), df["pos"].to_numpy()
    grp, n, k = df["group"].to_numpy(), df["n"].to_numpy(), df["k"].to_numpy()
    del df

    def tables(units, rows, with_h):
        ids, local = np.unique(units, return_inverse=True)
        del units
        local = local.astype(np.int32) if len(ids) < 2 ** 31 else local
        g_, n_, k_ = (grp, n, k) if len(rows) == len(grp) and not with_h else (grp[rows], n[rows], k[rows])
        res = {"ids": ids, "L": unit_loglik_arrays(local, g_, n_, k_, len(ids), M, det, grid, dtype=np.float32),
               "n_sites": np.bincount(local, minlength=len(ids))}
        count = np.bincount(local.astype(np.int64) * M + g_, minlength=len(ids) * M)  # sites per (unit, group)
        cell = np.flatnonzero(count)
        res["n_sites_group"] = (ids[cell // M], (cell % M).astype(np.int32), count[cell].astype(np.uint32))
        if with_h:  # expected-evidence tables: calibration fragments only
            key = (cfg["grid_size"], tuple(sorted(cfg["det"].items())))
            if key not in _EXPECTED:
                _EXPECTED.clear()  # another model: one table per process
                _EXPECTED[key] = {}
            res["H"] = expected_tables(pl.DataFrame({"unit": local, "group": g_, "n": n_}), len(ids), M, det, grid,
                                       dtype=np.float32, table=_EXPECTED[key])
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
        units, rows = _fragment_rows(cid, pos, fr)
        if len(rows):
            out["frags"] = tables(units, rows, True)
    return out


def unit_tables(parts: list[str], *, n_groups: int, det: dict, grid_size: int, min_cov: int, drop_groups=(),
                chunk_offset: np.ndarray, chunk_size: int, n_chunks: int, target_of: np.ndarray, n_targets: int,
                frags: pl.DataFrame | None, workers: int | None = None, log=print) -> dict:
    """Likelihood tables for host chunks, targets and calibration fragments, in parallel over parts.

    Returns float32 arrays L_chunks [K, M, G], L_t [U, M, G], L_frag, H_frag = (H1, H0), and site
    counts per chunk, per target, per (target, group) and per fragment.
    """
    G = grid_size
    cfg = {"grid_size": G, "n_groups": n_groups, "min_cov": min_cov, "chunk_size": chunk_size, "drop_groups": list(drop_groups),
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
    workers = workers or min(16, _cores())
    env = {k: os.environ.get(k) for k in ("POLARS_MAX_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
    os.environ.update({k: "1" for k in env})
    try:
        with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
            # results are taken as they complete: in submission order, the tables of parts that finished
            # behind a slow one waited unpickled in this process. Parts hold whole contigs, so each
            # chunk, target and fragment is written by exactly one part and the order does not matter.
            pending, i = {ex.submit(_unit_part, p, cfg) for p in parts}, 0
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                while done:
                    r = done.pop().result()
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
                    i += 1
                    if i % max(1, len(parts) // 5) == 0:
                        log(f"  unit tables: {i}/{len(parts)} parts")
                    del r
            note_worker_peaks()
    finally:
        for k, v in env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return res
