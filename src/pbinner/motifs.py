"""Motif handling: IUPAC matching on both strands, site definition, and grouping of motifs.

Two kinds of double counting are removed here:

1. Strands. For a palindromic motif the + and - strand modifications at one occurrence are one
   site (same molecule, same MTase event) and their counts are summed. A motif and its reverse
   complement (e.g. Type I bipartite motif + its complement) are canonicalised so that their
   records at one occurrence also share a site.
2. Nested/overlapping motifs (e.g. GATC and RGATCY). Motifs of the same mod type whose site
   sets overlap strongly are merged into one *group*; every (contig, pos, strand, mod_type) record
   is counted once. Groups are the features of the model.

Matching is done on a 2-bit encoded sequence, one vectorised pass per motif *shape* (the offsets
of a motif's non-N bases) for all motifs and orientations sharing it, so the cost grows with the
number of shapes rather than the number of motifs.
"""
from __future__ import annotations

import itertools
import re
from dataclasses import dataclass

import numpy as np
import polars as pl

IUPAC = {
    "A": "A", "C": "C", "G": "G", "T": "T", "R": "[AG]", "Y": "[CT]", "S": "[CG]", "W": "[AT]",
    "K": "[GT]", "M": "[AC]", "B": "[CGT]", "D": "[AGT]", "H": "[ACT]", "V": "[ACG]", "N": "[ACGT]",
}
_COMP = str.maketrans("ACGTRYSWKMBDHVN", "TGCAYRSWMKVHDBN")
MOD_BASE = {"a": "A", "m": "C", "h": "C", "21839": "C"}
_CODE = np.full(256, 4, np.uint8)
for _i, _c in enumerate("ACGT"):
    _CODE[ord(_c)] = _CODE[ord(_c.lower())] = _i
_PREFIX = 10  # fixed bases resolved with a dense lookup table; the rest are checked on candidates


def revcomp(seq: str) -> str:
    return seq.translate(_COMP)[::-1]


@dataclass(frozen=True)
class Motif:
    seq: str
    mod_pos: int
    mod_type: str

    @property
    def key(self) -> str:
        return f"{self.seq}_{self.mod_type}-{self.mod_pos}"

    @property
    def is_palindrome(self) -> bool:
        return revcomp(self.seq) == self.seq

    def is_valid(self) -> bool:
        base = MOD_BASE.get(self.mod_type)
        return 0 <= self.mod_pos < len(self.seq) and (base is None or self.seq[self.mod_pos] == base)

    def regex(self, rc: bool = False) -> re.Pattern:
        s = revcomp(self.seq) if rc else self.seq
        return re.compile("(?=(" + "".join(IUPAC[c] for c in s) + "))")


def encode(seq: str) -> np.ndarray:
    """A,C,G,T -> 0..3; anything else (N, IUPAC, separators) -> 4."""
    return _CODE[np.frombuffer(seq.encode("ascii"), np.uint8)]


class MotifMatcher:
    """All occurrences of a motif set on both strands, one vectorised pass per motif shape.

    Each (motif, orientation) is a pattern: the motif read on + (orientation 0) or its reverse
    complement read on + (orientation 1, i.e. the motif on -). Patterns with the same span and
    non-N offsets share a shape; degenerate bases are expanded into concrete variants.
    """

    def __init__(self, motifs: list[Motif]):
        self.motifs = motifs
        canon = canonical_index(motifs)
        self.canon = np.array(canon, np.int32)
        self.flip = np.array([canon[i] != i for i in range(len(motifs))])
        self.mod_pos = np.array([m.mod_pos for m in motifs], np.int64)
        self.length = np.array([len(m.seq) for m in motifs], np.int64)
        self.palin = np.array([m.is_palindrome for m in motifs])
        by_shape: dict[tuple, list] = {}
        for mi, m in enumerate(motifs):
            for orient, seq in ((0, m.seq), (1, revcomp(m.seq))):
                offs = tuple(i for i, c in enumerate(seq) if c != "N")
                choices = [IUPAC[seq[o]].strip("[]") for o in offs]
                if np.prod([len(c) for c in choices], dtype=float) > 1 << 16:
                    raise ValueError(f"motif {m.key} expands to too many concrete sequences")
                for combo in itertools.product(*choices):
                    code = 0
                    for c in combo:
                        code = code * 4 + "ACGT".index(c)
                    by_shape.setdefault((len(seq), offs), []).append((code, mi, orient))
        self.shapes = []
        for (span, offs), ents in by_shape.items():
            ents.sort()
            codes = np.array([e[0] for e in ents], np.uint64)
            ucodes, start = np.unique(codes, return_index=True)
            k1 = min(len(offs), _PREFIX)
            lut = np.zeros(4 ** k1, bool)
            lut[(ucodes >> np.uint64(2 * (len(offs) - k1))).astype(np.int64)] = True
            self.shapes.append({
                "span": span, "offs": np.array(offs, np.int64), "k1": k1, "lut": lut, "ucodes": ucodes,
                "ptr": np.r_[start, len(ents)].astype(np.int64),
                "mi": np.array([e[1] for e in ents], np.int32), "orient": np.array([e[2] for e in ents], np.int8)})

    def _scan(self, codes: np.ndarray, shape: dict):
        """Start positions and entry indices of every match of one shape."""
        n = len(codes) - shape["span"] + 1
        if n <= 0:
            return np.empty(0, np.int64), np.empty(0, np.int64)
        offs, k1 = shape["offs"], shape["k1"]
        key = np.zeros(n, np.int64 if k1 > 15 else np.int32)
        bad = np.zeros(n, bool)
        for o in offs[:k1]:
            c = codes[o:o + n]
            key <<= 2
            key |= c & 3
            bad |= c == 4
        cand = np.flatnonzero(shape["lut"][key] & ~bad)
        if len(offs) > k1 and len(cand):
            full = key[cand].astype(np.uint64)
            ok = np.ones(len(cand), bool)
            for o in offs[k1:]:
                c = codes[cand + o]
                full = (full << np.uint64(2)) | (c & 3).astype(np.uint64)
                ok &= c != 4
            cand, full = cand[ok], full[ok]
        else:
            full = key[cand].astype(np.uint64)
        j = np.searchsorted(shape["ucodes"], full)
        hit = j < len(shape["ucodes"])
        hit[hit] = shape["ucodes"][j[hit]] == full[hit]
        cand, j = cand[hit], j[hit]
        lo, hi = shape["ptr"][j], shape["ptr"][j + 1]
        cnt = hi - lo
        starts = np.repeat(cand, cnt)
        ent = np.repeat(lo - np.cumsum(np.r_[0, cnt[:-1]]), cnt) + np.arange(cnt.sum())
        return starts, ent

    def find(self, codes: np.ndarray) -> dict[str, np.ndarray]:
        """Records in one encoded sequence: pos, strand (0 '+', 1 '-'), motif, occ_motif, occ.

        occ = occurrence start * 2 + orientation of the canonical motif; palindromes share occ on
        both strands, and a motif and its reverse complement share occ at one occurrence.
        """
        parts = []
        for sh in self.shapes:
            starts, ent = self._scan(codes, sh)
            if len(starts):
                parts.append((starts, sh["mi"][ent], sh["orient"][ent]))
        if not parts:
            e = np.empty(0, np.int64)
            return {"pos": e, "strand": e.astype(np.int8), "motif": e.astype(np.int32),
                    "occ_motif": e.astype(np.int32), "occ": e}
        start = np.concatenate([p[0] for p in parts])
        mi = np.concatenate([p[1] for p in parts])
        orient = np.concatenate([p[2] for p in parts])
        fwd = orient == 0
        pos = np.where(fwd, start + self.mod_pos[mi], start + self.length[mi] - 1 - self.mod_pos[mi])
        occ = start * 2 + np.where(fwd | self.palin[mi], 0, 1)
        flip = self.flip[mi]
        occ[flip] = (occ[flip] // 2) * 2 + (1 - occ[flip] % 2)
        return {"pos": pos, "strand": orient, "motif": mi, "occ_motif": self.canon[mi], "occ": occ}

    def find_many(self, contigs: dict[str, str], batch_bp: int = 20_000_000):
        """Yield (contig names, contig index per record, record dict) for batches of contigs.

        Contigs of a batch are concatenated with an N separator so per-contig overhead is paid once
        per batch; positions are mapped back to contig coordinates.
        """
        sep = max(int(self.length.max()), 1) if len(self.length) else 1
        names, seqs, size = [], [], 0
        items = list(contigs.items())
        for i, (name, seq) in enumerate(items):
            names.append(name); seqs.append(seq); size += len(seq) + sep
            if size >= batch_bp or i == len(items) - 1:
                offsets = np.cumsum([0] + [len(x) + sep for x in seqs])[:-1]
                rec = self.find(encode(("N" * sep).join(seqs)))
                cidx = np.searchsorted(offsets, rec["pos"], side="right") - 1
                rec["pos"] = rec["pos"] - offsets[cidx]
                rec["occ"] = rec["occ"] - 2 * offsets[cidx]
                yield names, cidx, rec
                names, seqs, size = [], [], 0


def find_records(seq: str, motif: Motif) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Modified-base records of one motif in one sequence: (pos, strand, occ), strand 0='+', 1='-'.

    occ = occurrence start * 2 + orientation (0 = motif read on +, 1 = on -); for palindromes
    both strands share occ.
    """
    rec = MotifMatcher([motif]).find(encode(seq))
    return rec["pos"], rec["strand"], rec["occ"]


def canonical_index(motifs: list[Motif]) -> list[int]:
    """Index of the canonical motif for each motif: a motif and its reverse complement share one."""
    by_key = {(m.seq, m.mod_type): i for i, m in enumerate(motifs)}
    out = []
    for i, m in enumerate(motifs):
        j = by_key.get((revcomp(m.seq), m.mod_type), i)
        out.append(min(i, j))
    return out


def build_records(contigs: dict[str, str], motifs: list[Motif], matcher: MotifMatcher | None = None) -> pl.DataFrame:
    """All motif records in the assembly, before grouping (one row per motif x record)."""
    matcher = matcher or MotifMatcher(motifs)
    mod_types = pl.Series([m.mod_type for m in motifs])
    frames = []
    for names, cidx, rec in matcher.find_many(contigs):
        if not len(rec["pos"]):
            continue
        frames.append(pl.DataFrame({
            "contig": pl.Series(names)[cidx], "pos": rec["pos"], "strand": rec["strand"],
            "mod_type": mod_types[rec["motif"]], "motif": rec["motif"], "occ_motif": rec["occ_motif"],
            "occ": rec["occ"]}, schema_overrides={"motif": pl.Int32, "occ_motif": pl.Int32}))
    if not frames:
        raise ValueError("no motif occurrences found in the assembly")
    return pl.concat(frames)


def group_motifs(records: pl.DataFrame, motifs: list[Motif], min_containment: float = 0.5) -> list[int]:
    """Union-find over motifs: same mod type and (reverse-complement pair or site containment >= x)."""
    n = len(motifs)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(i)] = find(j)

    canon = canonical_index(motifs)
    for i, c in enumerate(canon):
        union(i, c)

    # containment from shared records: count, per motif pair, the records both match (one pass
    # over records matched by >= 2 motifs) instead of intersecting every pair of record sets
    contig_ids = {c: i for i, c in enumerate(records["contig"].unique().to_list())}
    keyed = records.select(
        (pl.col("contig").replace_strict(contig_ids, return_dtype=pl.Int64) * (1 << 34)
         + pl.col("pos") * 2 + pl.col("strand").cast(pl.Int64)).alias("rkey"), "motif").unique()
    sizes = dict(keyed.group_by("motif").len().iter_rows())
    multi = keyed.filter(pl.len().over("rkey") > 1)
    pairs = (multi.join(multi, on="rkey", suffix="_b").filter(pl.col("motif") < pl.col("motif_b"))
             .group_by("motif", "motif_b").len())
    for i, j, inter in pairs.iter_rows():
        if motifs[i].mod_type == motifs[j].mod_type and inter / min(sizes[i], sizes[j]) >= min_containment:
            union(i, j)
    # number groups by their first member, so the numbering does not depend on union order
    first: dict[int, int] = {}
    for i in range(n):
        first.setdefault(find(i), i)
    rid = {r: g for g, r in enumerate(sorted(first, key=first.get))}
    return [rid[find(i)] for i in range(n)]


def assign_sites(records: pl.DataFrame, group_of: list[int], motif_rank: dict[int, int] | None = None) -> pl.DataFrame:
    """Deduplicate records across motifs and give each a (group, site) id.

    Records matching motifs of *different* groups (e.g. the A in TCGANTC, shared by TCGA and
    GANTC) are dropped: their methylation can come from either MTase, which would leak one
    group's signal into the other. Within a group, a record shared by several motifs is claimed
    by the motif with the most records (most general). `motif_rank` (motif -> record count in the
    whole assembly or a sample of it) keeps that choice consistent when records come in batches.
    """
    keys = ["contig", "pos", "strand", "mod_type"]
    if motif_rank is None:
        sizes = records.group_by("motif").len().rename({"len": "motif_n"})
    else:
        sizes = pl.DataFrame({"motif": list(motif_rank), "motif_n": list(motif_rank.values())},
                             schema={"motif": pl.Int32, "motif_n": pl.Int64})
    rec = (
        records.join(sizes, on="motif", how="left").with_columns(pl.col("motif_n").fill_null(0))
        .with_columns(pl.col("motif").replace_strict(dict(enumerate(group_of)), return_dtype=pl.Int32).alias("group"))
        .filter(pl.col("group").n_unique().over(keys) == 1)
        .sort(["motif_n", "motif"], descending=[True, False])
        .unique(subset=["contig", "pos", "strand", "mod_type"], keep="first", maintain_order=True)
        .with_columns((pl.col("occ_motif").cast(pl.Int64) * (1 << 40) + pl.col("occ")).alias("site"))
        .with_columns(pl.when(pl.col("strand") == 0).then(pl.lit("+")).otherwise(pl.lit("-")).alias("strand"))
        .select("contig", "pos", "strand", "mod_type", "group", "site")
    )
    return rec


def describe_groups(motifs: list[Motif], group_of: list[int], sites: pl.DataFrame | None = None) -> pl.DataFrame:
    rows = []
    for g in sorted(set(group_of)):
        members = [motifs[i] for i in range(len(motifs)) if group_of[i] == g]
        rows.append({"group": g, "mod_type": members[0].mod_type,
                     "motifs": ",".join(m.key for m in members)})
    df = pl.DataFrame(rows, schema_overrides={"group": pl.Int32})
    if sites is not None:
        w = pl.col("w") if "w" in sites.columns else pl.lit(1)
        counts = sites.group_by("group").agg(w.sum().alias("n_sites"), (pl.col("n") * w).sum().alias("n_reads"))
        df = df.join(counts, on="group", how="left")
    return df
