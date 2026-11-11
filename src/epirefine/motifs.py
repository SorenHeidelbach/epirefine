"""Motif handling: IUPAC matching on both strands, site definition, and grouping of motifs.

Two kinds of double counting are removed here:

1. Strands. For a palindromic motif the + and - strand modifications at one occurrence are one
   site (same molecule, same MTase event) and their counts are summed. A motif and its reverse
   complement (e.g. Type I bipartite motif + its complement) are canonicalised so that their
   records at one occurrence also share a site.
2. Overlapping motifs. Motifs of the same mod type whose site sets overlap strongly (Jaccard >=
   min_jaccard, 0.9 by default: a motif and its complement, near-identical spellings) are merged into
   one *group*. A motif nested in a more general one (RGATCY in GATC, Jaccard about 0.25) stays a
   group of its own; every (contig, pos, strand, mod_type) record is counted once, in the most
   specific group that matches it. Groups are the features of the model.

Matching is done on a 2-bit encoded sequence, one vectorised pass per motif *shape* (the offsets
of a motif's non-N bases) for all motifs and orientations sharing it, so the cost grows with the
number of shapes rather than the number of motifs.
"""
from __future__ import annotations

import itertools
import os
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

    def __init__(self, motifs: list[Motif], pairing: str = "sequence"):
        self.motifs = motifs
        self.pairing = pairing
        self._compiled = False  # the scanner of epirefine-rs for these motifs, or None (set on first use)
        canon = canonical_index(motifs, pairing)
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

    def _scanner(self):
        """The compiled motif scan of the optional package epirefine-rs for these motifs (the same records in
        the same order as find, tested there and in tests/test_rs_model.py), or None: package not installed,
        switched off with EPIREFINE_RS=0, or a pairing of complements it does not implement."""
        if self._compiled is False:
            self._compiled = None
            if self.pairing == "sequence" and os.environ.get("EPIREFINE_RS", "1").strip().lower() not in ("0", "false", "no", "off"):
                try:
                    import epirefine_rs
                    if epirefine_rs.supports("api>=1") and epirefine_rs.supports("motif_scan"):
                        self._compiled = epirefine_rs.MotifScanner([(m.seq, m.mod_pos, m.mod_type) for m in self.motifs])
                except ImportError:
                    pass
        return self._compiled

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
                codes = encode(("N" * sep).join(seqs))
                scanner = self._scanner()
                rec = self.find(codes) if scanner is None else scanner.find(codes)
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


PAIRINGS = ("sequence", "position", "separate")


def canonical_index(motifs: list[Motif], pairing: str = "sequence") -> list[int]:
    """Index of the canonical motif for each motif: a motif and its reverse complement share one, so that
    their records at one occurrence are one two-strand site (a Type I motif and its complement: one enzyme,
    both strands).

    pairing:
      "sequence"  a motif is paired with the last listed motif that has the reverse-complement sequence and the
                  same modification type, whatever their modified positions. If a sequence is listed with two
                  positions (GAAN6TGC a-1 and a-2) next to its complement, which of them pair, and whether all three
                  collapse into one site (two adenines of one strand summed), depends on the row order.
      "position"  a motif and its complement are paired only when each is listed with exactly one modified position
                  for that modification type; otherwise every entry keeps its own sites. Independent of row order,
                  and two positions of one strand are never summed.
      "separate"  no pairing: a motif and its complement are different motifs with their own single-strand sites
                  (and their own groups), so a genome that methylates one strand only (GGATG without CATCC) reads
                  on / off instead of 0.5 at merged sites.
    """
    if pairing not in PAIRINGS:
        raise ValueError(f"pairing must be one of {PAIRINGS}, not {pairing!r}")
    if pairing == "separate":
        return list(range(len(motifs)))
    if pairing == "sequence":
        by_key = {(m.seq, m.mod_type): i for i, m in enumerate(motifs)}
        out = []
        for i, m in enumerate(motifs):
            j = by_key.get((revcomp(m.seq), m.mod_type), i)
            out.append(min(i, j))
        return out
    by_seq: dict[tuple, list[int]] = {}
    for i, m in enumerate(motifs):
        by_seq.setdefault((m.seq, m.mod_type), []).append(i)
    out = []
    for i, m in enumerate(motifs):
        own, comp = by_seq[(m.seq, m.mod_type)], by_seq.get((revcomp(m.seq), m.mod_type), [])
        out.append(min(i, comp[0]) if not m.is_palindrome and len(own) == 1 and len(comp) == 1 else i)
    return out


def build_records(contigs: dict[str, str], motifs: list[Motif], matcher: MotifMatcher | None = None,
                  pairing: str = "sequence") -> pl.DataFrame:
    """All motif records in the assembly, before grouping (one row per motif x record)."""
    matcher = matcher or MotifMatcher(motifs, pairing)
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


def group_motifs(records: pl.DataFrame, motifs: list[Motif], min_jaccard: float = 0.9,
                 pairing: str = "sequence") -> list[int]:
    """Union-find over motifs: same mod type and (reverse-complement pair or near-identical site sets).

    Motifs are merged only when they describe the same sites (Jaccard >= min_jaccard), not when one
    contains the other: merging by containment chained short degenerate motifs (CCNGG, CCAG, ...) with
    everything they contain into groups of up to 53-70 motifs, of which a bin methylates one, so the
    group's methylated fraction measured which member a genome carries rather than whether an MTase
    is active. Positions matched by several groups go to the most specific one (assign_sites).
    """
    n = len(motifs)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(i)] = find(j)

    canon = canonical_index(motifs, pairing)
    for i, c in enumerate(canon):
        union(i, c)

    # overlap from shared records: count, per motif pair, the records both match (one pass over
    # records matched by >= 2 motifs) instead of intersecting every pair of record sets
    contig_ids = {c: i for i, c in enumerate(records["contig"].unique().to_list())}
    keyed = records.select(
        (pl.col("contig").replace_strict(contig_ids, return_dtype=pl.Int64) * (1 << 34)
         + pl.col("pos") * 2 + pl.col("strand").cast(pl.Int64)).alias("rkey"), "motif").unique()
    sizes = dict(keyed.group_by("motif").len().iter_rows())
    multi = keyed.filter(pl.len().over("rkey") > 1)
    pairs = (multi.join(multi, on="rkey", suffix="_b").filter(pl.col("motif") < pl.col("motif_b"))
             .group_by("motif", "motif_b").len())
    for i, j, inter in pairs.iter_rows():
        if motifs[i].mod_type == motifs[j].mod_type and inter / (sizes[i] + sizes[j] - inter) >= min_jaccard:
            union(i, j)
    # number groups by their first member, so the numbering does not depend on union order
    first: dict[int, int] = {}
    for i in range(n):
        first.setdefault(find(i), i)
    rid = {r: g for g, r in enumerate(sorted(first, key=first.get))}
    return [rid[find(i)] for i in range(n)]


def group_sizes(group_of: list[int], motif_rank) -> list[int]:
    """Records per group (sum over its motifs): the 'specificity' order of assign_sites."""
    out = [0] * (max(group_of) + 1)
    for g, r in zip(group_of, motif_rank):
        out[g] += int(r)
    return out


def expected_frequency(m: Motif) -> float:
    """Probability that a position of a random sequence (uniform bases) starts an occurrence of the motif:
    the product over its positions of (allowed bases / 4). A property of the motif alone."""
    p = 1.0
    for c in m.seq:
        p *= len(IUPAC[c].strip("[]")) / 4
    return p


def specificity_ranks(motifs: list[Motif], group_of: list[int]) -> tuple[list[int], list[int]]:
    """(motif_rank, group_rank) for assign_sites, computed from the motifs themselves.

    A position matched by several groups goes to the most specific one. Measured by record counts in a sample of
    the assembly, "most specific" is a coin flip for motifs of similar frequency that are not nested (TCGA and
    GANTC share the A of TCGANTC, 1/16 of each motif's sites; their counts differ by 0.2-3% between samples, so
    the 6% changed sides with the seed), and a motif absent from the sample won every position. Here the order
    is the expected frequency of the motif in random sequence, ties broken by the motif key, so it depends on
    neither the sample, the seed nor the row order of the motif table. A nested motif always has a lower expected
    frequency than the motif that contains it, so nested motifs are ordered as before.

    motif_rank: position of each motif in ascending (expected frequency, key) order; larger = more general.
    group_rank: position of each group in ascending order of (its most general member's expected frequency,
    its smallest key); smaller = more specific. A group's members describe the same sites (near-identical site
    sets, or a motif and its reverse complement), so the group is as frequent as its most general member, not
    the sum of its members.
    """
    order = sorted(range(len(motifs)), key=lambda i: (expected_frequency(motifs[i]), motifs[i].key))
    motif_rank = [0] * len(motifs)
    for r, i in enumerate(order):
        motif_rank[i] = r
    n_groups = max(group_of) + 1
    members = [[i for i in range(len(motifs)) if group_of[i] == g] for g in range(n_groups)]
    gkey = [(max(expected_frequency(motifs[i]) for i in mem), min(motifs[i].key for i in mem)) for mem in members]
    group_rank = [0] * n_groups
    for r, g in enumerate(sorted(range(n_groups), key=lambda g: gkey[g])):
        group_rank[g] = r
    return motif_rank, group_rank


def assign_sites(records: pl.DataFrame, group_of: list[int], motif_rank: dict[int, int] | None = None,
                 group_rank: list[int] | None = None) -> pl.DataFrame:
    """Deduplicate records across motifs and give each a (group, site) id.

    A record matching motifs of *different* groups (e.g. the A in TCGANTC, shared by TCGA and GANTC;
    or a CCWGG site, which CCNGG also matches) goes to the most specific of them, the group with the
    fewest records. Its methylation can come from either MTase, but the specific group's sites then
    measure "either", and the general group's remaining sites measure the general MTase alone, so
    both stay informative; dropping such records lost half of all positions on ZymoFecal. Within a
    group, a record shared by several motifs is claimed by the motif with the most records (most
    general). `motif_rank` (motif -> record count in the whole assembly or a sample of it) keeps both
    choices consistent when records come in batches. With `group_rank` (specificity_ranks) the order of
    the groups is taken from it instead of from summed record counts.
    """
    keys = ["contig", "pos", "strand", "mod_type"]
    if motif_rank is None:
        sizes = records.group_by("motif").len().rename({"len": "motif_n"})
    else:
        sizes = pl.DataFrame({"motif": list(motif_rank), "motif_n": list(motif_rank.values())},
                             schema={"motif": pl.Int32, "motif_n": pl.Int64})
    rank = dict(sizes.iter_rows())
    gsize = list(group_rank) if group_rank is not None else group_sizes(group_of, [rank.get(i, 0) for i in range(len(group_of))])
    rec = (
        records.join(sizes, on="motif", how="left").with_columns(pl.col("motif_n").fill_null(0))
        .with_columns(pl.col("motif").replace_strict(dict(enumerate(group_of)), return_dtype=pl.Int32).alias("group"))
        .with_columns(pl.col("group").replace_strict(dict(enumerate(gsize)), return_dtype=pl.Int64).alias("group_n"))
        .sort(["group_n", "group", "motif_n", "motif"], descending=[False, False, True, False])
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


def _bases(c: str) -> set:
    return set(IUPAC[c].strip("[]"))


def can_share_site(a: Motif, b: Motif) -> bool:
    """True when a and b (same modification type) can both match one modified base: aligned on their
    modified positions, every overlapping position has a base both allow. Such motifs can be one
    methyltransferase's sites (GCWGC, CGCWGCR, YGGCAGCA); unrelated motifs of one genome (GATC, CCWGG) cannot.

    Both motifs are read on the strand of the modified base: a modified base is one (position, strand), so b on
    the other strand can never share it (its base there is the complement, T or G). Positions of one motif beyond
    the end of the other, or on its N positions, are unconstrained, so two Type I motifs with one half-site in
    common (GAAN6TGC, GAAN8CTG) can share a site too (1/16 of their sites); the co-methylation test of
    redundancy_weights, not this function, has to tell such enzymes apart."""
    if a.mod_type != b.mod_type or not (_bases(b.seq[b.mod_pos]) & _bases(a.seq[a.mod_pos])):
        return False
    off = a.mod_pos - b.mod_pos
    return all(not (0 <= i + off < len(a.seq)) or bool(_bases(a.seq[i + off]) & _bases(c)) for i, c in enumerate(b.seq))


def redundancy_weights(counts_bin: "pl.DataFrame", motifs: list[Motif], group_of: list[int], n_groups: int, *,
                       min_reads: int = 5, min_sites: int = 50, min_common_bins: int = 5,
                       min_jaccard: float = 0.9) -> tuple[np.ndarray, "pl.DataFrame"]:
    """Weights that stop one methyltransferase from being counted once per motif variant.

    Groups are linked when (1) some motif of one can share a site with some motif of the other
    (can_share_site) and (2) they are methylated in the same bins: over the bins where both are
    measured (>= min_sites sites with >= min_reads reads; at least min_common_bins of them), the bins
    with a majority of methylated sites agree with Jaccard >= min_jaccard, and at least 2 bins methylate
    them. Linked groups form clusters (single linkage); each group gets weight 1 / cluster size, so a
    cluster contributes the evidence of one group. On ZymoFecal, 19 GC[AT]GC-type 5mC groups behaved as
    one MTase and dominated contig-bin mismatches. Unlinked groups keep weight 1.

    counts_bin: columns group, bin, n, k, w (sites per pattern, as for the caller fit).
    Returns (weights [n_groups], cluster table).
    """
    import polars as pl
    st = (counts_bin.filter(pl.col("n") >= min_reads)
          .group_by("group", "bin").agg(pl.col("w").sum().alias("sites"),
                                        pl.col("w").filter(pl.col("k") / pl.col("n") > 0.5).sum().alias("meth"))
          .filter(pl.col("sites") >= min_sites)
          .with_columns((pl.col("meth") / pl.col("sites") > 0.5).alias("on")))
    meas = {g: dict(zip(d["bin"].to_list(), d["on"].to_list())) for (g,), d in st.group_by("group")}
    cand = [g for g, m in meas.items() if sum(m.values()) >= 2]
    members = {g: [motifs[i] for i in range(len(motifs)) if group_of[i] == g] for g in cand}
    parent = {g: g for g in cand}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    cand.sort()
    for i, a in enumerate(cand):
        for b in cand[i + 1:]:
            ma, mb = meas[a], meas[b]
            common = ma.keys() & mb.keys()
            if len(common) < min_common_bins:
                continue
            on_a = {x for x in common if ma[x]}; on_b = {x for x in common if mb[x]}
            u = on_a | on_b
            if len(u) < 2 or len(on_a & on_b) / len(u) < min_jaccard:
                continue
            if any(can_share_site(x, y) for x in members[a] for y in members[b]):
                parent[find(a)] = find(b)
    clusters: dict = {}
    for g in cand:
        clusters.setdefault(find(g), []).append(g)
    w = np.ones(n_groups)
    rows = []
    for root, gs in clusters.items():
        for g in gs:
            w[g] = 1.0 / len(gs)
            rows.append({"group": g, "cluster": min(gs), "cluster_size": len(gs), "weight": 1.0 / len(gs)})
    tab = pl.DataFrame(rows, schema={"group": pl.Int32, "cluster": pl.Int32, "cluster_size": pl.Int64, "weight": pl.Float64}).sort("cluster", "group")
    return w, tab
def noisy_groups(parts: list[str], contig_bin: "pl.DataFrame", contigs: list[int], *, min_reads: int = 5,
                 min_sites: int = 20, partial=(0.15, 0.6), min_mid_sites: float = 0.2, min_bin_share: float = 0.2,
                 min_noise_bins: int = 3, max_off_bin_share: float = 1.0) -> "pl.DataFrame":
    """Motif groups that look like caller noise: intermediate read-level calls at the sites themselves,
    in a large share of genomes.

    A real motif is methylated at (nearly) all of a genome's sites or at none. A group can still show a
    partial contig mean for real reasons, and those must be kept: a genome that methylates RGATCY but not
    every GATC has its GATC sites at either ~1 or ~0, with a mix set by sequence composition; a bipartite
    motif can follow a genome's Dam status. What marks noise is that the *sites* are intermediate (each
    site has some reads called methylated), on most genomes. On ZymoFecal, 4mC calls on CCNGG-type
    contexts did this in 20-81% of bins (mean fraction 0.15-0.3, a third of sites intermediate): most
    likely 5mC at those sites leaking into the 4mC call.

    Per (contig, group), over the given binned contigs (cache ids, typically those >= 10x): sites with
    >= min_reads reads; the contig is noise-like for the group when it has >= min_sites such sites, their
    mean read fraction lies in `partial` and >= min_mid_sites of them have a read fraction in (0.2, 0.8).
    A bin is noise-like when at least half its measured contigs are. A group is noisy when noise-like
    bins are >= min_bin_share of the bins that measure it and at least min_noise_bins. Unbinned contigs
    are not used (a genome must be identifiable to count). Returns one row per measured group.

    The rule cannot tell a real motif that the caller detects poorly (intermediate where it is methylated, zero
    elsewhere) from a leak (intermediate nearly everywhere). `off_bin_share`, the share of measuring bins whose
    level is below 0.05, tells them apart: on ZymoFecal the eight dropped 4mC groups have 1-8% of their bins off,
    while a Dam-linked complement pair with 18% noise-like bins (one strand methylated, summed sites at 0.5) has 63%.
    With max_off_bin_share < 1 a group is only called noise when at most that share of its bins is off (default 1:
    not used).
    """
    import polars as pl
    lo, hi = partial
    r = pl.col("k") / pl.col("n")
    cg = (pl.scan_parquet(parts).filter(pl.col("contig").is_in(contigs) & (pl.col("n") >= min_reads))
          .group_by("contig", "group").agg(pl.len().alias("sites"), r.mean().alias("mean_fraction"),
                                           ((r > 0.2) & (r < 0.8)).mean().alias("mid_sites"))
          .filter(pl.col("sites") >= min_sites).collect())
    cg = cg.join(contig_bin, on="contig", how="inner").with_columns(
        (pl.col("mean_fraction").is_between(lo, hi) & (pl.col("mid_sites") >= min_mid_sites)).alias("noise_like"))
    b = cg.group_by("group", "bin").agg(pl.col("noise_like").mean().alias("share"), pl.col("mean_fraction").median().alias("level"))
    g = (b.group_by("group").agg(pl.len().alias("bins"), (pl.col("share") >= 0.5).sum().alias("noise_bins"),
                                 pl.col("level").median().alias("median_level"),
                                 (pl.col("level") < 0.05).mean().alias("off_bin_share"))
         .with_columns((pl.col("noise_bins") / pl.col("bins")).alias("noise_bin_share"))
         .with_columns(((pl.col("noise_bin_share") >= min_bin_share) & (pl.col("noise_bins") >= min_noise_bins)
                        & (pl.col("off_bin_share") <= max_off_bin_share)).alias("noisy"))
         .sort("group"))
    return g
