"""The production site path (sites._Worker.records / _sites / extract_sites) against the reference
implementation (motifs.assign_sites) and against hand-worked pileups."""
import numpy as np
import polars as pl

from epirefine.motifs import (Motif, MotifMatcher, assign_sites, build_records, can_share_site, canonical_index,
                              encode, expected_frequency, group_motifs, revcomp, specificity_ranks)
from epirefine.sites import MOD_CODE, EncodedAssembly, _Worker, extract_sites

MOTIFS = [Motif("GATC", 1, "a"), Motif("RGATCY", 2, "a"), Motif("CCWGG", 1, "m"), Motif("CCWGG", 1, "21839"),
          Motif("TCGA", 3, "a"), Motif("GANTC", 1, "a"), Motif("GAGNNNNNGTCA", 1, "a"), Motif("TGACNNNNNCTC", 2, "a"),
          Motif("GGATG", 2, "a"), Motif("CATCC", 1, "a"), Motif("GCWGC", 1, "m")]


def _assembly(seed=0, n=3, length=20000):
    rng = np.random.default_rng(seed)
    out = {}
    for i in range(n):
        s = "".join(rng.choice(list("ACGT"), length))
        # plant the long bipartite motif on both strands and a TCGANTC overlap
        s = s[:500] + "GAGACGTAGTCA" + s[512:900] + revcomp("GAGTTTTTGTCA") + s[912:1500] + "TCGAATC" + s[1507:]
        out[f"c{i}"] = s
    return out


def _worker(tmp_path, contigs, motifs, group_of, motif_rank, **kw):
    asm = EncodedAssembly.write(contigs, tmp_path / "asm")
    cfg = {"asm_dir": asm.directory, "motifs": [(m.seq, m.mod_pos, m.mod_type) for m in motifs],
           "group_of": list(group_of), "motif_rank": list(motif_rank), "wanted": None, **kw}
    return asm, _Worker(cfg)


def _decode(rec: pl.DataFrame, names: list[str]) -> pl.DataFrame:
    key = rec["key"].to_numpy()
    code_mod = {v: k for k, v in MOD_CODE.items()}
    return pl.DataFrame({"contig": [names[i] for i in key >> 36], "pos": (key >> 3) & ((1 << 33) - 1),
                         "strand": np.where((key >> 2) & 1, "-", "+"), "mod_type": [code_mod[m] for m in key & 3],
                         "group": rec["group"], "site": rec["site"]})


def _partition(df: pl.DataFrame) -> set:
    """Sites as sets of (contig, pos, strand, mod type) records, with their group."""
    out = {}
    for c, p, s, m, g, site in df.select("contig", "pos", "strand", "mod_type", "group", "site").iter_rows():
        out.setdefault((c, site, g), set()).add((c, p, s, m))
    return {(g, frozenset(v)) for (c, site, g), v in out.items()}


def test_worker_records_match_assign_sites(tmp_path):
    contigs = _assembly()
    recs = build_records(contigs, MOTIFS)
    group_of = group_motifs(recs, MOTIFS)
    rank_d = dict(recs.group_by("motif").len().iter_rows())
    rank = [int(rank_d.get(i, 0)) for i in range(len(MOTIFS))]
    ref = assign_sites(recs, group_of, {i: r for i, r in enumerate(rank)})
    asm, w = _worker(tmp_path, contigs, MOTIFS, group_of, rank)
    got = _decode(w.records(list(range(len(asm.names)))), asm.names)
    assert got.height == ref.height == recs.unique(["contig", "pos", "strand", "mod_type"]).height
    assert _partition(got) == _partition(ref)
    # the same records in two batches of contigs (ids are per batch) give the same sites
    two = pl.concat([_decode(w.records([0, 1]), asm.names[:2]), _decode(w.records([2]), asm.names[2:])])
    assert _partition(two) == _partition(ref)


def test_worker_records_match_assign_sites_with_motif_ranks(tmp_path):
    contigs = _assembly(seed=1)
    recs = build_records(contigs, MOTIFS)
    group_of = group_motifs(recs, MOTIFS)
    motif_rank, group_rank = specificity_ranks(MOTIFS, group_of)
    ref = assign_sites(recs, group_of, dict(enumerate(motif_rank)), group_rank)
    asm, w = _worker(tmp_path, contigs, MOTIFS, group_of, motif_rank, group_rank=group_rank)
    got = _decode(w.records(list(range(len(asm.names)))), asm.names)
    assert _partition(got) == _partition(ref)


def test_specificity_from_the_motif_not_the_sample():
    assert expected_frequency(Motif("GATC", 1, "a")) == 1 / 256
    assert expected_frequency(Motif("RGATCY", 2, "a")) == 1 / 1024
    assert expected_frequency(Motif("GAANNNNNNTGC", 2, "a")) == 4.0 ** -6
    motifs = [Motif("TCGA", 3, "a"), Motif("GANTC", 1, "a"), Motif("GATC", 1, "a"), Motif("RGATCY", 2, "a"),
              Motif("GAGNNNNNGTCA", 1, "a"), Motif("TGACNNNNNCTC", 2, "a"), Motif("GACBNNNNBGTCB", 1, "a")]
    group_of = [0, 1, 2, 3, 4, 4, 5]
    motif_rank, group_rank = specificity_ranks(motifs, group_of)
    # nested: RGATCY is more specific than GATC; equal frequency: the key decides (GANTC < TCGA), always
    assert group_rank[3] < group_rank[2] and group_rank[1] < group_rank[0]
    # a complement pair is as frequent as one of its motifs, not twice: the pair (7 fixed bases) stays more
    # specific than a lone motif whose frequency is 1.69 times that of one motif of the pair
    assert group_rank[4] < group_rank[5]
    # the order does not depend on the row order of the motif table
    perm = [6, 3, 0, 5, 2, 1, 4]
    m2, g2 = [motifs[i] for i in perm], [group_of[i] for i in perm]
    mr2, gr2 = specificity_ranks(m2, g2)
    assert [mr2[perm.index(i)] for i in range(7)] == motif_rank
    assert [gr2[g] for g in range(6)] == group_rank
    # TCGANTC: the shared A (pos 3, +) goes to GANTC in every sequence, whatever the counts in it
    rng = np.random.default_rng(3)
    for _ in range(6):
        seq = "".join(rng.choice(list("ACGT"), 30000)) + "TTTCGAATCTT"
        recs = build_records({"c": seq}, motifs[:2])
        sites = assign_sites(recs, [0, 1], *[dict(enumerate(r)) if i == 0 else r
                                              for i, r in enumerate(specificity_ranks(motifs[:2], [0, 1]))])
        shared = sites.filter((pl.col("pos") == 30000 + 5) & (pl.col("strand") == "+"))
        assert shared["group"].to_list() == [1]


def _pileup_line(contig, pos, mod, strand, n, k):
    rest = "\t".join(str(x) for x in [n, f"{k / max(n, 1):.2f}", k, n - k, 0, 0, 0, 0, 0])
    return f"{contig}\t{pos}\t{pos + 1}\t{mod}\t{n}\t{strand}\t{pos}\t{pos + 1}\t255,0,0\t{rest}\n"


def test_extract_sites_hand_worked(tmp_path):
    #          0         1         2         3
    #          0123456789012345678901234567890123456
    seq = "TTGATCTTTTGGATGTTTTTTTGAGACGTAGTCATTT"
    contigs = {"c1": seq, "c2": "TTTTGATCTTTT"}
    motifs = [Motif("GATC", 1, "a"), Motif("GGATG", 2, "a"), Motif("CATCC", 1, "a"),
              Motif("GAGNNNNNGTCA", 1, "a"), Motif("TGACNNNNNCTC", 2, "a")]
    recs = build_records(contigs, motifs)
    group_of = group_motifs(recs, motifs)
    assert group_of == [0, 1, 1, 2, 2]
    rank = [recs.filter(pl.col("motif") == i).height for i in range(len(motifs))]
    rows = [("c1", 3, "a", "+", 10, 9), ("c1", 4, "a", "-", 8, 8),          # GATC: A on +, A on - (T at 4)
            ("c1", 12, "a", "+", 6, 6), ("c1", 13, "a", "-", 4, 0),         # GGATG + strand, CATCC - strand (hemi)
            ("c1", 23, "a", "+", 5, 5), ("c1", 31, "a", "-", 7, 6),         # GAGN5GTCA and its complement (T at 31)
            ("c1", 0, "a", "-", 9, 9), ("c1", 3, "m", "+", 9, 9),           # not motif positions / wrong mod type
            ("c2", 5, "a", "+", 3, 1), ("c2", 6, "a", "-", 0, 0)]           # second contig; a row without valid reads
    pile = tmp_path / "pileup.bed"
    pile.write_text("".join(_pileup_line(*r) for r in rows))
    asm = EncodedAssembly.write(contigs, tmp_path / "asm")
    parts = extract_sites(asm, pile, motifs, group_of, rank, tmp_path / "sites", workers=1, log=lambda *_: None)
    got = pl.concat([pl.read_parquet(p) for p in parts]).sort("contig", "pos")
    assert got.select("contig", "group", "pos", "n", "k").rows() == [
        (0, 0, 3, 18, 17),   # the GATC occurrence: both strands summed, position of the first record
        (0, 1, 12, 10, 6),   # GGATG/CATCC merged: a hemimethylated occurrence reads 0.6
        (0, 2, 23, 12, 11),  # the Type I pair
        (1, 0, 5, 3, 1)]

    def run(tag, **kw):
        recs = build_records(contigs, motifs, pairing=kw.get("pairing", "sequence"))
        g = group_motifs(recs, motifs, pairing=kw.get("pairing", "sequence"))
        parts = extract_sites(asm, pile, motifs, g, rank, tmp_path / tag, workers=1, log=lambda *_: None, **kw)
        return g, pl.concat([pl.read_parquet(p) for p in parts]).sort("contig", "pos").select("contig", "group", "pos", "n", "k").rows()

    # one site per strand, groups unchanged: the hemimethylated occurrence is one methylated and one unmethylated site
    g, rows_split = run("split", strand_sites="split")
    assert g == [0, 1, 1, 2, 2]
    assert rows_split == [(0, 0, 3, 10, 9), (0, 0, 4, 8, 8), (0, 1, 12, 6, 6), (0, 1, 13, 4, 0),
                          (0, 2, 23, 5, 5), (0, 2, 31, 7, 6), (1, 0, 5, 3, 1)]
    # complement motifs as motifs of their own: single-strand sites in separate groups; the palindrome still sums
    g, rows_sep = run("separate", pairing="separate")
    assert g == [0, 1, 2, 3, 4]
    assert rows_sep == [(0, 0, 3, 18, 17), (0, 1, 12, 6, 6), (0, 2, 13, 4, 0), (0, 3, 23, 5, 5), (0, 4, 31, 7, 6),
                        (1, 0, 5, 3, 1)]
    # unambiguous pairs: "position" pairs exactly as "sequence" does
    assert run("position", pairing="position") == ([0, 1, 1, 2, 2], got.select("contig", "group", "pos", "n", "k").rows())


def test_two_modification_types_on_one_cytosine_stay_two_sites_when_split(tmp_path):
    contigs = {"c": "TTCCAGGTT"}
    motifs = [Motif("CCWGG", 1, "m"), Motif("CCWGG", 1, "21839")]
    asm = EncodedAssembly.write(contigs, tmp_path / "asm")
    pile = tmp_path / "pileup.bed"
    pile.write_text("".join(_pileup_line(*r) for r in [("c", 3, "m", "+", 10, 9), ("c", 3, "21839", "+", 10, 1),
                                                        ("c", 5, "m", "-", 8, 7), ("c", 5, "21839", "-", 8, 0)]))
    for mode, want in (("sum", [(0, 3, 18, 16), (1, 3, 18, 1)]),
                       ("split", [(0, 3, 10, 9), (0, 5, 8, 7), (1, 3, 10, 1), (1, 5, 8, 0)])):
        parts = extract_sites(asm, pile, motifs, [0, 1], [1, 1], tmp_path / mode, workers=1, log=lambda *_: None,
                              strand_sites=mode)
        got = pl.concat([pl.read_parquet(p) for p in parts]).sort("group", "pos")
        assert got.select("group", "pos", "n", "k").rows() == want


def test_complement_pairing_with_two_modified_positions():
    a2, a1, c2 = Motif("GAANNNNNNTGC", 2, "a"), Motif("GAANNNNNNTGC", 1, "a"), Motif("GCANNNNNNTTC", 2, "a")
    # by sequence: the outcome depends on the row order, and one order sums two adenines of one strand
    assert canonical_index([c2, a2, a1]) == [0, 0, 0]
    assert canonical_index([a2, a1, c2]) == [0, 1, 1]
    seq = "TTGAACCCCCCTGCTT"
    rec = MotifMatcher([c2, a2, a1]).find(encode(seq))
    assert len(set(zip(rec["occ_motif"].tolist(), rec["occ"].tolist()))) == 1  # A at 3, A at 4 and the complement A
    # by position: the ambiguous pair is not merged, in any order; three sites
    for order in ([c2, a2, a1], [a2, a1, c2], [a1, c2, a2]):
        assert canonical_index(order, "position") == [0, 1, 2]
        rec = MotifMatcher(order, "position").find(encode(seq))
        assert len(set(zip(rec["occ_motif"].tolist(), rec["occ"].tolist()))) == 3
    # unambiguous pairs and palindromes are as before; a palindrome with two positions keeps two site sets
    ms = [Motif("GAGNNNNNGTCA", 1, "a"), Motif("TGACNNNNNCTC", 2, "a"), Motif("CCWGG", 0, "m"), Motif("CCWGG", 1, "m"),
          Motif("GATC", 1, "a")]
    assert canonical_index(ms, "position") == canonical_index(ms) == [0, 0, 2, 3, 4]
    assert canonical_index(ms, "separate") == [0, 1, 2, 3, 4]


def test_can_share_site_reads_both_motifs_on_the_modified_strand():
    assert can_share_site(Motif("GCWGC", 1, "m"), Motif("CGCWGCR", 2, "m"))
    assert not can_share_site(Motif("GATC", 1, "a"), Motif("CCWGG", 1, "m"))
    assert not can_share_site(Motif("GATC", 1, "a"), Motif("GTAC", 2, "a"))
    # a motif and its reverse complement modify different bases: never one site
    assert not can_share_site(Motif("GAANNNNNNTGC", 2, "a"), Motif("GCANNNNNNTTC", 2, "a"))
    # known limit: two Type I motifs with one half-site in common can share a site (GAANNNNNNTGCTG)
    assert can_share_site(Motif("GAANNNNNNTGC", 2, "a"), Motif("GAANNNNNNNNCTG", 2, "a"))


def test_auto_splits_strands_only_where_they_can_differ(tmp_path):
    #      0123456789012345678901234567890123456
    seq = "TTATCGATCTTTTTTGGATCAATTTTTGGATGTTTT"
    assert seq[2:8] == "ATCGAT" and seq[5:9] == "GATC" and seq[16:20] == "GATC" and seq[27:32] == "GGATG"
    contigs = {"c": seq}
    motifs = [Motif("GATC", 1, "a"), Motif("ATCGAT", 4, "a"), Motif("GGATG", 2, "a"), Motif("CATCC", 1, "a")]
    group_of = [0, 1, 2, 2]
    rank = [8, 2, 1, 1]
    asm = EncodedAssembly.write(contigs, tmp_path / "asm")
    # a Dam genome without the ATCGAT enzyme: every GATC adenine is methylated, the ATCGAT adenine outside GATC is not
    rows = [("c", 3, "a", "-", 10, 0),   # ATCGAT, - strand: not a GATC position
            ("c", 6, "a", "+", 10, 10),  # ATCGAT, + strand: also the GATC adenine of ATCGATC
            ("c", 7, "a", "-", 10, 10),  # that GATC, - strand
            ("c", 17, "a", "+", 9, 9), ("c", 18, "a", "-", 11, 11),  # a plain GATC
            ("c", 29, "a", "+", 6, 6), ("c", 30, "a", "-", 4, 0)]    # GGATG methylated, CATCC not
    pile = tmp_path / "pileup.bed"
    pile.write_text("".join(_pileup_line(*r) for r in rows))

    def run(mode):
        parts = extract_sites(asm, pile, motifs, group_of, rank, tmp_path / mode, workers=1, log=lambda *_: None,
                              strand_sites=mode)
        return pl.concat([pl.read_parquet(p) for p in parts]).sort("pos").select("group", "pos", "n", "k").rows()

    # summed: the ATCGAT occurrence and the GGATG/CATCC occurrence read one half
    assert run("sum") == [(1, 3, 20, 10), (0, 7, 10, 10), (0, 17, 20, 20), (2, 29, 10, 6)]
    # auto: those two occurrences are one site per strand; the plain GATC stays one two-strand site
    assert run("auto") == [(1, 3, 10, 0), (1, 6, 10, 10), (0, 7, 10, 10), (0, 17, 20, 20), (2, 29, 6, 6), (2, 30, 4, 0)]
    assert run("split") == [(1, 3, 10, 0), (1, 6, 10, 10), (0, 7, 10, 10), (0, 17, 9, 9), (0, 18, 11, 11),
                            (2, 29, 6, 6), (2, 30, 4, 0)]
    # auto marks the single-strand halves of two-strand occurrences (for the per-occurrence contig coverage); the
    # lone - strand record of the GATC occurrence whose + strand went to ATCGAT is not a half. sum writes no column.
    auto = pl.concat([pl.read_parquet(p) for p in sorted((tmp_path / "auto").glob("*.parquet"))]).sort("pos")
    assert auto["half"].to_list() == [1, 1, 0, 0, 1, 1]
    assert "half" not in pl.read_parquet(sorted((tmp_path / "sum").glob("*.parquet"))[0]).columns
