"""The production site path (sites._Worker.records / _sites / extract_sites) against the reference
implementation (motifs.assign_sites) and against hand-worked pileups."""
import numpy as np
import polars as pl

from epirefine.motifs import Motif, assign_sites, build_records, group_motifs, revcomp
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
