"""The synthetic community generator writes what the readers expect, and the same thing for the same seed."""
import hashlib

import polars as pl

import synth
from epirefine import io as pio
from epirefine.sites import _parse

SMALL = {g: (30_000 if g != "E" else 12_000, b, m) for g, (_, b, m) in synth.GENOMES.items()}


def _digest(com):
    return [hashlib.sha256(p.read_bytes()).hexdigest() for p in (com.assembly, com.pileup, com.contig_bin, com.motifs)]


def test_same_seed_same_files(tmp_path):
    a = synth.make_community(tmp_path / "a", seed=5, genomes=SMALL, extras=False)
    b = synth.make_community(tmp_path / "b", seed=5, genomes=SMALL, extras=False)
    c = synth.make_community(tmp_path / "c", seed=6, genomes=SMALL, extras=False)
    assert _digest(a) == _digest(b)
    assert _digest(a)[:2] != _digest(c)[:2]


def test_files_are_readable_by_epirefine(community):
    com = community
    assert pio.read_fasta(com.assembly) == com.seqs
    assert pio.read_contig_bin(com.contig_bin) == com.given
    keys = {m.key for m in pio.read_motifs(com.motifs)}
    # every motif a binned genome methylates, with the complements of the two Type I motifs; AGCT (only the
    # binless genome has it) is not listed
    assert {"GATC_a-1", "CCWGG_m-1", "AACNNNNNNGTGC_a-1", "GCACNNNNNNGTT_a-2", "CAAGNNNNNNCTG_a-2"} <= keys
    assert len(keys) == 14 and "AGCT_m-2" not in keys
    pile = _parse(com.pileup.read_bytes())
    assert pile.columns == ["contig", "pos", "mod_type", "strand", "n_valid", "n_mod"]
    assert pile["contig"].unique(maintain_order=True).to_list() == list(com.seqs)  # assembly order, contiguous
    assert set(pile["mod_type"].unique()) == {"a", "m", "h"} and set(pile["strand"].unique()) == {"+", "-"}
    assert (pile["n_valid"] >= 1).all() and (pile["n_mod"] <= pile["n_valid"]).all()
    # 18 tab-separated bedMethyl columns, modified base matches the row's code and strand
    first = com.pileup.read_text().split("\n", 1)[0].split("\t")
    assert len(first) == 18
    for contig, pos, code, strand in pile.head(300).select("contig", "pos", "mod_type", "strand").iter_rows():
        base = com.seqs[contig][pos]
        assert base == {("a", "+"): "A", ("a", "-"): "T", ("m", "+"): "C", ("m", "-"): "G",
                        ("h", "+"): "C", ("h", "-"): "G"}[(code, strand)]


def test_methylation_follows_the_genome(community):
    """GATC sites are methylated on contigs of genome A and not on contigs of genome B."""
    com = community
    pile = _parse(com.pileup.read_bytes()).filter(pl.col("mod_type") == "a")
    for genome, lo, hi in (("A", 0.8, 1.0), ("B", 0.0, 0.1)):
        contig = next(c for c in com.seqs if com.genome_of[c] == genome and len(com.seqs[c]) >= 20_000)
        plus, minus = synth.motif_sites(com.seqs[contig], "GATC", 1)
        rows = pile.filter((pl.col("contig") == contig) & pl.col("pos").is_in(list(plus) + list(minus)))
        assert rows.height == len(plus) + len(minus) > 50
        assert lo <= rows["n_mod"].sum() / rows["n_valid"].sum() <= hi


def test_planted_truth_is_consistent(community):
    com = community
    assert len(com.contaminants) == 5 and len(com.missing) == 6 and len(com.orphans) >= 3
    for c, home in com.contaminants.items():
        assert com.given[c] != home and com.true_bin[c] == home
    assert sum(home is None for home in com.contaminants.values()) == 1
    for c, home in com.missing.items():
        assert c not in com.given and com.true_bin[c] == home
    assert all(c not in com.given and com.true_bin[c] is None for c in com.orphans)
    assert len(com.seqs[com.short]) < 1000 and com.short not in com.given
    assert com.given[com.low_coverage] == com.true_bin[com.low_coverage]
    assert com.seqs[com.duplicate] == com.seqs[com.duplicate_of] == com.seqs[com.duplicate_binned]
    assert com.duplicate not in com.given and com.given[com.duplicate_binned] != com.true_bin[com.duplicate_binned]
    assert [c for c, b in com.given.items() if b == "binE"] == [com.sole]
