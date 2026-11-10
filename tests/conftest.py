"""A small synthetic community for end-to-end tests: three genomes with different methylated motifs, each a
bin of a few contigs, plus unbinned contigs; modkit-style pileup rows at the motif sites only."""
import re

import numpy as np
import pytest

MOTIFS = (("GATC", "a", 1), ("CCWGG", "m", 1), ("GCATGC", "a", 2))
_RE = {"GATC": "GATC", "CCWGG": "CC[AT]GG", "GCATGC": "GCATGC"}
# methylated motifs per genome
GENOMES = {"g1": {"GATC"}, "g2": {"CCWGG"}, "g3": {"GATC", "GCATGC"}}


def write_community(root, n_contigs=4, contig_len=30_000, coverage=20, seed=0, single_bin=False):
    """Write assembly.fasta, pileup.bed, contig_bin.tsv, motifs.tsv into root; returns their paths by name.

    The last contig of every genome is left unbinned. single_bin: only genome g1 is binned."""
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    fasta, pileup, cb = [], [], []
    for g, meth in GENOMES.items():
        for k in range(n_contigs):
            name = f"{g}_c{k}"
            seq = "".join(rng.choice(list("ACGT"), contig_len))
            fasta.append(f">{name}\n{seq}\n")
            if k < n_contigs - 1 and (g == "g1" or not single_bin):
                cb.append(f"{name}\t{g}\n")
            rows = []
            for motif, mod, pos in MOTIFS:
                code = mod
                for m in re.finditer(f"(?={_RE[motif]})", seq):
                    # palindromes: the modified base on + and its partner on -
                    for strand, p in (("+", m.start() + pos), ("-", m.start() + len(motif) - 1 - pos)):
                        nmod = rng.binomial(coverage, 0.95 if motif in meth else 0.02)
                        rows.append((p, f"{name}\t{p}\t{p + 1}\t{code}\t{coverage}\t{strand}\t{p}\t{p + 1}\t255,0,0\t"
                                        f"{coverage}\t{100 * nmod / coverage:.2f}\t{nmod}\t{coverage - nmod}\t0\t0\t0\t0\t0\n"))
            pileup.extend(r for _, r in sorted(rows))
    paths = {k: root / v for k, v in (("assembly", "assembly.fasta"), ("pileup", "pileup.bed"),
                                      ("contig_bin", "contig_bin.tsv"), ("motifs", "motifs.tsv"))}
    paths["assembly"].write_text("".join(fasta))
    paths["pileup"].write_text("".join(pileup))
    paths["contig_bin"].write_text("".join(cb))
    paths["motifs"].write_text("motif\tmod_type\tmod_position\n" + "".join(f"{m}\t{t}\t{p}\n" for m, t, p in MOTIFS))
    return paths


@pytest.fixture(scope="session")
def community(tmp_path_factory):
    return write_community(tmp_path_factory.mktemp("community"))


def run_community(paths, out, **kw):
    from epirefine.pipeline import run
    kw = {"workers": 1, "chunk_size": 10_000, "frag_lengths": (2_000, 5_000), "frags_per_bin": 6, **kw}
    return run(paths["assembly"], paths["pileup"], paths["contig_bin"], paths["motifs"], out, **kw)
