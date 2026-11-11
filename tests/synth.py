"""A small synthetic community for end-to-end tests: genomes with distinct methylation, contigs cut from
them, a modkit pileup simulated directly at the motif sites (no reads), and a binning with planted errors.

Nothing here uses EpiRefine's own motif matching: sites are found with regular expressions, so the files
are an independent input for the pipeline rather than an echo of it.

The community (default options):

    genome  bin    methylated motifs                                     role
    A       binA   GATC, TCGA (6mA), CCWGG, GCGC (5mC)                   bin
    B       binB   GANTC, CATG, Type I pair AACNNNNNNGTGC /              bin
                   GCACNNNNNNGTT (6mA), CCGG (5mC)
    C       binC   CTGCAG, GTAC, Type I pair CAGNNNNNNCTTG /             bin
                   CAAGNNNNNNCTG (6mA), CCWGG, ACGT (5mC)
    D       -      GATC, CTGCAG, CATG (a combination no bin has) and     genome without a bin
                   AGCT (5mC, in no bin, so not in the motif table)
    E       binE   GATC, GANTC, GTAC                                     bin of one contig

Planted on top of the true binning (see `Community` for the lists):
  * contaminants: contigs of another binned genome, and one of the binless genome D, put in a bin;
  * missing members: contigs of A, B, C left out of contig_bin (they should be included);
  * every other contig of D is unbinned and should stay so;
  * one unbinned contig shorter than 1 kb, one binned contig at ~2x coverage, an unbinned exact copy of
    a binned contig, and a second copy of it placed in a wrong bin (a contaminant whose way back to its own
    bin is closed by the redundancy check).

Everything is a deterministic function of `seed`. Files: `assembly.fasta`, `pileup.bed` (modkit bedMethyl,
18 tab-separated columns, contigs in assembly order, rows sorted by position), `contig_bin.tsv` (contig, bin;
no header) and `bin-motifs.tsv` (Nanomotif columns, with the complement of the Type I motif).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_IUPAC = {"A": "A", "C": "C", "G": "G", "T": "T", "R": "[AG]", "Y": "[CT]", "S": "[CG]", "W": "[AT]",
          "K": "[GT]", "M": "[AC]", "B": "[CGT]", "D": "[AGT]", "H": "[ACT]", "V": "[ACG]", "N": "[ACGT]"}
_COMP = str.maketrans("ACGTRYSWKMBDHVN", "TGCAYRSWMKVHDBN")

# (motif, position of the modified base, modkit code, complement motif, its modified position)
GATC = ("GATC", 1, "a", None, None)
TCGA = ("TCGA", 3, "a", None, None)
CATG = ("CATG", 1, "a", None, None)
GTAC = ("GTAC", 2, "a", None, None)
GANTC = ("GANTC", 1, "a", None, None)
CTGCAG = ("CTGCAG", 4, "a", None, None)
CCWGG = ("CCWGG", 1, "m", None, None)
GCGC = ("GCGC", 1, "m", None, None)
CCGG = ("CCGG", 1, "m", None, None)
ACGT = ("ACGT", 1, "m", None, None)
AGCT = ("AGCT", 2, "m", None, None)
TYPE_I = ("AACNNNNNNGTGC", 1, "a", "GCACNNNNNNGTT", 2)
TYPE_I2 = ("CAGNNNNNNCTTG", 1, "a", "CAAGNNNNNNCTG", 2)
MOTIFS = (GATC, TCGA, CATG, GTAC, GANTC, CTGCAG, CCWGG, GCGC, CCGG, ACGT, AGCT, TYPE_I, TYPE_I2)

# genome -> (length, bin or None, methylated motifs)
GENOMES = {
    "A": (300_000, "binA", (GATC, TCGA, CCWGG, GCGC)),
    "B": (300_000, "binB", (GANTC, CATG, TYPE_I, CCGG)),
    "C": (300_000, "binC", (CTGCAG, GTAC, TYPE_I2, CCWGG, ACGT)),
    "D": (160_000, None, (GATC, CTGCAG, CATG, AGCT)),
    "E": (40_000, "binE", (GATC, GANTC, GTAC)),
}
_CONTIG_LENGTHS = (4_000, 6_000, 9_000, 12_000, 14_000, 18_000, 22_000, 28_000, 35_000, 60_000)


def revcomp(seq: str) -> str:
    return seq.translate(_COMP)[::-1]


def motif_sites(seq: str, motif: str, mod_pos: int) -> tuple[np.ndarray, np.ndarray]:
    """0-based positions of the modified base of every occurrence of `motif`, on + and on - (as + coordinates)."""
    fwd = re.compile("(?=" + "".join(_IUPAC[c] for c in motif) + ")")
    rc = revcomp(motif)
    rev = re.compile("(?=" + "".join(_IUPAC[c] for c in rc) + ")")
    plus = np.array([m.start() + mod_pos for m in fwd.finditer(seq)], np.int64)
    minus = np.array([m.start() + len(motif) - 1 - mod_pos for m in rev.finditer(seq)], np.int64)
    return plus, minus


@dataclass
class Community:
    directory: Path
    assembly: Path
    pileup: Path
    contig_bin: Path
    motifs: Path
    seqs: dict[str, str]                      # contig -> sequence, in assembly order
    genome_of: dict[str, str]                 # contig -> genome letter
    given: dict[str, str]                     # contig -> bin in contig_bin.tsv
    true_bin: dict[str, str | None]           # contig -> the bin of its genome (None: genome without a bin)
    contaminants: dict[str, str | None] = field(default_factory=dict)  # planted contaminant -> its true bin
    missing: dict[str, str] = field(default_factory=dict)              # unbinned member -> the bin it belongs to
    orphans: list[str] = field(default_factory=list)                   # unbinned contigs of the binless genome
    short: str = ""                           # unbinned contig < 1 kb
    low_coverage: str = ""                    # binned contig at ~2x
    duplicate: str = ""                       # unbinned exact copy of a binned contig
    duplicate_of: str = ""
    duplicate_binned: str = ""                # a second copy of duplicate_of, placed in another genome's bin
    sole: str = ""                            # the only contig of binE

    def args(self, out, **kw) -> dict:
        """Keyword arguments for epirefine.pipeline.run."""
        return dict(assembly=str(self.assembly), pileup=str(self.pileup), contig_bin=str(self.contig_bin),
                    motifs_path=str(self.motifs), out=str(out), **kw)

    def cli_args(self, out, *extra) -> list[str]:
        return ["run", "--assembly", str(self.assembly), "--pileup", str(self.pileup), "--contig-bin",
                str(self.contig_bin), "--motifs", str(self.motifs), "--out", str(out), *map(str, extra)]


def _cut(length: int, rng) -> list[int]:
    """Contig lengths that add up to `length`."""
    out, left = [], length
    while left > 0:
        n = int(rng.choice(_CONTIG_LENGTHS))
        n = left if left - n < 3_000 else n
        out.append(min(n, left)); left -= out[-1]
    return out


def pileup_rows(name: str, seq: str, methylated, rng, coverage: float = 15.0, background: float = 0.004,
                motifs=MOTIFS) -> list[str]:
    """modkit bedMethyl rows of one contig: every site of every listed motif on both strands, plus a thin
    background of other A and C positions (a real pileup has a row for every covered A and C).

    Per strand, n_valid ~ 1 + Poisson(coverage - 1). At a site of a motif the genome methylates, 97% of the
    sites are methylated; the fraction of reads called modified is Beta(mean 0.90) at a methylated site and
    Beta(mean 0.03) otherwise, and n_mod ~ Binomial(n_valid, fraction). 6mA rows carry code `a`; cytosine
    positions get an `m` row and an `h` row, as modkit writes them.
    """
    meth_keys = {(m[0], m[2]) for m in methylated}
    site: dict[tuple[int, str, str], bool] = {}
    for motif, pos, code, comp, comp_pos in motifs:
        pairs = [(motif, pos)] + ([(comp, comp_pos)] if comp else [])
        for mseq, mpos in pairs:
            plus, minus = motif_sites(seq, mseq, mpos)
            for strand, where in (("+", plus), ("-", minus)):
                for p in where:
                    key = (int(p), strand, code)
                    site[key] = site.get(key, False) or (motif, code) in meth_keys
    comp_base = {"A": "T", "C": "G"}
    for code, base in (("a", "A"), ("m", "C")):
        for strand, b in (("+", base), ("-", comp_base[base])):
            where = np.flatnonzero(np.frombuffer(seq.encode(), np.uint8) == ord(b))
            for p in where[rng.random(len(where)) < background]:
                site.setdefault((int(p), strand, code), False)
    rows = []
    for (p, strand, code) in sorted(site):
        n = 1 + int(rng.poisson(max(coverage - 1, 0.0)))
        is_meth = site[(p, strand, code)] and rng.random() < 0.97
        frac = rng.beta(18, 2) if is_meth else rng.beta(0.6, 19.4)
        k = int(rng.binomial(n, frac))
        codes = [(code, k)] if code == "a" else [("m", k), ("h", int(rng.binomial(n - k, 0.01)))]
        for c, kk in codes:
            other = k if c == "h" else codes[1][1] if c == "m" else 0  # reads called as the other C mod
            canon = n - kk - other
            rows.append(f"{name}\t{p}\t{p + 1}\t{c}\t{n}\t{strand}\t{p}\t{p + 1}\t255,0,0\t{n}\t"
                        f"{100 * kk / n:.2f}\t{kk}\t{canon}\t{other}\t0\t0\t0\t0\n")
    return rows


def make_community(directory, seed: int = 1, coverage: float = 15.0, genomes=GENOMES, motifs=MOTIFS,
                   extras: bool = True) -> Community:
    """Write the four input files into `directory` and return their paths with the planted truth.

    `extras=False` leaves out the short, low-coverage and duplicated contigs (a plain community for the
    edge-case tests, which build their own oddities).
    """
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    pieces = []  # (genome, sequence)
    for g, (length, _, _) in genomes.items():
        genome = "".join(rng.choice(list("ACGT"), size=length))
        cuts = [length] if g == "E" else _cut(length, rng)
        start = 0
        for n in cuts:
            pieces.append((g, genome[start:start + n])); start += n
    # names carry a number unrelated to the genome, so that sorting by name mixes the genomes
    order = rng.permutation(len(pieces))
    seqs, genome_of = {}, {}
    for number, i in enumerate(order):
        g, s = pieces[i]
        name = f"c{number:03d}_{g}"
        seqs[name], genome_of[name] = s, g
    true_bin = {c: genomes[g][1] for c, g in genome_of.items()}
    given = {c: b for c, b in true_bin.items() if b is not None}
    com = Community(d, d / "assembly.fasta", d / "pileup.bed", d / "contig_bin.tsv", d / "bin-motifs.tsv",
                    seqs, genome_of, given, true_bin)

    def of(g, lo=9_000, hi=40_000):
        """Contigs of genome g, 9-40 kb (long enough for a clear decision), in name order."""
        return [c for c in sorted(seqs) if genome_of[c] == g and lo <= len(seqs[c]) <= hi]

    binned = [g for g, (_, b, _) in genomes.items() if b is not None and g != "E"]
    cov_of = {c: coverage for c in seqs}
    used = set()
    # contaminants: one contig of each binned genome moved into the next genome's bin (two for the first)
    for k, g in enumerate(binned):
        target = genomes[binned[(k + 1) % len(binned)]][1]
        for c in of(g)[: 2 if k == 0 else 1]:
            com.contaminants[c] = true_bin[c]; given[c] = target; used.add(c)
    orphan_genomes = [g for g, (_, b, _) in genomes.items() if b is None]
    for g in orphan_genomes:
        c = of(g)[0]
        com.contaminants[c] = None; given[c] = genomes[binned[0]][1]; used.add(c)
    # members left unbinned: two contigs of each binned genome
    for g in binned:
        for c in [c for c in of(g) if c not in used][:2]:
            com.missing[c] = given.pop(c); used.add(c)
    com.orphans = [c for c in sorted(seqs) if true_bin[c] is None and c not in given]
    sole = [c for c in seqs if genome_of[c] == "E"]
    com.sole = sole[0] if sole else ""
    if extras:
        g = binned[0]
        donor = [c for c in of(g, 20_000, 70_000) if c not in used][0]
        used.add(donor)
        com.short = "c900_" + g
        seqs[com.short] = seqs[donor][:700]; genome_of[com.short] = g; true_bin[com.short] = genomes[g][1]
        cov_of[com.short] = coverage
        com.low_coverage = [c for c in of(binned[1]) if c not in used][0]
        used.add(com.low_coverage)
        cov_of[com.low_coverage] = 2.0
        com.duplicate_of = [c for c in of(binned[2]) if c not in used][0]
        com.duplicate = "c901_" + binned[2]
        com.duplicate_binned = "c902_" + binned[2]
        for c in (com.duplicate, com.duplicate_binned):
            seqs[c] = seqs[com.duplicate_of]; genome_of[c] = binned[2]; true_bin[c] = genomes[binned[2]][1]
            cov_of[c] = coverage
        given[com.duplicate_binned] = genomes[binned[0]][1]

    with open(com.assembly, "w") as fh:
        for c, s in seqs.items():
            fh.write(f">{c} synthetic\n")
            fh.writelines(s[i:i + 80] + "\n" for i in range(0, len(s), 80))
    with open(com.pileup, "w") as fh:
        for c, s in seqs.items():
            fh.writelines(pileup_rows(c, s, genomes[genome_of[c]][2], rng, cov_of[c], motifs=motifs))
    write_contig_bin(com.contig_bin, given)
    write_motifs(com.motifs, genomes, motifs)
    return com


def write_contig_bin(path, given: dict[str, str]):
    Path(path).write_text("".join(f"{c}\t{b}\n" for c, b in given.items()))


def write_motifs(path, genomes=GENOMES, motifs=MOTIFS):
    """Nanomotif bin-motifs.tsv: one row per (bin, motif) a binned genome methylates, so a motif that only
    the binless genome has is not listed (Nanomotif reports motifs per bin)."""
    cols = ["bin", "mod_type", "motif", "mod_position", "n_mod_bin", "n_nomod_bin", "motif_type",
            "motif_complement", "mod_position_complement", "n_mod_complement", "n_nomod_complement"]
    lines = ["\t".join(cols) + "\n"]
    for g, (_, b, meth) in genomes.items():
        if b is None:
            continue
        for motif, pos, code, comp, comp_pos in motifs:
            if (motif, pos, code, comp, comp_pos) not in meth:
                continue
            kind = "palindrome" if revcomp(motif) == motif else "bipartite" if "NNN" in motif else "non-palindrome"
            lines.append("\t".join([b, code, motif, str(pos), "950", "50", kind, comp or "NA",
                                    str(comp_pos) if comp else "NA", "940" if comp else "NA",
                                    "60" if comp else "NA"]) + "\n")
    Path(path).write_text("".join(lines))
