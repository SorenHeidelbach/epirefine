"""Synthetic community with known plasmid hosts, written in the same formats as real inputs.

Built-in hard cases:
  * H2 methylates RGATCY, which is nested in GATC (H1, H8): tests motif grouping, phi ~ 0.25.
  * H3 and H4 have identical motif sets: not separable, the credible set should hold both.
  * H7 has no methylation at all: its plasmids look like any motif-free host.
  * H9 is never binned: its plasmids should come out as "unknown".
  * Four plasmids carry their own MTase: annotated with motif (H1: GAAGAC, H8: TCGA), annotated
    with mod type only (H2: CCWGG), and unannotated (H6: GANTC).
  * Plasmids are under-methylated relative to their host (mean 0.85 vs 0.96) and vary per plasmid.
  * One H1 plasmid is binned into bin_H1 by the binner; one H5 contig is misbinned into bin_H6 and
    a 150 kb H1 contig is misbinned into bin_H2 (contamination).
  * Type I bipartite motif reported with its complement (canonicalisation test).
  * mtases.tsv annotates host and plasmid MTases, with missing motifs and one false (inactive) MTase.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from .motifs import Motif, find_records

POOL = {
    "GATC": Motif("GATC", 1, "a"), "RGATCY": Motif("RGATCY", 2, "a"), "GANTC": Motif("GANTC", 1, "a"),
    "CCWGG": Motif("CCWGG", 1, "m"), "GCGC": Motif("GCGC", 1, "m"), "CTCGAG": Motif("CTCGAG", 2, "21839"),
    "GAGNNNNNGTCA": Motif("GAGNNNNNGTCA", 1, "a"), "TGACNNNNNCTC": Motif("TGACNNNNNCTC", 2, "a"),
    "TCGA": Motif("TCGA", 3, "a"), "CAGCTG": Motif("CAGCTG", 3, "m"), "GAAGAC": Motif("GAAGAC", 4, "a"),
}
TYPE_I = ["GAGNNNNNGTCA", "TGACNNNNNCTC"]
HOSTS = {
    "H1": ["GATC", "CCWGG"], "H2": ["RGATCY"], "H3": ["GANTC"] + TYPE_I, "H4": ["GANTC"] + TYPE_I,
    "H5": ["TCGA", "CTCGAG"], "H6": ["GCGC", "GAAGAC"], "H7": [], "H8": ["GATC", "CAGCTG"],
    "H9": ["CCWGG", "TCGA"],
}
UNBINNED_HOSTS = {"H9"}
OWN_MTASE = [("H1_plasmid_mtase", "H1", "GAAGAC", 30_000), ("H8_plasmid_mtase", "H8", "TCGA", 25_000),
             ("H2_plasmid_mtase", "H2", "CCWGG", 20_000), ("H6_plasmid_mtase", "H6", "GANTC", 40_000)]
MTASE_ANNOTATION = {"H1_plasmid_mtase": ("a", "GAAGAC"), "H8_plasmid_mtase": ("a", "TCGA"),
                    "H2_plasmid_mtase": ("m", "")}
# caller behaviour per mod type: (mu0, mu1, kappa)
CALLER = {"a": (0.04, 0.88, 12.0), "m": (0.03, 0.85, 12.0), "21839": (0.08, 0.70, 12.0)}


def _random_seq(rng, n, gc):
    p = [(1 - gc) / 2, gc / 2, gc / 2, (1 - gc) / 2]
    return "".join(rng.choice(list("ACGT"), size=n, p=p))


def simulate(out, seed: int = 1, chrom_len: int = 600_000):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    contigs, truth = {}, []
    plasmid_lengths = [1_500, 4_000, 8_000, 15_000, 30_000, 60_000]
    for h, ms in HOSTS.items():
        gc = rng.uniform(0.35, 0.65)
        cov = float(np.clip(rng.lognormal(np.log(25), 0.5), 6, 80))
        n_pieces = int(rng.integers(2 if h == "H5" else 1, 5))
        cuts = np.sort(rng.choice(np.arange(50_000, chrom_len - 50_000, 10_000), n_pieces - 1, replace=False))
        bounds = np.r_[0, cuts, chrom_len]
        for i in range(n_pieces):
            name = f"{h}_chrom_{i + 1}"
            contigs[name] = _random_seq(rng, int(bounds[i + 1] - bounds[i]), gc)
            truth.append({"contig": name, "host": h, "type": "chromosome", "cov": cov, "extra": ""})
        for j, plen in enumerate(rng.choice(plasmid_lengths, 3, replace=False)):
            name = f"{h}_plasmid_{j + 1}"
            contigs[name] = _random_seq(rng, int(plen), gc + rng.normal(0, 0.03))
            truth.append({"contig": name, "host": h, "type": "plasmid",
                          "cov": cov * float(rng.uniform(1, 6)), "extra": ""})
    for name, h, extra, plen in OWN_MTASE:
        contigs[name] = _random_seq(rng, plen, 0.5)
        truth.append({"contig": name, "host": h, "type": "plasmid", "cov": 60.0, "extra": extra})
    contigs["H1_chrom_contam"] = _random_seq(rng, 150_000, 0.5)
    truth.append({"contig": "H1_chrom_contam", "host": "H1", "type": "chromosome", "cov": 30.0, "extra": ""})

    tdf = pl.DataFrame(truth)
    host_of = dict(zip(tdf["contig"], tdf["host"]))

    # binning, including two binner errors
    contig_bin = {}
    for r in tdf.iter_rows(named=True):
        if r["type"] == "chromosome" and r["host"] not in UNBINNED_HOSTS:
            contig_bin[r["contig"]] = f"bin_{r['host']}"
    contig_bin["H1_plasmid_1"] = "bin_H1"
    contig_bin["H1_chrom_contam"] = "bin_H2"
    h5_chroms = [c for c in contigs if c.startswith("H5_chrom")]
    if len(h5_chroms) > 1:
        contig_bin[min(h5_chroms, key=lambda c: len(contigs[c]))] = "bin_H6"

    # pileup
    lines = []
    for r in tdf.iter_rows(named=True):
        c, seq, cov = r["contig"], contigs[r["contig"]], r["cov"]
        is_plasmid = r["type"] == "plasmid"
        active = HOSTS[r["host"]] + ([r["extra"]] if r["extra"] else [])
        methylated = set()
        for mname in active:
            m = POOL[mname]
            mean, conc = (0.85, 60) if is_plasmid else (0.96, 200)
            phi = rng.beta(mean * conc, (1 - mean) * conc)
            pos, strand, occ = find_records(seq, m)
            occ_state = {o: rng.random() < phi for o in np.unique(occ)}
            methylated.update((int(p), int(s), m.mod_type) for p, s, o in zip(pos, strand, occ) if occ_state[o])
        recs = set()
        for m in POOL.values():
            pos, strand, _ = find_records(seq, m)
            recs.update((int(p), int(s), m.mod_type) for p, s in zip(pos, strand))
        for _ in range(len(seq) // 200):  # background non-motif rows, should be ignored
            p = int(rng.integers(len(seq)))
            recs.add((p, 0, "a" if seq[p] == "A" else "m"))
        for p, s, mt in sorted(recs):
            n = int(rng.poisson(cov))
            if n == 0:
                continue
            mu0, mu1, kappa = CALLER[mt]
            mu = mu1 if (p, s, mt) in methylated else mu0
            k = int(rng.binomial(n, rng.beta(mu * kappa, (1 - mu) * kappa)))
            st = "+" if s == 0 else "-"
            lines.append(f"{c}\t{p}\t{p + 1}\t{mt}\t{n}\t{st}\t{p}\t{p + 1}\t255,0,0\t"
                         f"{n} {100 * k / n:.2f} {k} {n - k} 0 0 0 0 0\n")

    with open(out / "assembly.fasta", "w") as fh:
        for c, s in contigs.items():
            fh.write(f">{c}\n")
            for i in range(0, len(s), 80):
                fh.write(s[i:i + 80] + "\n")
    with open(out / "pileup.bed", "w") as fh:
        fh.writelines(lines)
    with open(out / "contig_bin.tsv", "w") as fh:
        fh.writelines(f"{c}\t{b}\n" for c, b in contig_bin.items())
    bm = []
    for h, ms in HOSTS.items():
        if h in UNBINNED_HOSTS:
            continue
        for mname in ms:
            if mname == "TGACNNNNNCTC":
                continue  # reported as the complement of GAGNNNNNGTCA
            m = POOL[mname]
            comp = ("TGACNNNNNCTC", "2") if mname == "GAGNNNNNGTCA" else ("", "")
            bm.append({"bin": f"bin_{h}", "motif": m.seq, "mod_position": str(m.mod_pos), "mod_type": m.mod_type,
                       "motif_complement": comp[0], "mod_position_complement": comp[1]})
    pl.DataFrame(bm).write_csv(out / "bin-motifs.tsv", separator="\t")
    mt = []
    for h, ms in HOSTS.items():
        chrom = f"{h}_chrom_1"
        for mname in ms:
            if mname == "TGACNNNNNCTC":
                continue
            m = POOL[mname]
            mt.append({"contig": chrom, "mod_type": m.mod_type, "motif": "" if rng.random() < 0.2 else m.seq})
    mt.append({"contig": "H7_chrom_1", "mod_type": "a", "motif": "GATC"})  # inactive MTase
    for name, _, extra, _ in OWN_MTASE:
        if name in MTASE_ANNOTATION:
            mod, motif = MTASE_ANNOTATION[name]
            mt.append({"contig": name, "mod_type": mod, "motif": motif})
    pl.DataFrame(mt).write_csv(out / "mtases.tsv", separator="\t")
    plasmids = [c for c in contigs if "plasmid" in c]
    (out / "plasmids.txt").write_text("\n".join(plasmids) + "\n")
    tdf.with_columns(
        pl.col("contig").replace_strict(contig_bin, default="", return_dtype=pl.Utf8).alias("binned_as"),
        pl.when(pl.col("host").is_in(list(UNBINNED_HOSTS))).then(pl.lit("unknown"))
        .otherwise(pl.lit("bin_") + pl.col("host")).alias("expected"),
    ).write_csv(out / "truth.tsv", separator="\t")
    print(f"simulated {len(contigs)} contigs, {len(lines)} pileup rows -> {out}")
