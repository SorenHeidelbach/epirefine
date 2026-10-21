"""Simulated metagenome with look-alike methylomes, for contamination / inclusion benchmarks.

Genomes come in clades. A clade has a methylome drawn from a real motif pool (e.g. Nanomotif's
human-gut bin-motifs.tsv); each strain in it drops or gains a motif with some probability, and a
fraction of strains copy a sibling exactly (indistinguishable by methylation). Some genome-motif pairs
are phase-variable: their methylated fraction varies from contig to contig. Genomes are cut into
contigs with log-uniform lengths; reads per site are Poisson around a per-genome coverage, and the
modification caller is beta-binomial per mod type.

    python scripts/simulate_community.py --motif-pool human_gut/nanomotif_1.1.3/bin-motifs.tsv --out sim1

Writes assembly.fasta, pileup.bed (modkit bedMethyl, rows at motif sites plus background),
truth.tsv (contig, length, expected = the genome's bin, genome, clade) and genome_motifs.tsv
(bin, motif, mod_position, mod_type, kind) from which perturb_bins.py writes bin-motifs.tsv for the
bins that exist in a scenario.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from pbinner import io as pio
from pbinner.motifs import MotifMatcher

CALLER = {"a": (0.03, 0.85, 12.0), "m": (0.03, 0.80, 12.0), "21839": (0.06, 0.70, 10.0)}
LETTERS = np.frombuffer(b"ACGT", np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motif-pool", required=True, help="bin-motifs.tsv to draw motifs from")
    ap.add_argument("--out", required=True)
    ap.add_argument("--clades", type=int, default=14)
    ap.add_argument("--genomes", type=int, default=50)
    ap.add_argument("--pool-size", type=int, default=120)
    ap.add_argument("--genome-mb", type=float, nargs=2, default=(1.0, 2.0))
    ap.add_argument("--contig-len", type=int, nargs=2, default=(3_000, 150_000))
    ap.add_argument("--p-drop", type=float, default=0.12, help="per motif, strain loses a clade motif")
    ap.add_argument("--p-gain", type=float, default=0.35, help="strain gains one extra motif")
    ap.add_argument("--p-clone", type=float, default=0.15, help="strain copies a sibling's methylome")
    ap.add_argument("--p-phase", type=float, default=0.08, help="genome-motif pair is phase-variable")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    pool_all = pio.read_motifs(a.motif_pool)
    pool_all = [m for m in pool_all if m.mod_type in CALLER]
    pool = [pool_all[i] for i in rng.choice(len(pool_all), min(a.pool_size, len(pool_all)), replace=False)]

    # clade methylomes, then strains
    clade_motifs = [set(rng.choice(len(pool), int(rng.integers(3, 11)), replace=False).tolist())
                    for _ in range(a.clades)]
    sizes = np.bincount(rng.integers(0, a.clades, a.genomes), minlength=a.clades)
    genomes = []
    for c, n in enumerate(sizes):
        members = []
        for s in range(n):
            if members and rng.random() < a.p_clone:
                ms = set(members[int(rng.integers(len(members)))])
            else:
                ms = {m for m in clade_motifs[c] if rng.random() > a.p_drop}
                if rng.random() < a.p_gain:
                    ms.add(int(rng.integers(len(pool))))
            members.append(ms)
            phase = {m for m in ms if rng.random() < a.p_phase}
            genomes.append({"name": f"G{len(genomes) + 1:02d}", "clade": c, "motifs": ms, "phase": phase,
                            "gc": rng.uniform(0.35, 0.65), "cov": float(np.clip(rng.lognormal(np.log(20), 0.6), 4, 120)),
                            "size": int(rng.uniform(*a.genome_mb) * 1e6)})

    matcher = MotifMatcher(pool)
    modc = np.array([m.mod_type for m in pool])
    fasta, truth, rows = [], [], []
    for g in genomes:
        p = [(1 - g["gc"]) / 2, g["gc"] / 2, g["gc"] / 2, (1 - g["gc"]) / 2]
        codes = rng.choice(4, size=g["size"], p=p).astype(np.uint8)
        lo, hi = a.contig_len
        cuts = [0]
        while g["size"] - cuts[-1] > hi:
            cuts.append(cuts[-1] + int(np.exp(rng.uniform(np.log(lo), np.log(hi)))))
        cuts.append(g["size"])
        phase_level = {m: rng.uniform(0.2, 0.8) for m in g["phase"]}
        for k, (b, e) in enumerate(zip(cuts[:-1], cuts[1:])):
            name = f"{g['name']}_c{k}"
            seg = codes[b:e]
            fasta.append((name, LETTERS[seg].tobytes().decode()))
            truth.append({"contig": name, "length": e - b, "expected": f"bin_{g['name']}", "genome": g["name"],
                          "clade": g["clade"]})
            rec = matcher.find(seg)
            if not len(rec["pos"]):
                continue
            cov = g["cov"] * rng.lognormal(0, 0.15)
            # methylation state per occurrence: active motifs ~ 0.97, phase-variable ones vary per contig
            phi = np.zeros(len(pool))
            for m in g["motifs"]:
                phi[m] = 0.97 if m not in g["phase"] else rng.beta(phase_level[m] * 8, (1 - phase_level[m]) * 8)
            # one state per occurrence, shared by the records (strands) of that occurrence
            occ_key = rec["occ_motif"].astype(np.int64) * (1 << 40) + rec["occ"]
            _, first, inv = np.unique(occ_key, return_index=True, return_inverse=True)
            meth = (rng.random(len(first)) < phi[rec["motif"][first]])[inv]
            # one row per (pos, strand, mod type); a position matched by several motifs is methylated if any is
            df = pl.DataFrame({"pos": rec["pos"], "strand": rec["strand"], "mod": modc[rec["motif"]], "meth": meth})
            df = df.group_by("pos", "strand", "mod").agg(pl.col("meth").any())
            n = rng.poisson(cov, df.height)
            mt = df["mod"].to_numpy()
            mu = np.where(df["meth"].to_numpy(), [CALLER[x][1] for x in mt], [CALLER[x][0] for x in mt])
            kap = np.array([CALLER[x][2] for x in mt])
            kk = rng.binomial(n, rng.beta(mu * kap, (1 - mu) * kap))
            keep = n > 0
            rows.append(pl.DataFrame({"contig": name, "pos": df["pos"].to_numpy()[keep],
                                      "mod": mt[keep], "n": n[keep], "k": kk[keep],
                                      "strand": np.where(df["strand"].to_numpy()[keep] == 0, "+", "-")}))
    with open(out / "assembly.fasta", "w") as fh:
        for name, s in fasta:
            fh.write(f">{name}\n")
            for i in range(0, len(s), 80):
                fh.write(s[i:i + 80] + "\n")
    pile = pl.concat(rows).sort("contig", "pos")
    pile.select(
        "contig", "pos", (pl.col("pos") + 1).alias("end"), "mod", pl.col("n").alias("score"), "strand",
        pl.col("pos").alias("ts"), (pl.col("pos") + 1).alias("te"), pl.lit("255,0,0").alias("color"),
        pl.col("n").alias("nv"), (100 * pl.col("k") / pl.col("n")).round(2).alias("frac"), "k",
        (pl.col("n") - pl.col("k")).alias("nc"), *[pl.lit(0).alias(f"z{i}") for i in range(5)],
    ).write_csv(out / "pileup.bed", separator="\t", include_header=False)
    pl.DataFrame(truth).write_csv(out / "truth.tsv", separator="\t")
    gm = [{"bin": f"bin_{g['name']}", "motif": pool[m].seq, "mod_position": pool[m].mod_pos,
           "mod_type": pool[m].mod_type, "kind": "phase" if m in g["phase"] else "active"}
          for g in genomes for m in sorted(g["motifs"])]
    pl.DataFrame(gm).write_csv(out / "genome_motifs.tsv", separator="\t")
    ident = sum(1 for i, g in enumerate(genomes) for h in genomes[:i] if g["motifs"] == h["motifs"])
    print(f"{len(genomes)} genomes in {a.clades} clades, {len(fasta)} contigs, {pile.height} pileup rows; "
          f"{ident} pairs of genomes with identical motif sets -> {out}")


if __name__ == "__main__":
    main()
