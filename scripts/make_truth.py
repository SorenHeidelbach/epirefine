"""Build a truth table for a mock: assembly contigs -> reference species/replicon -> expected host bin.

Usage: python make_truth.py <asm_vs_ref.paf> <assembly.fasta> <contig_bin.tsv> <reference.fasta> <out.tsv>
           [--plasmids genomad_plasmid_summary.tsv] [--small-replicon 200000]

PAF from: minimap2 -x asm10 --secondary=no reference.fasta assembly.fasta
Reference records are named <Species>-<replicon>. A contig's source is the replicon covering the
most of its length (union of aligned query intervals). A species' expected bin is the bin holding
most of that species' chromosomal (large-replicon) contig length; "unknown" if none is binned.
"""
import argparse
from collections import defaultdict

import polars as pl

from pbinner.io import read_contig_bin, read_contig_list, read_fasta


def union_len(iv):
    iv = sorted(iv)
    tot, cs, ce = 0, None, None
    for s, e in iv:
        if cs is None or s > ce:
            if cs is not None:
                tot += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    return tot + (ce - cs if cs is not None else 0)


ap = argparse.ArgumentParser()
ap.add_argument("paf"); ap.add_argument("assembly"); ap.add_argument("contig_bin"); ap.add_argument("reference")
ap.add_argument("out"); ap.add_argument("--plasmids"); ap.add_argument("--small-replicon", type=int, default=200_000)
a = ap.parse_args()

asm = {c: len(s) for c, s in read_fasta(a.assembly).items()}
ref = {c: len(s) for c, s in read_fasta(a.reference).items()}
cb = read_contig_bin(a.contig_bin)
cand = read_contig_list(a.plasmids) if a.plasmids else set()
cand |= {c.rsplit("_", 1)[0] for c in cand if c not in asm and "_" in c}  # geNomad region names

hits = defaultdict(lambda: defaultdict(list))
ident = defaultdict(lambda: defaultdict(lambda: [0, 0]))
for line in open(a.paf):
    f = line.rstrip("\n").split("\t")
    q, qs, qe, t, nm, bl = f[0], int(f[2]), int(f[3]), f[5], int(f[9]), int(f[10])
    hits[q][t].append((qs, qe))
    ident[q][t][0] += nm; ident[q][t][1] += bl

rows = []
for c, L in asm.items():
    cov = {t: union_len(iv) for t, iv in hits[c].items()}
    if cov:
        t = max(cov, key=cov.get)
        sp_cov = defaultdict(int)
        for tt, v in cov.items():
            sp_cov[tt.rsplit("-", 1)[0]] += v
        m, b = ident[c][t]
        rows.append({"contig": c, "length": L, "species": t.rsplit("-", 1)[0], "replicon": t,
                     "replicon_length": ref[t], "query_coverage": cov[t] / L, "identity": m / b,
                     "species_coverage": max(sp_cov.values()) / L, "n_species_hit": len(sp_cov)})
    else:
        rows.append({"contig": c, "length": L, "species": "", "replicon": "", "replicon_length": 0,
                     "query_coverage": 0.0, "identity": 0.0, "species_coverage": 0.0, "n_species_hit": 0})
df = pl.DataFrame(rows).with_columns(
    pl.col("contig").replace_strict(cb, default="", return_dtype=pl.Utf8).alias("binned_as"),
    pl.col("contig").is_in(list(cand)).alias("plasmid_candidate"),
    (pl.col("replicon_length") < a.small_replicon).alias("small_replicon"))

# species -> expected bin: bin with most chromosomal (large-replicon) length of that species
chrom = df.filter((pl.col("binned_as") != "") & ~pl.col("small_replicon") & (pl.col("species") != ""))
best = (chrom.group_by("species", "binned_as").agg(pl.col("length").sum().alias("bp"))
        .sort("bp", descending=True).unique("species", keep="first"))
sp_bin = dict(zip(best["species"], best["binned_as"]))
df = df.with_columns(
    pl.col("species").replace_strict(sp_bin, default="unknown", return_dtype=pl.Utf8).alias("expected"))
df.sort("species", "replicon", "contig").write_csv(a.out, separator="\t", float_precision=4)

print("species -> bin:", dict(sorted(sp_bin.items())))
with pl.Config(tbl_rows=100, tbl_cols=20, fmt_str_lengths=40, tbl_width_chars=250):
    print(df.filter(pl.col("small_replicon") | pl.col("plasmid_candidate") | (pl.col("binned_as") == ""))
          .select("contig", "length", "species", "replicon", "replicon_length", "query_coverage", "identity",
                  "binned_as", "plasmid_candidate", "expected").sort("species", "replicon"))
