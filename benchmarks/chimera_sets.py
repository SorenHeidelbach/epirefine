"""Scenarios with long chimeric contigs, long members and long contaminants, built from a monoculture community.

The other benchmarks plant whole-contig errors only, so they can't test whether pbinner tells a contig
that is partly foreign along its length (an island, a chimeric join) from one that is foreign as a
whole. On ZymoFecal every flag that Hi-C contradicted on a long contig was of the first kind. This takes
a community from benchmarks/monoculture_sets.py and joins its pieces into long contigs of known make-up,
all placed in the bin of their host isolate A:

  chimera           >= a_min bp of A's member pieces, then >= b_min bp of another isolate B's pieces
                    (truth: kept in A's bin, foreign segment = the B part)
  long_member       >= member_min bp of A's member pieces (truth: kept)
  long_contaminant  >= b_min * 1.5 bp of B's pieces (truth: flagged; true bin B, or "" if B has no bin)

and, left unbinned (for the regional model of inclusion):

  long_missing      >= member_min bp of A's member pieces (truth: included into A's bin)
  unbinned_chimera  >= a_min bp of A's pieces, then >= b_min bp of B's (truth: included into A's bin,
                    foreign segment = the B part)
  unbinned_foreign  >= b_min * 1.5 bp of B's pieces (truth: B's bin, or stays unbinned if B has none)

Only isolates at >= min_cov target coverage are used, so the contigs get decisions. The joined pieces are
removed as separate contigs; the pileup is rewritten in the new coordinates (streamed, only the joined
pieces' rows change).

    python benchmarks/chimera_sets.py --community mono_bench/train/mono_train_1 --out mono_bench/chimera/chim_1 --seed 1

Writes assembly.fasta, pileup.bed, contig_bin.tsv, bin-motifs.tsv, truth_actions.tsv (perturb_bins.py's
columns plus foreign_start, foreign_end) and joined.tsv (which pieces went into which new contig).
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import numpy as np
import polars as pl


def read_fasta(path):
    seqs, name, buf = {}, None, []
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if name is not None:
                    seqs[name] = "".join(buf)
                name, buf = line[1:].split()[0], []
            else:
                buf.append(line.strip())
    if name is not None:
        seqs[name] = "".join(buf)
    return seqs


def take(pool: list[tuple[str, int]], need: int, rng) -> list[tuple[str, int]]:
    """Pop pieces from pool (shuffled) until their total length reaches need; [] if it can't."""
    out, tot = [], 0
    while pool and tot < need:
        out.append(pool.pop()); tot += out[-1][1]
    return out if tot >= need else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--community", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chimeras", type=int, default=8)
    ap.add_argument("--long-members", type=int, default=6)
    ap.add_argument("--long-contaminants", type=int, default=4)
    ap.add_argument("--long-missing", type=int, default=4)
    ap.add_argument("--unbinned-chimeras", type=int, default=4)
    ap.add_argument("--unbinned-foreign", type=int, default=3)
    ap.add_argument("--a-min", type=int, default=150_000)
    ap.add_argument("--b-min", type=int, default=100_000)
    ap.add_argument("--member-min", type=int, default=200_000)
    ap.add_argument("--min-cov", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    src, out = Path(a.community), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    pieces = pl.read_csv(src / "pieces.tsv", separator="\t")
    truth = pl.read_csv(src / "truth_actions.tsv", separator="\t", infer_schema_length=0).with_columns(
        pl.col("true_bin", "given_bin").fill_null(""), pl.col("length").cast(pl.Int64))
    x = truth.join(pieces.select("contig", "isolate", "plasmid", "target_coverage"), on="contig")
    rich = x.filter((pl.col("target_coverage") >= a.min_cov) & ~pl.col("plasmid"))
    binned = sorted(set(rich.filter(pl.col("true_bin") != "")["isolate"].to_list()))
    isolates = sorted(set(rich["isolate"].to_list()))
    if len(binned) < 1 or len(isolates) < 2:
        raise SystemExit(f"{src}: needs >= 1 binned and >= 2 isolates at >= {a.min_cov}x")

    def pool(iso, members_only):
        sel = rich.filter((pl.col("isolate") == iso) & (pl.col("category") == "member" if members_only else pl.lit(True)))
        p = list(zip(sel["contig"].to_list(), sel["length"].to_list()))
        rng.shuffle(p)
        return p
    pools_a = {i: pool(i, True) for i in binned}
    pools_b = {i: pool(i, False) for i in isolates}
    used = set()

    def fresh(p):  # drop pieces already used elsewhere
        p[:] = [q for q in p if q[0] not in used]

    made = []  # (name, kind, host isolate, donor isolate, a pieces, b pieces)
    kinds = (("chimera", a.chimeras), ("long_member", a.long_members), ("long_contaminant", a.long_contaminants),
             ("long_missing", a.long_missing), ("unbinned_chimera", a.unbinned_chimeras), ("unbinned_foreign", a.unbinned_foreign))
    a_part = {"chimera": a.a_min, "unbinned_chimera": a.a_min, "long_member": a.member_min, "long_missing": a.member_min}
    b_part = {"chimera": a.b_min, "unbinned_chimera": a.b_min, "long_contaminant": int(a.b_min * 1.5), "unbinned_foreign": int(a.b_min * 1.5)}
    for kind, n in kinds:
        for k in range(n):
            A = binned[rng.integers(len(binned))]
            B = [i for i in isolates if i != A][rng.integers(len(isolates) - 1)] if kind in b_part else None
            for p in list(pools_a.values()) + list(pools_b.values()):
                fresh(p)
            pa = take(pools_a[A], a_part[kind], rng) if kind in a_part else []
            used.update(c for c, _ in pa)
            for p in pools_b.values():
                fresh(p)
            pb = take(pools_b[B], b_part[kind], rng) if B else []
            used.update(c for c, _ in pb)
            if (kind in a_part and not pa) or (kind in b_part and not pb):
                continue
            made.append((f"{kind}_{k}", kind, A, B, pa, pb))

    seqs = read_fasta(src / "assembly.fasta")
    true_bin_of = dict(zip(rich["isolate"].to_list(), rich["true_bin"].to_list()))
    for iso in isolates:  # bin of each isolate ("" if it has none: an orphan)
        tb = rich.filter((pl.col("isolate") == iso) & (pl.col("true_bin") != ""))["true_bin"]
        true_bin_of[iso] = tb[0] if tb.len() else ""
    remap, rows, joined = [], [], []
    with open(out / "assembly.fasta", "w") as fa:
        for name, s in seqs.items():
            if name not in used:
                fa.write(f">{name}\n{s}\n")
        for name, kind, A, B, pa, pb in made:
            parts, off, fstart, fend = [], 0, -1, -1
            for i, (c, L) in enumerate(pa + pb):
                if i == len(pa) and pb:
                    fstart = off
                remap.append((c, name, off)); parts.append(seqs[c]); joined.append((name, c, off, L)); off += L
            if pb:
                fend = off if pa else off
                fstart = fstart if pa else 0
            fa.write(f">{name}\n{''.join(parts)}\n")
            host_bin = true_bin_of[A]
            tb = true_bin_of[B] if kind in ("long_contaminant", "unbinned_foreign") else host_bin
            given = host_bin if kind in ("chimera", "long_member", "long_contaminant") else ""
            cat = {"chimera": "chimera", "long_member": "member", "long_contaminant": "contaminant" if tb else "orphan_contaminant",
                   "long_missing": "missing", "unbinned_chimera": "missing", "unbinned_foreign": "missing" if tb else "orphan"}[kind]
            rows.append({"contig": name, "length": off, "true_bin": tb, "given_bin": given, "category": cat,
                         "kind": kind, "foreign_start": fstart, "foreign_end": fend})
    pl.DataFrame(joined, schema=["contig", "piece", "offset", "length"], orient="row").write_csv(out / "joined.tsv", separator="\t")
    t2 = (truth.filter(~pl.col("contig").is_in(list(used)))
          .with_columns(pl.lit("").alias("kind"), pl.lit(-1).alias("foreign_start"), pl.lit(-1).alias("foreign_end")))
    t2 = pl.concat([t2, pl.DataFrame(rows, schema=t2.schema)])
    t2.write_csv(out / "truth_actions.tsv", separator="\t")
    t2.filter(pl.col("given_bin") != "").select("contig", "given_bin").write_csv(out / "contig_bin.tsv", separator="\t", include_header=False)
    shutil.copy(src / "bin-motifs.tsv", out / "bin-motifs.tsv")
    m = out / "remap.tsv"
    pl.DataFrame(remap, schema=["piece", "contig", "offset"], orient="row").write_csv(m, separator="\t", include_header=False)
    # bedMethyl: chrom, start, end, ..., thickStart (7), thickEnd (8); shift positions of joined pieces
    # pbinner needs each contig's rows contiguous: rows of joined pieces go to a side file, which is
    # sorted by new contig and position and appended
    side = out / "joined_rows.bed"
    awk = ('BEGIN{FS=OFS="\\t"} NR==FNR{n[$1]=$2; o[$1]=$3; next} '
           f'($1 in n){{d=o[$1]; $1=n[$1]; $2+=d; $3+=d; $7+=d; $8+=d; print > "{side}"; next}} {{print}}')
    with open(out / "pileup.bed", "w") as fo:
        subprocess.run(["awk", awk, str(m), str(src / "pileup.bed")], stdout=fo, check=True)
    with open(out / "pileup.bed", "a") as fo:
        subprocess.run(["sort", "-t", "\t", "-k1,1", "-k2,2n", "-S", "2G", str(side)], stdout=fo, check=True)
    side.unlink()
    print(f"{out}: {sum(1 for r in rows if r['kind'] == 'chimera')} chimeras, "
          f"{sum(1 for r in rows if r['kind'] == 'long_member')} long members, "
          f"{sum(1 for r in rows if r['kind'] == 'long_contaminant')} long contaminants; {len(used)} pieces joined")


if __name__ == "__main__":
    main()
