"""Hi-C refinement without the changes that only add sequence the bin already holds.

Hi-C links strain variants and duplicated stretches of a genome to their bin, correctly, but their
extra copies of marker genes read as contamination to CheckM2. For every contig Hi-C adds to or moves
into a bin, the FracMinHash containment of the contig in the target bin's original contigs is
measured (epirefine's redundancy sketch, k = 21, scale 50); changes with containment >= --max-redundancy
are undone (an added contig stays unbinned, a moved one stays in its bin).

    python hic/nonredundant.py --assembly asm.fasta --contig-bin contig_bin.tsv --hic-calls hic/hic_calls.tsv \
        --out-redundancy hic_quality/hic_change_redundancy.tsv --out hic_nonredundant/new_contig_bin.tsv
Needs epirefine importable (the repository's .venv) for the sketches.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl

from epirefine.io import read_fasta
from epirefine.motifs import encode
from epirefine.redundancy import containment, sketch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assembly", required=True)
    ap.add_argument("--contig-bin", required=True)
    ap.add_argument("--hic-calls", required=True)
    ap.add_argument("--max-redundancy", type=float, default=0.3)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--out-redundancy", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cb = pl.read_csv(a.contig_bin, separator="\t", has_header=False, new_columns=["contig", "bin"], infer_schema_length=0)
    calls = (pl.read_csv(a.hic_calls, separator="\t", infer_schema_length=0).filter(pl.col("confident") == "true")
             .select("contig", "hic_bin"))
    given = dict(cb.iter_rows())
    changes = [(c, h, "added" if c not in given else "moved") for c, h in calls.iter_rows() if given.get(c) != h]
    asm = read_fasta(a.assembly)
    members: dict[str, list[str]] = {}
    for c, b in cb.iter_rows():
        members.setdefault(b, []).append(c)

    def bin_sketch(b):
        return b, np.unique(np.concatenate([sketch(encode(asm[c])) for c in members[b]]))

    def one(t):
        c, h, kind = t
        s = sketch(encode(asm[c]))
        return c, h, kind, len(asm[c]), containment(s, sketches[h]) if len(s) else float("nan")

    with ThreadPoolExecutor(a.threads) as ex:
        sketches = dict(ex.map(bin_sketch, sorted({h for _, h, _ in changes})))
        rows = list(ex.map(one, changes))
    red = pl.DataFrame(rows, schema=["contig", "hic_bin", "kind", "length", "containment"], orient="row")
    Path(a.out_redundancy).parent.mkdir(parents=True, exist_ok=True)
    red.write_csv(a.out_redundancy, separator="\t", float_precision=4)
    drop = red.filter(pl.col("containment") >= a.max_redundancy)
    added = set(drop.filter(pl.col("kind") == "added")["contig"])
    moved = set(drop.filter(pl.col("kind") == "moved")["contig"])
    hic = dict(cb.iter_rows())
    hic.update({c: h for c, h, _ in changes if c not in added and c not in moved})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(sorted(hic.items(), key=lambda x: (x[1], x[0])), schema=["contig", "bin"], orient="row").write_csv(
        a.out, separator="\t", include_header=False)
    print(f"{len(changes)} Hi-C changes; undone as redundant (containment >= {a.max_redundancy}): "
          f"{len(added)} additions, {len(moved)} moves")
    print(red.group_by("kind").agg(pl.len(), pl.col("length").sum().alias("bp"),
                                   (pl.col("containment") >= a.max_redundancy).sum().alias("redundant")))


if __name__ == "__main__":
    main()
