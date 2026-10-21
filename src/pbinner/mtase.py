"""MTase annotations -> rho_um = prior P(unit u methylates motif group m with its own MTase).

Input is a TSV with one row per MTase gene. Column names are matched flexibly so MTase-linker,
DefenseFinder-derived or hand-made tables work:
    contig      : contig / contig_id / seq_name   (or gene_id like <contig>_<n>, prodigal style)
    mod_type    : mod_type / modification         (a, m, 21839 or 6mA, 5mC, 4mC; optional)
    motif       : motif / detected_motif / motif_pred / predicted_motif / motif_guess (optional, IUPAC)

Annotation level per (unit, group):
    2 = an MTase on the unit has a predicted motif that is a member of the group (or its revcomp)
    1 = an MTase of the group's mod type with no usable motif
    0 = nothing

rho for levels 2 and 1 is estimated on the host bins when they are annotated too: it is the
fraction of annotated (bin, group) pairs whose host profile shows the group methylated. The base
rate for level 0 is a parameter (default 0.02); the host-level rate of methylated-but-unannotated
groups is reported alongside as a diagnostic of annotation recall.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import polars as pl

from .motifs import Motif, revcomp

_MOD = {"a": "a", "6ma": "a", "m6a": "a", "m": "m", "5mc": "m", "m5c": "m", "21839": "21839", "4mc": "21839",
        "m4c": "21839"}


def _pick(cols, options):
    low = {c.lower(): c for c in cols}
    return next((low[o] for o in options if o in low), None)


def read_mtases(path) -> pl.DataFrame:
    df = pl.read_csv(path, separator="\t", infer_schema_length=0)
    c_contig = _pick(df.columns, ["contig", "contig_id", "seq_name", "contig_name"])
    c_gene = _pick(df.columns, ["gene_id", "gene", "protein_id"])
    c_mod = _pick(df.columns, ["mod_type", "modification", "modtype"])
    c_motif = _pick(df.columns, ["motif", "detected_motif", "motif_pred", "predicted_motif", "motif_guess"])
    if c_contig is None and c_gene is None:
        raise ValueError(f"{path}: needs a contig or gene_id column, found {df.columns}")
    contig = df[c_contig] if c_contig else df[c_gene].str.replace(r"_\d+$", "")
    mod = (df[c_mod].str.to_lowercase().replace_strict(_MOD, default=None) if c_mod
           else pl.Series([None] * df.height, dtype=pl.Utf8))
    motif = df[c_motif].str.to_uppercase() if c_motif else pl.Series([None] * df.height, dtype=pl.Utf8)
    out = pl.DataFrame({"contig": contig, "mod_type": mod, "motif": motif})
    return out.with_columns(
        pl.when(pl.col("motif").str.contains(r"^[ACGTRYSWKMBDHVN]+$")).then(pl.col("motif")).otherwise(None))


def annotation_levels(ann: pl.DataFrame, units: list[str], motifs: list[Motif], group_of: list[int],
                      group_modtype: list[str]) -> np.ndarray:
    """levels[u, m] in {0, 1, 2} for the given unit names (contigs)."""
    M = len(group_modtype)
    seq_to_groups: dict[str, set[int]] = {}
    for mot, g in zip(motifs, group_of):
        for s in (mot.seq, revcomp(mot.seq)):
            seq_to_groups.setdefault(s, set()).add(g)
    idx = {u: i for i, u in enumerate(units)}
    levels = np.zeros((len(units), M), dtype=np.int8)
    for contig, mod, motif in ann.iter_rows():
        if contig not in idx:
            continue
        i = idx[contig]
        groups = seq_to_groups.get(motif, set()) if motif else set()
        if mod is not None:
            groups = {g for g in groups if group_modtype[g] == mod} or groups
        if groups:
            levels[i, list(groups)] = 2
        else:
            for g in range(M):
                if (mod is None or group_modtype[g] == mod) and levels[i, g] == 0:
                    levels[i, g] = 1
    return levels


@dataclass
class RhoParams:
    motif: float = 0.8
    modtype: float = 0.3
    base: float = 0.02

    def matrix(self, levels: np.ndarray) -> np.ndarray:
        return np.choose(levels, [self.base, self.modtype, self.motif]).astype(float)


def estimate_rho(levels_bins: np.ndarray, p_present: np.ndarray, base: float = 0.02, min_n: int = 5):
    """levels_bins[b, m] (max over the bin's contigs), p_present[b, m] from host profiles."""
    rho, report = RhoParams(base=base), {}
    for lvl, name in ((2, "motif"), (1, "modtype")):
        sel = levels_bins == lvl
        n = int(sel.sum())
        est = (p_present[sel].sum() + 1) / (n + 2)
        report[f"rho_{name}_n"] = n
        report[f"rho_{name}_estimate"] = float(est)
        if n >= min_n:
            setattr(rho, name, float(np.clip(est, 0.05, 0.99)))
    sel = levels_bins == 0
    report["host_unannotated_methylated_rate"] = float(p_present[sel].mean()) if sel.any() else float("nan")
    report.update(rho_motif=rho.motif, rho_modtype=rho.modtype, rho_base=rho.base)
    return rho, report


def bin_levels(ann: pl.DataFrame, contig_bin: dict[str, str], bins: list[str], motifs, group_of, group_modtype):
    contigs = sorted(contig_bin)
    lv = annotation_levels(ann, contigs, motifs, group_of, group_modtype)
    out = np.zeros((len(bins), len(group_modtype)), dtype=np.int8)
    bidx = {b: i for i, b in enumerate(bins)}
    for c, row in zip(contigs, lv):
        out[bidx[contig_bin[c]]] = np.maximum(out[bidx[contig_bin[c]]], row)
    return out
