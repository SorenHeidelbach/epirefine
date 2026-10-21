"""Time pbinner's actual functions at realistic sizes (~50k contigs, ~1000 motifs, hundreds of bins).

Each stage is timed on real or realistically shaped inputs. When a full-size run would exceed this
machine, a fraction is timed and the result scaled linearly in the dimension that is reduced (stated
in the output). Results go to <out>/components.tsv.

    python scripts/scale_components.py --assembly human_gut/m2_assembly.fasta --pileup human_gut/...bed \
        --motifs ZymoOral/nanomotif_1.1.3/bin-motifs.tsv --out bench_components
"""
from __future__ import annotations

import argparse
import gc
import os
import resource
import time
from pathlib import Path

import numpy as np
import polars as pl

from pbinner import io as pio
from pbinner.detection import DetectionParams
from pbinner.grid import Grid
from pbinner.model import FittedModel, fit_population_prior, marginalize_unit, unit_loglik, group_scores
from pbinner.motifs import Motif, build_records, revcomp

ROWS = []


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def record(stage, measured_s, scale=1.0, note="", **dims):
    est = measured_s * scale
    row = {"stage": stage, "measured_s": round(measured_s, 2), "scale_factor": round(scale, 2),
           "estimated_full_s": round(est, 1), "peak_rss_gb": round(rss_gb(), 2), "note": note, **dims}
    ROWS.append(row)
    print(f"{stage:<38} measured {measured_s:8.2f}s  x{scale:<8.2f} -> {est:9.1f}s  {note}", flush=True)


def realistic_motifs(base: list[Motif], n: int, rng) -> list[Motif]:
    """Grow a real Nanomotif motif set to n motifs with the same length/type/bipartite mix."""
    out, seen = list(base), {m.seq for m in base}
    while len(out) < n:
        tmpl = base[int(rng.integers(len(base)))]
        seq = "".join(c if c in "N" else str(rng.choice(list("ACGT"))) for c in tmpl.seq)
        need = "A" if tmpl.mod_type == "a" else "C"
        if seq[tmpl.mod_pos] != need:
            seq = seq[:tmpl.mod_pos] + need + seq[tmpl.mod_pos + 1:]
        if seq in seen or revcomp(seq) in seen:
            continue
        seen.add(seq)
        out.append(Motif(seq, tmpl.mod_pos, tmpl.mod_type))
    return out[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assembly", required=True)
    ap.add_argument("--pileup", required=True)
    ap.add_argument("--pileup-assembly", help="assembly matching --pileup (default: --assembly)")
    ap.add_argument("--motifs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-contigs", type=int, default=50_000)
    ap.add_argument("--n-motifs", type=int, default=1000)
    ap.add_argument("--n-groups", type=int, default=800, help="motif groups after merging")
    ap.add_argument("--bins", type=int, nargs="+", default=[400, 1000])
    ap.add_argument("--plasmids", type=int, default=5000)
    ap.add_argument("--binned-gb", type=float, default=1.5)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)

    # ---------- 1. motif search on real sequence ----------
    t = time.time(); contigs = pio.read_fasta(a.assembly); t_read = time.time() - t
    total_bp = sum(map(len, contigs.values()))
    record("read_assembly", t_read, 1.0, f"{len(contigs)} contigs, {total_bp / 1e9:.2f} Gb", contigs=len(contigs))
    motifs = realistic_motifs(pio.read_motifs(a.motifs), a.n_motifs, rng)
    names = list(contigs)
    sub, acc = {}, 0
    for c in names:  # ~150 Mb sample of real contigs
        sub[c] = contigs[c]; acc += len(contigs[c])
        if acc > 150e6:
            break
    sub_bp = sum(map(len, sub.values()))
    k = 5
    t = time.time(); rec = build_records(sub, motifs[:k]); dt = time.time() - t
    recs_per_bp_motif = rec.height / sub_bp / k
    bytes_per_rec = rec.estimated_size() / rec.height
    # scale: per motif (x n_motifs/k) and per bp (x total_bp/sub_bp); contig overhead scaled by contig ratio
    scale_bp = total_bp / sub_bp * a.n_motifs / k
    record("motif_search (build_records)", dt, scale_bp,
           f"{k} motifs on {len(sub)} contigs/{sub_bp / 1e6:.0f} Mb; {recs_per_bp_motif * 1e3:.2f} records/kb/motif",
           motifs=a.n_motifs)
    n_records = recs_per_bp_motif * total_bp * a.n_motifs
    ROWS.append({"stage": "records_table_memory", "estimated_full_s": None, "note":
                 f"{n_records / 1e9:.2f} G records x {bytes_per_rec:.0f} B = {n_records * bytes_per_rec / 1e9:.0f} GB"})
    print(ROWS[-1]["note"])
    del rec; gc.collect()

    # per-(motif, contig) python overhead: tiny contigs isolate it
    tiny = {f"t{i}": "".join(rng.choice(list("ACGT"), 200)) for i in range(2000)}
    t = time.time(); build_records(tiny, motifs[:20]); dt = time.time() - t
    per_pair = dt / (2000 * 20)
    record("motif_search overhead per (motif,contig)", dt, a.n_contigs * a.n_motifs / (2000 * 20),
           f"{per_pair * 1e6:.0f} us per pair; scaled to {a.n_contigs} contigs x {a.n_motifs} motifs")

    # ---------- 2. pileup streaming throughput ----------
    pa = pio.read_fasta(a.pileup_assembly) if a.pileup_assembly else contigs
    recs = build_records(pa, motifs[:40]).with_columns(
        pl.when(pl.col("strand") == 0).then(pl.lit("+")).otherwise(pl.lit("-")).alias("strand"))
    recs = recs.select("contig", "pos", "strand", "mod_type", "motif")
    size = os.path.getsize(a.pileup)
    t = time.time(); hit = pio.read_pileup_at_sites(a.pileup, recs); dt = time.time() - t
    record("pileup stream+join", dt, 1.0,
           f"{size / 1e9:.1f} GB real pileup, {recs.height / 1e6:.1f} M records -> {size / dt / 1e9:.3f} GB/s; "
           f"{hit.height / 1e6:.1f} M rows matched")
    del recs, hit, contigs, sub, pa; gc.collect()

    # ---------- 3. model stages at realistic dims ----------
    G = 64; grid = Grid(G); M = a.n_groups
    K = int(a.binned_gb * 1e9 / 50_000)
    det = {g: DetectionParams(0.02, 50, 0.9, 20) for g in range(M)}
    sites_per_chunk = recs_per_bp_motif * 50_000 * a.n_motifs / 2  # palindromes etc ~ 2 records per site
    # unit_loglik on a sample of chunks
    k_ch = 200
    n_sites = int(sites_per_chunk * k_ch)
    df = pl.DataFrame({"unit": rng.integers(0, k_ch, n_sites), "group": rng.integers(0, M, n_sites),
                       "n": rng.poisson(30, n_sites) + 1})
    df = df.with_columns((pl.col("n") * pl.Series(rng.random(n_sites) < 0.2).cast(pl.Int64) * 0.85)
                         .cast(pl.Int64).alias("k"))
    t = time.time(); unit_loglik(df, k_ch, M, det, grid); dt = time.time() - t
    record("unit likelihoods (chunks)", dt, K / k_ch, f"{sites_per_chunk:.0f} sites/chunk, K={K} chunks", K=K, M=M)
    ROWS.append({"stage": "L_chunks memory", "note": f"K*M*G*8 = {K * M * G * 8 / 1e9:.1f} GB (x2 with marginal copy)"})
    print(ROWS[-1]["note"])

    for H in a.bins:
        gc.collect()
        Kh = K
        # realistic peaked L: each chunk/group has a level near 0 or 1
        def fake_L(n_units, nsite=60):
            lvl = np.where(rng.random((n_units, M)) < 0.15, 0.95, 0.01)
            return (nsite * (lvl[..., None] * grid.logx + (1 - lvl[..., None]) * grid.log1mx)).astype(np.float64)
        frac = min(1.0, 4_000_000_000 / (Kh * M * G * 8))  # keep L_chunks under ~4 GB
        Ks = int(Kh * frac)
        Lc = fake_L(Ks)
        cb = rng.integers(0, H, Ks)
        t = time.time(); m0 = FittedModel(Lc, cb, H, ["a"] * M, grid, float("inf")); dt = time.time() - t
        record(f"FittedModel (prior EM + robust), H={H}", dt, 1 / frac if frac < 1 else 1.0,
               f"{Ks} of {Kh} chunks; scaling assumes linear in chunks", H=H)
        # population prior alone (dominant inner loop)
        t = time.time(); fit_population_prior(m0.ell, ["a"] * M); dt = time.time() - t
        record(f"population prior EM (1 call), H={H}", dt, 1.0, "called 5x per model, 8 tau + final = 45x", H=H)
        # scoring U plasmids against H bins
        U = a.plasmids
        Us = min(U, max(50, int(2_000_000_000 / (H * M * 8 * 2))))
        Lp = fake_L(Us, 20)
        rho = np.full((Us, M), 0.02)
        t = time.time(); sc = m0.score(Lp, rho, plasmid=True); dt = time.time() - t
        record(f"score plasmids, H={H}", dt, U / Us, f"{Us} of {U} plasmids; A,Pw tensors {2 * U * H * M * 8 / 1e9:.0f} GB at full U", H=H, U=U)
        del sc; gc.collect()
        # calibration: F = 180*H fragments x 8 tau, plus per-fragment LOO
        F = 180 * H
        Fs = min(F, max(50, int(2_000_000_000 / (H * M * 8 * 2))))
        Lf = fake_L(Fs, 20)
        rf = np.full((Fs, M), 0.02)
        t = time.time(); scf = m0.score(Lf, rf, plasmid=False); dt_s = time.time() - t
        n_loo = min(Fs, 200)
        t = time.time()
        for i in range(n_loo):
            m0.loo_rescore(scf, i, int(cb[i % len(cb)]), np.array([i % Ks]), rf[i])
        dt_l = (time.time() - t) / n_loo
        model_build = ROWS[-3]["estimated_full_s"]
        cal = 8 * (model_build + dt_s * F / Fs + dt_l * F)
        record(f"calibration (8 tau), H={H}", cal, 1.0,
               f"F={F} fragments: per tau = model {model_build:.0f}s + score {dt_s * F / Fs:.0f}s + LOO {dt_l * F:.0f}s; "
               f"fragment A,Pw tensors {2 * F * H * M * 8 / 1e9:.0f} GB", H=H, F=F)
        del scf; gc.collect()
        # assignability detectability: H marginalisations of U x M x G
        Ua = min(U, 1000)
        H1 = fake_L(Ua, 20); H0 = fake_L(Ua, 20)
        t = time.time()
        for h in range(5):
            Lx = 0.5 * H1 + 0.5 * H0
            Lpx = m0.marginal(Lx, True)
            from pbinner.model import score_units
            score_units(Lpx, m0.log_prof[h:h + 1]); score_units(Lpx, m0.log_pi[None])
        dt = (time.time() - t) / 5
        record(f"detectability (H hosts), H={H}", dt, H * U / Ua, f"per host {dt:.2f}s for {Ua} plasmids", H=H)
        # delta x tau_p grid: 42 scorings
        sc_full = [r for r in ROWS if r["stage"] == f"score plasmids, H={H}"][0]["estimated_full_s"]
        record(f"plasmid params (42 grid pts), H={H}", sc_full * 42, 1.0, "42 x full plasmid scoring", H=H)
        del m0, Lc, Lp, Lf, H1, H0; gc.collect()

    pl.DataFrame(ROWS, infer_schema_length=None).write_csv(out / "components.tsv", separator="\t")


if __name__ == "__main__":
    main()
