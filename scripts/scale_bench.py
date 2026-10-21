"""Scalability benchmark: generate synthetic communities of increasing size, run pbinner, record
per-stage wall time and peak memory.

    python scripts/scale_bench.py gen --out DIR --hosts 40 --motifs-per-host 2 4 --plasmids-per-host 3
    python scripts/scale_bench.py run --out bench/      # runs the configured grid, writes bench/results.tsv
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

from pbinner.motifs import Motif, find_records, revcomp

BASES = np.array(list("ACGT"))


def motif_pool(rng, n):
    pool, seen = [], set()
    while len(pool) < n:
        L = int(rng.integers(4, 7))
        seq = "".join(rng.choice(BASES, L))
        if seq in seen or revcomp(seq) in seen:
            continue
        mt = rng.choice(["a", "a", "m", "21839"])
        base = "A" if mt == "a" else "C"
        pos = [i for i, c in enumerate(seq) if c == base]
        if not pos:
            continue
        seen.add(seq)
        pool.append(Motif(seq, int(rng.choice(pos)), str(mt)))
    return pool


def generate(out, hosts=40, genome=1_000_000, motifs_per_host=(1, 4), plasmids_per_host=3, pool_size=200,
             cov=30.0, seed=1):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    pool = motif_pool(rng, pool_size)
    host_motifs = [list(rng.choice(len(pool), int(rng.integers(motifs_per_host[0], motifs_per_host[1] + 1)),
                                   replace=False)) for _ in range(hosts)]
    used = sorted({m for hm in host_motifs for m in hm})
    contigs, rows, contig_bin, plasmids = {}, [], {}, []
    for h in range(hosts):
        gc = rng.uniform(0.35, 0.65)
        p = [(1 - gc) / 2, gc / 2, gc / 2, (1 - gc) / 2]
        n_pieces = int(rng.integers(1, 6))
        cuts = np.sort(rng.choice(np.arange(20_000, genome - 20_000, 1000), n_pieces - 1, replace=False))
        bounds = np.r_[0, cuts, genome]
        units = [(f"h{h}_c{i}", int(bounds[i + 1] - bounds[i]), False) for i in range(n_pieces)]
        units += [(f"h{h}_p{j}", int(rng.choice([3_000, 10_000, 30_000])), True) for j in range(plasmids_per_host)]
        for name, L, is_p in units:
            seq = "".join(rng.choice(BASES, L, p=p))
            contigs[name] = seq
            if is_p:
                plasmids.append(name)
            else:
                contig_bin[name] = f"bin_{h}"
            ucov = cov * (rng.uniform(1, 5) if is_p else 1.0)
            mean = 0.88 if is_p else 0.96
            meth = set()
            for mi in host_motifs[h]:
                pos, strand, occ = find_records(seq, pool[mi])
                if not len(pos):
                    continue
                uo, inv = np.unique(occ, return_inverse=True)
                st = rng.random(len(uo)) < rng.beta(mean * 100, (1 - mean) * 100)
                sel = st[inv]
                meth.update(zip(pos[sel].tolist(), strand[sel].tolist(), [pool[mi].mod_type] * int(sel.sum())))
            recs = {}
            for mi in used:
                pos, strand, _ = find_records(seq, pool[mi])
                for pp, ss in zip(pos.tolist(), strand.tolist()):
                    recs[(pp, ss, pool[mi].mod_type)] = None
            if not recs:
                continue
            keys = list(recs)
            n = rng.poisson(ucov, len(keys))
            m = np.array([k in meth for k in keys])
            mu = np.where(m, 0.85, 0.03)
            k = rng.binomial(n, rng.beta(mu * 15, (1 - mu) * 15))
            ok = n > 0
            pos_arr = np.array([kk[0] for kk in keys])[ok]
            rows.append(pl.DataFrame({
                "c": name, "s": pos_arr, "e": pos_arr + 1, "code": np.array([kk[2] for kk in keys])[ok],
                "score": n[ok], "strand": np.where(np.array([kk[1] for kk in keys])[ok] == 0, "+", "-"),
                "s2": pos_arr, "e2": pos_arr + 1, "col": "255,0,0", "nv": n[ok],
                "frac": np.round(100 * k[ok] / n[ok], 2), "nm": k[ok], "nc": n[ok] - k[ok],
                "o1": 0, "o2": 0, "o3": 0, "o4": 0, "o5": 0}))
    with open(out / "assembly.fasta", "w") as fh:
        for c, s in contigs.items():
            fh.write(f">{c}\n{s}\n")
    pl.concat(rows).write_csv(out / "pileup.bed", separator="\t", include_header=False)
    (out / "contig_bin.tsv").write_text("".join(f"{c}\t{b}\n" for c, b in contig_bin.items()))
    (out / "plasmids.txt").write_text("\n".join(plasmids) + "\n")
    pl.DataFrame([{"bin": f"bin_{h}", "motif": pool[mi].seq, "mod_position": pool[mi].mod_pos,
                   "mod_type": pool[mi].mod_type} for h in range(hosts) for mi in host_motifs[h]]
                 ).write_csv(out / "bin-motifs.tsv", separator="\t")
    truth = [{"contig": p_, "expected": f"bin_{p_.split('_')[0][1:]}"} for p_ in plasmids]
    pl.DataFrame(truth).write_csv(out / "truth.tsv", separator="\t")
    return {"contigs": len(contigs), "assembly_mb": sum(map(len, contigs.values())) / 1e6,
            "pileup_rows": sum(r.height for r in rows), "motifs_used": len(used), "plasmids": len(plasmids)}


STAGES = [  # (label, regex on the log line that ENDS the stage)
    ("read_inputs", r"contigs, \d+ bins"),
    ("motif_records", r"motifs -> \d+ groups"),
    ("pileup", r"sites with coverage"),
    ("caller_model", r"detection model fitted"),
    ("chunk_likelihoods", r"host chunks of"),
    ("calibration", r"calibrated: tau"),
    ("robust_profiles", r"robust host profiles"),
    ("plasmid_params", r"plasmid params|MTase annotations|robust host profiles"),
    ("scoring_outputs", r"done ->"),
]


def parse_log(text):
    times = {}
    for line in text.splitlines():
        m = re.match(r"\[\s*([\d.]+)s\] (.*)", line)
        if not m:
            continue
        t, msg = float(m.group(1)), m.group(2)
        for label, pat in STAGES:
            if re.search(pat, msg):
                times[label] = t
    out, prev = {}, 0.0
    for label, _ in STAGES:
        if label in times:
            out[label] = round(times[label] - prev, 1)
            prev = times[label]
    out["total"] = round(prev, 1)
    m = re.search(r"(\d+) motifs -> (\d+) groups, (\d+) motif records", text)
    if m:
        out.update(groups=int(m.group(2)), records=int(m.group(3)))
    m = re.search(r"(\d+) sites with coverage", text)
    if m:
        out["sites"] = int(m.group(1))
    m = re.search(r"(\d+) host chunks", text)
    if m:
        out["chunks"] = int(m.group(1))
    m = re.search(r"Maximum resident set size \(kbytes\): (\d+)", text)
    if m:
        out["peak_rss_gb"] = round(int(m.group(1)) / 1e6, 2)
    return out


def run_one(name, cfg, root: Path, pbinner: str):
    d = root / name
    t0 = time.time()
    info = generate(d / "data", **cfg)
    gen_s = time.time() - t0
    cmd = ["/usr/bin/time", "-v", pbinner, "run", "--assembly", str(d / "data/assembly.fasta"),
           "--pileup", str(d / "data/pileup.bed"), "--contig-bin", str(d / "data/contig_bin.tsv"),
           "--motifs", str(d / "data/bin-motifs.tsv"), "--plasmids", str(d / "data/plasmids.txt"),
           "--out", str(d / "out")]
    p = subprocess.run(cmd, capture_output=True, text=True)
    (d / "run.log").write_text(p.stdout + p.stderr)
    res = {"config": name, **{k: v for k, v in cfg.items() if k != "motifs_per_host"},
           "motifs_per_host": f"{cfg.get('motifs_per_host', (1, 4))}", **info,
           "pileup_mb": round((d / "data/pileup.bed").stat().st_size / 1e6, 1), "gen_s": round(gen_s, 1),
           "exit": p.returncode, **parse_log(p.stdout + p.stderr)}
    if p.returncode == 0:
        a = pl.read_csv(d / "out/assignments.tsv", separator="\t")
        t = pl.read_csv(d / "data/truth.tsv", separator="\t")
        j = a.join(t, on="contig")
        res["plasmid_acc"] = round((j["assignment"] == j["expected"]).mean(), 3)
        res["in_set95"] = round(j.select(pl.col("credible_set_95").str.split(",").list.contains(
            pl.col("expected"))).to_series().mean(), 3)
    print(res, flush=True)
    return res


GRID = {
    # bins (assembly size grows with it)
    "H10": dict(hosts=10), "H20": dict(hosts=20), "H40": dict(hosts=40), "H80": dict(hosts=80),
    "H160": dict(hosts=160),
    # motif groups at fixed bins
    "H40_M2": dict(hosts=40, motifs_per_host=(2, 2)), "H40_M8": dict(hosts=40, motifs_per_host=(8, 8)),
    # plasmid candidates at fixed bins
    "H40_P15": dict(hosts=40, plasmids_per_host=15), "H40_P60": dict(hosts=40, plasmids_per_host=60),
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--out", required=True); g.add_argument("--hosts", type=int, default=40)
    g.add_argument("--motifs-per-host", type=int, nargs=2, default=(1, 4))
    g.add_argument("--plasmids-per-host", type=int, default=3); g.add_argument("--seed", type=int, default=1)
    r = sub.add_parser("run")
    r.add_argument("--out", required=True); r.add_argument("--only", nargs="*")
    a = ap.parse_args()
    if a.cmd == "gen":
        print(generate(a.out, a.hosts, motifs_per_host=tuple(a.motifs_per_host),
                       plasmids_per_host=a.plasmids_per_host, seed=a.seed))
    else:
        root = Path(a.out); root.mkdir(parents=True, exist_ok=True)
        pb = str(Path(sys.executable).parent / "pbinner")
        results = []
        for name, cfg in GRID.items():
            if a.only and name not in a.only:
                continue
            results.append(run_one(name, cfg, root, pb))
            pl.DataFrame(results).write_csv(root / "results.tsv", separator="\t")
