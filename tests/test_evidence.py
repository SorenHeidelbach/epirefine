"""Evaluation tooling for the referee evidence: per-contig metrics, trivial baselines, harder planted scenarios."""
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
em = pytest.importorskip("evidence_metrics")
tb = pytest.importorskip("trivial_baselines")
hs = pytest.importorskip("hard_scenarios")
from evaluate_refinement import evaluate  # noqa: E402

from epirefine.sites import EncodedAssembly  # noqa: E402


def _truth():
    return pl.DataFrame({
        "contig": list("abcdefgh"), "length": [3000, 8000, 15000, 30000, 60000, 4000, 4000, 4000],
        "true_bin": ["y", "y", "x", "x", "x", "", "", "y"], "given_bin": ["x", "x", "x", "", "", "", "x", "y"],
        "category": ["contaminant", "contaminant", "member", "missing", "missing", "orphan", "orphan_contaminant", "member"]})


def test_tally_gives_the_evaluators_counts(tmp_path):
    t = _truth()
    t.write_csv(tmp_path / "truth.tsv", separator="\t")
    flagged = {"a", "c"}                                  # one right, one false flag; b and g missed
    final = {"a": "y", "b": "x", "d": "x", "e": "y", "f": "x", "g": "x", "h": "y"}   # d right, e wrong, f orphan placed
    pl.DataFrame({"contig": sorted(flagged)}).write_csv(tmp_path / "c.tsv", separator="\t")
    pl.DataFrame({"contig": list(final), "bin": list(final.values())}).write_csv(tmp_path / "n.tsv", separator="\t")
    r = evaluate(tmp_path / "truth.tsv", tmp_path / "c.tsv", tmp_path / "n.tsv")
    c = {k: int(v) for k, v in zip(em.COUNTS, em.tally(t, flagged, final).select(em.COUNTS).sum().row(0))}
    assert (c["tp"], c["fp"], c["fn"]) == (1, 1, 2) and c["tp"] + c["fp"] == r["flagged"]
    assert (c["want"], c["included_ok"]) == (r["to_include"], r["included_correct"]) == (4, 2)
    assert (c["new"], c["new_wrong"]) == (r["new_placements"], r["new_placements_wrong"])
    assert (c["stay"], c["stay_wrong"]) == (r["should_stay_unbinned"], r["placed_wrongly"]) == (2, 2)
    rt = em.rates(c)
    assert rt["contam_precision"] == r["contam_precision"] and rt["contam_recall"] == r["contam_recall"]
    assert rt["inclusion_recall"] == r["inclusion_recall"]


def test_precision_recall_curve_and_matched_precision():
    score = np.array([0.9, 0.8, 0.8, 0.3, np.nan])
    label = np.array([1, 1, 0, 1, 1])                     # the unscored positive can never be found
    thr, p, r = em.pr_curve(score, label)
    assert list(thr) == [0.9, 0.8, 0.3]                   # ties form one threshold
    assert np.allclose(p, [1, 2 / 3, 3 / 4]) and np.allclose(r, [1 / 4, 2 / 4, 3 / 4])
    assert em.average_precision(score, label) == pytest.approx(1 * .25 + 2 / 3 * .25 + .75 * .25)
    assert em.recall_at_precision(score, label, 1.0) == (0.25, 0.9)
    assert em.recall_at_precision(score, label, 0.7)[0] == 0.75
    assert em.recall_at_precision(np.array([1.0]), np.array([0]), 0.5, positive=np.array([1])) == (0.0, float("inf"))
    # a weight of 2 is the same as listing the item twice
    w = np.array([2, 1, 1, 1, 1])
    assert em.average_precision(score, label, weight=w) == pytest.approx(em.average_precision(np.r_[0.9, score], np.r_[1, label]))


def test_cluster_bootstrap_resamples_whole_clusters():
    counts = np.array([[10, 0], [0, 10]], float)          # one cluster all right, one all wrong
    lo, hi = em.cluster_bootstrap(counts, lambda c: c[0] / c.sum(), n_boot=500)
    assert lo == 0.0 and hi == 1.0                        # a contig-level interval would be about 0.3-0.7
    assert list(em.length_class([2500, 5000, 70000])) == ["2-5 kb", "5-10 kb", ">= 50 kb"]


def _clean(n=40):
    rows = []
    for g in ("A-one", "A-two", "B-one", "C-orph"):
        for i in range(n):
            rows.append({"contig": f"{g}__c__p{i}", "length": 2000 + 500 * i, "true_bin": "" if g == "C-orph" else g,
                         "given_bin": "" if g == "C-orph" else g, "category": "orphan" if g == "C-orph" else "member"})
    return pl.DataFrame(rows)


def test_reset_undoes_planted_contaminants():
    t = hs.reset(_truth())
    assert t.filter(pl.col("contig") == "a").row(0, named=True)["given_bin"] == "y"
    assert t.filter(pl.col("contig") == "g").row(0, named=True)["given_bin"] == ""
    assert set(t["category"]) == {"member", "missing", "orphan"}


def test_planting_variants():
    t = _clean()
    genome = {c: c.split("__")[0] for c in t["contig"]}
    cov = {"A-one": 30.0, "A-two": 10.0, "B-one": 20.0, "C-orph": 6.0}
    tax = {g: {"genus": g.split("-")[0]} for g in cov}
    rng = np.random.default_rng(1)
    new, p = hs.plant(t, genome, 12, rng, length=(2000, 5000))
    assert p.height == 12 and (new.join(p, on="contig")["length"] <= 5000).all()
    assert set(new.join(p, on="contig")["category"]) <= {"contaminant", "orphan_contaminant"}
    j = p.join(new, on="contig")
    assert (j["given_bin"] == j["bin"]).all() and (p["source"] != p["bin"]).all()
    _, p = hs.plant(t, genome, 12, rng, dest="related", taxonomy=tax)
    assert set(p["relation"]) == {"genus"} and set(p["source"]) == {"A-one", "A-two"}
    _, p = hs.plant(t, genome, 12, rng, coverage="match", genome_cov=cov)
    assert all(cov[s] >= cov[b] for s, b in zip(p["source"], p["bin"]))
    assert np.allclose(p["thin"], [cov[b] / cov[s] for s, b in zip(p["source"], p["bin"])])
    _, p = hs.plant(t, genome, 12, rng, coverage=20.0, genome_cov=cov)
    assert set(p["source"]) <= {"A-one", "B-one"} and p["thin"].min() == pytest.approx(20 / 30)
    dist = {(a, b): abs(len(a) - len(b)) + (a[0] != b[0]) for a in cov for b in cov}
    _, p = hs.plant(t, genome, 12, rng, dest="nearest", dist=dist)
    assert set(zip(p["source"], p["bin"])) <= {("A-one", "A-two"), ("A-two", "A-one"), ("B-one", "A-one"), ("B-one", "A-two"),
                                               ("C-orph", "A-one"), ("C-orph", "A-two"), ("C-orph", "B-one")}
    _, p = hs.plant(t, genome, 10, rng, group=5)
    assert p.group_by("source", "bin").len()["len"].to_list() == [5, 5]
    new, p = hs.plant(_clean(4), genome, 50, rng, min_keep=3)        # bins are never emptied
    assert new.filter(pl.col("category") == "member").group_by("true_bin").len()["len"].min() == 3


def _cache(tmp_path, seqs, sites):
    asm = EncodedAssembly.write(seqs, tmp_path / "asm")
    (tmp_path / "sites").mkdir()
    pl.DataFrame(sites, schema={"contig": pl.Int32, "group": pl.Int32, "pos": pl.Int64, "n": pl.Int32, "k": pl.Int32}, orient="row"
                 ).write_parquet(tmp_path / "sites" / "r0.parquet")
    (tmp_path / "manifest.json").write_text(json.dumps({"inputs": {}, "parts": ["somewhere/else/sites/r0.parquet"]}))
    return asm


def test_thin_cache_thins_only_the_named_contigs(tmp_path):
    rng = np.random.default_rng(0)
    seqs = {c: "".join(rng.choice(list("ACGT"), 500)) for c in ("a", "b")}
    sites = [(c, 0, p, 20, 10) for c in (0, 1) for p in range(200)]
    _cache(tmp_path / "src", seqs, sites)
    hs.thin_cache(tmp_path / "src", tmp_path / "dst", {"b": 0.25}, seed=1)
    d = pl.read_parquet(tmp_path / "dst" / "sites" / "r0.parquet")
    a, b = d.filter(pl.col("contig") == 0), d.filter(pl.col("contig") == 1)
    assert a["n"].to_list() == [20] * 200 and a["k"].to_list() == [10] * 200
    assert 4 < b["n"].mean() < 6 and (b["k"] <= b["n"]).all() and (b["n"] > 0).all()
    assert (tmp_path / "dst" / "manifest.json").exists() and (tmp_path / "dst" / "asm" / "assembly.u8").exists()


def test_composition_is_strand_symmetric(tmp_path):
    rng = np.random.default_rng(0)
    s = "".join(rng.choice(list("ACGT"), 3000, p=[.1, .4, .4, .1]))
    rc = s[::-1].translate(str.maketrans("ACGT", "TGCA"))
    tnf, gc = tb.composition(_cache(tmp_path, {"f": s, "r": rc, "n": "ACGTN" * 10}, []))
    assert np.allclose(tnf[0], tnf[1]) and tnf[0].sum() == pytest.approx(1) and gc[0] == pytest.approx(gc[1])
    assert 0.75 < gc[0] < 0.85 and gc[2] == 0.5


def test_baselines_find_the_obvious_outlier(tmp_path):
    """Two bins: x (AT-rich, 20x, motif group 0 methylated) and y (GC-rich, 60x, group 0 unmethylated).
    A y contig planted in x must top every contamination score, and unbinned contigs go back to their genome."""
    rng = np.random.default_rng(0)
    seqs, sites, given = {}, [], {}
    for i in range(24):
        own = "x" if i < 12 else "y"
        name = f"{own}{i}"
        seqs[name] = "".join(rng.choice(list("ACGT"), 6000, p=[.35, .15, .15, .35] if own == "x" else [.15, .35, .35, .15]))
        n = 20 if own == "x" else 60
        for p in range(0, 6000, 100):
            cov = int(rng.poisson(n)) + 1
            sites.append((i, 0, p, cov, int(rng.binomial(cov, 0.9 if own == "x" else 0.03))))
        if i not in (11, 23):                              # one contig of each genome is left unbinned
            given[name] = own
    given["y12"] = "x"                                     # the planted contaminant
    _cache(tmp_path, seqs, sites)
    f = tb.load_features(tmp_path)
    assert f["cov_median"][0] == pytest.approx(21, abs=4) and f["S"][0, 0] == 60
    c = tb.contamination_scores(f, given)
    for b in ("coverage", "tetra", "gc", "cov_tetra", "naive_meth", "naive_meth2"):
        assert c.sort(b, descending=True)["contig"][0] == "y12", b
    assert c.filter(pl.col("contig") == "y12")["naive_meth"][0] > 0.9
    assert c.filter(pl.col("contig") != "y12")["naive_meth"].max() < 0.3
    assert tb.contamination_scores(f, given, min_sites=100)["naive_meth"].max() == 0   # too few sites: no evidence
    inc = tb.inclusion_scores(f, given)
    for b in ("naive_meth", "cov_tetra"):
        got = dict(inc.filter(pl.col("baseline") == b).select("contig", "bin").iter_rows())
        assert got == {"x11": "x", "y23": "y"}, b
    assert dict(inc.filter(pl.col("baseline") == "naive_meth").select("contig", "rule_bin").iter_rows()) == {"x11": "x", "y23": "y"}
