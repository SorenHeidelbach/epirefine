"""End-to-end runs on the small synthetic community (conftest.py): provenance, the log, edge-case inputs."""
import json
import math
import warnings
import polars as pl
import pytest

from conftest import run_community, write_community
from epirefine.io import InputError


def _strict_json(path):
    def bad(name):
        raise ValueError(f"{name} is not JSON")
    return json.loads(path.read_text(), parse_constant=bad)


@pytest.fixture(scope="module")
def finished(community, tmp_path_factory):
    out = tmp_path_factory.mktemp("run") / "out"
    run_community(community, out, seed=7, include_threshold=0.91)
    return out


def test_params_json_is_valid_json_with_provenance(community, finished):
    import epirefine
    p = _strict_json(finished / "params.json")  # tau is infinite on this community: "inf", not the bare word
    assert p["tau"] == "inf"
    # the keys written before this change are still there
    for k in ("chunk_size", "grid_size", "p_unknown", "min_cov", "contam_eps", "fitted_temperature", "n_fragments",
              "unknown_eta", "unknown_lam", "tau", "temperature", "contamination_rate"):
        assert k in p, k
    assert p["version"] == epirefine.__version__ and p["seed"] == 7 and p["workers"] == 1 and p["finished"] is True
    assert p["include_threshold"] == 0.91 and p["contamination_threshold"] == 0.5
    for k in ("include_threshold_unmethylated", "min_methylome_difference", "max_redundancy", "min_evidence",
              "min_contig_coverage", "min_contig_length", "motif_noise_filter", "group_weights", "regional_model",
              "regional_inclusion", "do_calibrate", "frag_lengths", "frags_per_bin", "frag_budget", "allow_sharpening",
              "min_jaccard", "variation", "group_sample_bp", "p_unknown_given", "tau_given", "temperature_given",
              "start_time", "command_line"):
        assert k in p, k
    assert p["p_unknown_given"] is None and isinstance(p["p_unknown"], float)
    for name in ("assembly", "pileup", "contig_bin", "motifs"):
        st = community[name].stat()
        assert p["inputs"][name] == {"path": str(community[name].resolve()), "size": st.st_size, "mtime": int(st.st_mtime)}


def test_params_json_has_the_settings_of_a_run_that_did_not_finish(community, tmp_path):
    with pytest.raises(InputError):
        run_community(community, tmp_path / "out", min_contig_coverage=1000.0)
    p = _strict_json(tmp_path / "out" / "params.json")
    assert p["finished"] is False and p["min_contig_coverage"] == 1000.0 and "version" in p


def test_json_safe_writes_no_bare_nan_or_infinity(tmp_path):
    import numpy as np
    from epirefine.runinfo import write_params
    write_params(tmp_path / "p.json", {"a": float("inf"), "b": float("nan"), "c": np.array([1.0, -np.inf]),
                                       "d": np.float32(0.5), "e": (1, 2)})
    assert _strict_json(tmp_path / "p.json") == {"a": "inf", "b": None, "c": [1.0, "-inf"], "d": 0.5, "e": [1, 2]}


def test_single_bin_skips_contamination_detection_with_a_warning(tmp_path, capsys):
    paths = write_community(tmp_path / "data", single_bin=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)  # no "divide by zero" / "invalid value"
        run_community(paths, tmp_path / "out")
    log = capsys.readouterr().out
    assert "WARNING: the contig-bin table names one bin only" in log and "contamination detection is skipped" in log
    d = pl.read_csv(tmp_path / "out" / "contig_decisions.tsv", separator="\t")
    binned = d.filter(pl.col("given_bin") == "g1")
    assert binned.height == 3
    # binned contigs are not tested: kept in their bin, no P(contaminant), the only alternative is no bin at all
    assert binned["new_bin"].to_list() == ["g1"] * 3 and binned["p_contaminant"].is_nan().all()
    assert binned["best_alternative"].to_list() == ["unknown"] * 3 and binned["p_best_alternative"].to_list() == [1.0] * 3
    assert not d["p_best_alternative"].is_nan().any()
    assert pl.read_csv(tmp_path / "out" / "bin_contamination.tsv", separator="\t").height == 0
    p = _strict_json(tmp_path / "out" / "params.json")
    assert p["contamination_rate"] is None and p["contamination_detection"] == "skipped: one bin only"
    assert math.isfinite(p["p_unknown"])


def test_refine_with_one_bin_gives_no_nan_alternatives():
    import numpy as np
    from epirefine.refine import refine
    # two binned contigs (one fits its bin badly) and two unbinned ones (one fits the bin)
    S = np.array([[0.], [-500.], [0.], [-80.]])
    Su = np.array([-40., -50., -40., -40.])
    given = np.array([0, 0, -1, -1])
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        r = refine(S, Su, given, given >= 0, temperature=1.0)
    assert not r.contaminant.any() and np.isnan(r.p_foreign).all() and np.isnan(r.eps)
    assert r.best_alt.tolist()[:2] == [1, 1] and r.p_best_alt.tolist()[:2] == [1.0, 1.0]
    assert not np.isnan(r.p_best_alt).any() and r.include_bin.tolist() == [-1, -1, 0, -1]
