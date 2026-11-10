"""End-to-end runs on the small synthetic community (conftest.py): provenance, the log, edge-case inputs."""
import json
import polars as pl
import pytest

from conftest import run_community
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
