"""End to end on the synthetic community (tests/synth.py): the planted errors are found, nothing else is
touched, and the output files agree with each other."""
import inspect
import json
import re
from pathlib import Path

import pytest

from helpers import INVARIANT_FILES, OUTPUT_FILES, assert_same_outputs, read_tsv

DOCS = Path(__file__).resolve().parents[1] / "docs" / "outputs.md"


def test_planted_contaminants_are_flagged(community, base_run):
    dec = base_run.decisions
    flagged = dict(base_run.table("bin_contamination.tsv").select("contig", "cluster").iter_rows())
    for contig, home in community.contaminants.items():
        row = dec[contig]
        assert contig in flagged, f"{contig} (genome {community.genome_of[contig]} in {row['given_bin']}) not flagged"
        assert row["p_contaminant"] >= 0.99 and row["methylome_difference"] >= 0.4
        if home is None:  # from the genome without a bin: removed, not placed anywhere
            assert (row["status"], row["new_bin"], row["best_alternative"]) == ("contaminant", "", "unknown")
            assert flagged[contig] == "unknown"
        else:             # from another bin: moved back to it
            assert (row["status"], row["new_bin"], row["best_alternative"]) == ("moved", home, home)
            assert flagged[contig] == home


def test_only_planted_contaminants_are_flagged(community, base_run):
    flagged = set(base_run.table("bin_contamination.tsv")["contig"])
    assert flagged == set(community.contaminants) | {community.duplicate_binned}
    for contig, row in base_run.decisions.items():
        if row["given_bin"] and contig not in flagged:
            assert row["new_bin"] == row["given_bin"]
            if row["status"] == "kept":
                assert row["p_contaminant"] < 0.05


def test_unbinned_members_are_included_in_their_bin(community, base_run):
    dec = base_run.decisions
    for contig, home in community.missing.items():
        row = dec[contig]
        assert (row["status"], row["new_bin"], row["best_alternative"]) == ("included", home, home), contig
        assert row["p_best_alternative"] >= 0.9 and row["redundancy_with_bin"] < 0.3
    included = {c for c, r in dec.items() if r["status"] == "included"}
    assert included == set(community.missing)


def test_contigs_of_the_binless_genome_stay_unbinned(community, base_run):
    dec = base_run.decisions
    new = set(base_run.table("new_contig_bin.tsv")["contig"])
    for contig in community.orphans:
        row = dec[contig]
        assert (row["status"], row["new_bin"], row["given_bin"]) == ("unbinned", "", ""), contig
        assert row["best_alternative"] == "unknown" and row["p_unknown"] >= 0.9
        assert contig not in new


def test_contigs_without_a_decision(community, base_run):
    dec = base_run.decisions
    short, low, sole = dec[community.short], dec[community.low_coverage], dec[community.sole]
    assert short["status"] == "too_short" and short["new_bin"] == ""
    assert low["status"] == "low_coverage" and low["new_bin"] == low["given_bin"] and low["coverage"] < 5
    assert sole["status"] == "sole_contig_of_bin" and sole["new_bin"] == "binE"
    dup = dec[community.duplicate]  # its bin already holds this sequence
    assert dup["status"] == "redundant" and dup["new_bin"] == "" and dup["redundancy_with_bin"] == 1.0
    assert dup["best_alternative"] == community.true_bin[community.duplicate]


def test_documented_output_files_exist(base_run, cli_run):
    for f in OUTPUT_FILES:
        assert (base_run.out / f).stat().st_size > 0, f
    assert not (base_run.out / "motif_fit.tsv").exists()
    assert (cli_run.out / "motif_fit.tsv").stat().st_size > 0  # written with --motif-fit only
    # every file name the documentation mentions is covered by OUTPUT_FILES
    documented = set(re.findall(r"`([a-z_]+\.(?:tsv|json))`", DOCS.read_text()))
    optional = {"motif_fit.tsv", "unbinned_prior.tsv"}  # --motif-fit; --unknown-prior unbinned
    assert documented <= {Path(f).name for f in OUTPUT_FILES} | optional


def test_decision_table_has_the_documented_columns(community, base_run):
    df = base_run.table("contig_decisions.tsv")
    assert df.columns == [
        "contig", "length", "n_sites", "given_bin", "status", "new_bin", "p_contaminant", "methylome_difference",
        "redundancy_with_bin", "best_alternative", "p_best_alternative", "map", "p_map", "p_unknown",
        "credible_set_95", "expected_evidence", "coverage", "p_regional_foreign", "p_regional_mixed",
        "foreign_segments", "segment_bins"]
    section = DOCS.read_text().split("## `contig_decisions.tsv`")[1].split("### Statuses")[0]
    assert set(df.columns) == set(re.findall(r"`([a-z_0-9]+)`", section)) - {"mixed", "unknown"}
    assert df["contig"].to_list() == sorted(community.seqs)  # one row per contig of the assembly
    assert dict(zip(df["contig"], df["length"])) == {c: len(s) for c, s in community.seqs.items()}
    assert dict(zip(df["contig"], df["given_bin"])) == {c: community.given.get(c, "") for c in community.seqs}
    statuses = set(re.findall(r"^\| `([a-z_]+)` \|", DOCS.read_text().split("### Statuses")[1], re.M))
    assert set(df["status"]) <= statuses


def test_nanomotif_compatible_files_have_nanomotif_columns(base_run):
    assert base_run.table("bin_contamination.tsv").columns == [
        "contig", "bin", "method", "cluster", "bin_cluster", "bin_length", "n_contigs_bin", "fraction_contigs",
        "fraction_length", "p_contaminant", "p_cluster"]
    assert base_run.table("include_contigs.tsv").columns == [
        "contig", "bin", "assigned_bin", "method", "prob", "mean_prob", "confidence"]
    for f in ("new_contig_bin.tsv", "decontaminated_contig_bin.tsv"):
        assert base_run.table(f).columns == ["contig", "bin"]


def test_output_files_agree_with_each_other(community, base_run):
    dec = base_run.decisions
    flagged = base_run.table("bin_contamination.tsv")
    decont = dict(base_run.table("decontaminated_contig_bin.tsv").iter_rows())
    new = dict(base_run.table("new_contig_bin.tsv").iter_rows())
    cand = base_run.table("include_contigs.tsv")
    # the input bins minus the flagged contigs
    assert decont == {c: b for c, b in community.given.items() if c not in set(flagged["contig"])}
    # the refined binning is the decision table's new_bin column
    assert new == {c: r["new_bin"] for c, r in dec.items() if r["new_bin"]}
    # ... and equals the decontaminated bins plus the contigs included or moved
    added = {c: r["new_bin"] for c, r in dec.items() if r["status"] in ("included", "moved")}
    assert new == {**decont, **added} and not set(decont) & set(added)
    # every contig added is an inclusion candidate for that bin; candidates that were not added are the
    # ones the redundancy check dropped (include_contigs.tsv lists candidates, not decisions)
    cand_bin = dict(zip(cand["contig"], cand["assigned_bin"]))
    assert added.items() <= cand_bin.items()
    assert set(cand_bin) - set(added) == {community.duplicate, community.duplicate_binned}
    assert (cand["prob"] >= 0.5).all() and set(cand["method"]) == {"epirefine"}
    # flagged contigs: probabilities and bins as in the decision table
    for r in flagged.iter_rows(named=True):
        d = dec[r["contig"]]
        assert (r["bin"], r["cluster"], r["p_contaminant"]) == (d["given_bin"], d["best_alternative"], d["p_contaminant"])
        assert r["n_contigs_bin"] == sum(b == r["bin"] for b in community.given.values())
        assert r["bin_length"] == sum(len(community.seqs[c]) for c, b in community.given.items() if b == r["bin"])
    # bins: every input bin has a profile and an evidence row
    bins = sorted(set(community.given.values()))
    assert base_run.table("bin_evidence.tsv")["bin"].to_list() == bins
    assert sorted(set(base_run.table("host_profiles.tsv")["bin"])) == sorted(bins + ["_population_prior_"])


def test_log_counts_agree_with_the_files(base_run):
    n = base_run.log_counts()
    dec = base_run.decisions
    assert n["flagged"] == base_run.table("bin_contamination.tsv").height
    assert n["included"] == sum(r["status"] == "included" for r in dec.values())
    assert n["moved"] == sum(r["status"] == "moved" for r in dec.values())
    assert n["unbinned"] == sum(not r["given_bin"] for r in dec.values())


def test_flagged_statuses_agree_with_bin_contamination(base_run):
    """A flagged contig whose move is dropped as redundant keeps the status `contaminant` (review 01 finding 5;
    a strict xfail until 0.11, when `redundant` overwrote it)."""
    flagged = set(base_run.table("bin_contamination.tsv")["contig"])
    by_status = {c for c, r in base_run.decisions.items() if r["status"] in ("contaminant", "moved")}
    assert by_status == flagged


def test_flagged_contig_with_redundant_move_is_removed_from_its_bin(community, base_run):
    """What the outputs do agree on for that contig: it is flagged and in no bin of the refined binning."""
    c = community.duplicate_binned
    row = base_run.decisions[c]
    assert c in set(base_run.table("bin_contamination.tsv")["contig"])
    assert c not in set(base_run.table("new_contig_bin.tsv")["contig"]) and row["new_bin"] == ""
    assert row["p_contaminant"] >= 0.99 and row["redundancy_with_bin"] == 1.0


def test_params_json(base_run):
    params = json.loads((base_run.out / "params.json").read_text())
    assert {"tau", "temperature", "p_unknown", "contamination_rate", "chunk_size", "grid_size", "min_cov",
            "contam_eps", "unknown_eta", "unknown_lam", "fitted_temperature", "n_fragments"} <= set(params)
    assert 0 < params["p_unknown"] < 1 and 0 < params["contamination_rate"] < 0.5
    assert params["temperature"] >= 1.0  # never sharpened by default


def test_command_line_gives_the_same_outputs(base_run, cli_run):
    """cli.main with --workers 3 on a pileup read in several ranges against pipeline.run with one worker."""
    assert "pileup pass: 12 byte ranges on 3 workers" in cli_run.log
    assert "pileup pass: 1 byte ranges on 1 workers" in base_run.log
    assert_same_outputs(base_run.out, cli_run.out, INVARIANT_FILES)


def test_command_line_defaults_and_switches(monkeypatch):
    """Without options the command line passes run()'s own defaults; each --no-* switch turns one off."""
    from epirefine import cli, pipeline
    calls = []
    sig = inspect.signature(pipeline.run).parameters
    monkeypatch.setattr(pipeline, "run", lambda *a, **kw: calls.append((a, kw)))
    base = ["run", "--assembly", "a.fa", "--pileup", "p.bed", "--contig-bin", "cb.tsv", "--motifs", "m.tsv", "--out", "o"]
    cli.main(base)
    cli.main(base + ["--no-group-weights", "--no-motif-noise-filter", "--no-regional-model", "--no-calibrate",
                     "--regional-inclusion", "--allow-sharpening", "--motif-fit", "--group-sample-mb", "2",
                     "--frag-lengths", "2000,5000"])
    (args, plain), (_, switched) = calls
    assert args == ("a.fa", "p.bed", "cb.tsv", "m.tsv", "o")
    assert set(plain) == set(sig) - {"assembly", "pileup", "contig_bin", "motifs_path", "out"}
    assert {k: v for k, v in plain.items() if v != sig[k].default} == {}
    changed = {k: v for k, v in switched.items() if v != plain[k]}
    assert changed == {"group_weights": False, "motif_noise_filter": False, "regional_model": False,
                       "do_calibrate": False, "regional_inclusion": True, "allow_sharpening": True,
                       "write_motif_fit": True, "group_sample_bp": 2e6, "frag_lengths": (2000, 5000)}
