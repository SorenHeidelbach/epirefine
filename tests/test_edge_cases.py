"""Unusual inputs, on the synthetic community with one thing changed. Tests without a mark pin what the
current code does; strict xfails state what it should do where the 2026-11-10 reviews confirmed a defect."""
import math

import pytest

import synth
from helpers import DECISION_FILES, assert_same_outputs, copy_cache, run_pipeline


# --- one bin -----------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def single_bin_run(community, base_run, tmp_path_factory):
    """Only binA in contig_bin, with the three foreign contigs planted in it; everything else unbinned."""
    d = tmp_path_factory.mktemp("single_bin")
    synth.write_contig_bin(d / "contig_bin.tsv", {c: b for c, b in community.given.items() if b == "binA"})
    return run_pipeline(**{**community.args(copy_cache(base_run, d / "out")), "contig_bin": str(d / "contig_bin.tsv")})


def _foreign_in_bin_a(community):
    return sorted(c for c, b in community.given.items() if b == "binA" and community.true_bin[c] != "binA")


def test_single_bin_run_completes_and_still_includes(community, single_bin_run):
    """Current behaviour with one bin: the run finishes, inclusion works, P(contaminant) is NaN for every
    binned contig and the contamination rate is NaN."""
    dec = single_bin_run.decisions
    assert "56 contigs, 1 bins" in single_bin_run.log
    included = {c for c, r in dec.items() if r["status"] == "included"}
    assert included and all(community.true_bin[c] == "binA" for c in included)
    assert {c for c, b in community.missing.items() if b == "binA"} <= included
    binned = [r for r in dec.values() if r["given_bin"]]
    assert len(binned) == 14 and all(math.isnan(r["p_contaminant"]) for r in binned)
    assert "contamination rate=nan" in single_bin_run.log
    assert single_bin_run.table("bin_contamination.tsv").height == 0
    # the foreign contigs are visibly different (methylome difference above the 0.4 a flag needs) and unflagged
    for c in _foreign_in_bin_a(community):
        assert dec[c]["methylome_difference"] > 0.8 and dec[c]["status"] == "kept"


@pytest.mark.xfail(strict=True, reason="refine.refine: with one bin the number of alternative bins is 0, -inf - log(0) "
                   "gives NaN for P(foreign) of every binned contig, and no contaminant can be flagged "
                   "(review 01 finding 1, review 03 F2)")
def test_single_bin_contaminants_are_flagged(community, single_bin_run):
    foreign = _foreign_in_bin_a(community)
    assert len(foreign) == 3
    assert sorted(single_bin_run.table("bin_contamination.tsv")["contig"]) == foreign


# --- contig_bin rows that match nothing ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def absent_contigs_run(community, base_run, tmp_path_factory):
    """contig_bin with three extra rows naming contigs that are not in the assembly: one in binA, two forming
    a bin of their own (`ghost`, a bin without any contig of the assembly)."""
    d = tmp_path_factory.mktemp("absent")
    extra = {"nope_1": "binA", "nope_2": "ghost", "nope_3": "ghost"}
    synth.write_contig_bin(d / "contig_bin.tsv", {**community.given, **extra})
    return run_pipeline(**{**community.args(copy_cache(base_run, d / "out")), "contig_bin": str(d / "contig_bin.tsv")})


def test_contig_bin_rows_absent_from_the_assembly_change_no_decision(base_run, absent_contigs_run):
    """Such rows are dropped (and reported, next test), a bin made only of them does not exist for the run, and
    every decision is the same as without them."""
    run = absent_contigs_run
    assert_same_outputs(base_run.out, run.out, DECISION_FILES)
    assert "56 contigs, 4 bins, 43 host contigs" in run.log
    for f in ("new_contig_bin.tsv", "decontaminated_contig_bin.tsv", "bin_evidence.tsv", "host_profiles.tsv"):
        text = (run.out / f).read_text()
        assert "nope_" not in text and "ghost" not in text, f


def test_contig_bin_rows_absent_from_the_assembly_are_reported(absent_contigs_run):
    assert "3" in next(line for line in absent_contigs_run.log.splitlines() if "not in the assembly" in line)


def test_contig_bin_matching_no_contig_gives_a_clear_error(community, base_run, tmp_path):
    spaces = tmp_path / "contig_bin.tsv"
    spaces.write_text("".join(f"{c} {b}\n" for c, b in community.given.items()))
    with pytest.raises((ValueError, SystemExit), match="(?i)contig.bin|no bin"):
        run_pipeline(**{**community.args(copy_cache(base_run, tmp_path / "out")), "contig_bin": str(spaces)})


# --- pileup and motif table that match nothing ---------------------------------------------------------------

def test_pileup_matching_no_contig_gives_a_clear_error(community, tmp_path):
    other = tmp_path / "pileup.bed"
    other.write_text("".join("x" + row for row in community.pileup.read_text().splitlines(keepends=True)))
    with pytest.raises((ValueError, SystemExit), match="(?i)pileup"):
        run_pipeline(**{**community.args(tmp_path / "out"), "pileup": str(other)})


def test_pileup_matching_no_contig_stops_before_any_decision(community, tmp_path):
    """Current behaviour: an exception (not a result) and no decision file."""
    other = tmp_path / "pileup.bed"
    other.write_text("".join("x" + row for row in community.pileup.read_text().splitlines(keepends=True)))
    with pytest.raises(Exception):
        run_pipeline(**{**community.args(tmp_path / "out"), "pileup": str(other)})
    assert not (tmp_path / "out" / "contig_decisions.tsv").exists()
    assert not (tmp_path / "out" / "new_contig_bin.tsv").exists()


def test_motif_table_without_motifs_is_an_error(community, tmp_path):
    """A header-only motif table stops at once with a ValueError that names the cause."""
    empty = tmp_path / "bin-motifs.tsv"
    empty.write_text(community.motifs.read_text().split("\n", 1)[0] + "\n")
    with pytest.raises(ValueError, match="no valid motif"):
        run_pipeline(**{**community.args(tmp_path / "out"), "motifs_path": str(empty)})
    assert not (tmp_path / "out" / "cache" / "manifest.json").exists()  # stopped before the pileup pass


def test_motif_table_without_the_needed_columns_is_an_error(community, tmp_path):
    bad = tmp_path / "bin-motifs.tsv"
    bad.write_text("bin\tmotif\nbinA\tGATC\n")
    with pytest.raises(ValueError, match="lacks the column.s. mod_position, mod_type"):
        run_pipeline(**{**community.args(tmp_path / "out"), "motifs_path": str(bad)})


# --- a bin without any site ----------------------------------------------------------------------------------

def test_bin_without_sites_takes_no_inclusions(community, base_run, tmp_path):
    """binE's only contig has no row in the pileup, so the bin has no site and no profile of its own.
    Current behaviour: the contig gets no decision (low_coverage), the bin is reported as
    no_methylation_evidence, nothing is included into it, and the planted errors of the other bins are
    still found (review 01 finding 7 suspects inclusions into such a bin when p_unknown is near zero)."""
    pileup = tmp_path / "pileup.bed"
    pileup.write_text("".join(r for r in community.pileup.read_text().splitlines(keepends=True)
                              if not r.startswith(community.sole + "\t")))
    run = run_pipeline(**{**community.args(tmp_path / "out"), "pileup": str(pileup)})
    dec = run.decisions
    sole = dec[community.sole]
    assert (sole["n_sites"], sole["coverage"], sole["status"], sole["new_bin"]) == (0, 0.0, "low_coverage", "binE")
    evidence = {r["bin"]: r for r in run.table("bin_evidence.tsv").iter_rows(named=True)}
    assert evidence["binE"]["status"] == "no_methylation_evidence" and evidence["binE"]["sites"] == 0
    assert [c for c, r in dec.items() if r["new_bin"] == "binE"] == [community.sole]
    assert set(run.table("bin_contamination.tsv")["contig"]) == set(base_run.table("bin_contamination.tsv")["contig"])
    for c, home in community.missing.items():
        assert (dec[c]["status"], dec[c]["new_bin"]) == ("included", home)
    for c in community.orphans:
        assert dec[c]["new_bin"] == "" and dec[c]["status"] in ("unbinned", "no_methylation_evidence")


# --- duplicate contig names in the assembly --------------------------------------------------------------------

def test_duplicate_contig_name_in_the_assembly(community, tmp_path):
    """A second FASTA record under the name of an existing contig is an error that names the contig (0.11; before,
    the name silently kept one row with the length of the last record)."""
    name = community.orphans[0]
    other = community.seqs[next(iter(community.missing))][:5000]
    assembly = tmp_path / "assembly.fasta"
    assembly.write_text(community.assembly.read_text() + f">{name}\n{other}\n")
    with pytest.raises(ValueError, match=f"{name}.*more than once"):
        run_pipeline(**{**community.args(tmp_path / "out"), "assembly": str(assembly)})
