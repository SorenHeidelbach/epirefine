"""Labels in the output files agree with the decisions applied to the bins."""
import numpy as np
import polars as pl

from epirefine.refine import refine, write_outputs


def _read(path):
    return pl.read_csv(path, separator="\t")


def test_move_dropped_as_redundant_stays_a_contaminant(tmp_path):
    # a, a2: members of x. b: binned in x but fits y (flagged, move to y). c, d: unbinned, fit z and y.
    S = np.array([[0., -60, -60], [0., -60, -60], [-60, 0., -60], [-60, -60, 0.], [-60, 0., -60]])
    Su = np.full(5, -40.)
    given = np.array([0, 0, 0, -1, -1])
    contigs = ["a", "a2", "b", "c", "d"]
    r = refine(S, Su, given, given >= 0, temperature=1.0, methylome_difference=np.array([0, 0, 0.9, 0, 0]))
    assert r.contaminant.tolist() == [False, False, True, False, False]
    assert r.include_bin.tolist() == [-1, -1, 1, 2, 1]
    # the redundancy check (pipeline.run) drops the move of b and the inclusion of c
    redundancy = np.array([np.nan, np.nan, 0.8, 0.9, 0.0])
    r.include_bin[[2, 3]] = -1
    write_outputs(tmp_path, contigs, {c: 5000 for c in contigs}, given, ["x", "y", "z"], r, np.full(5, 100), set(),
                  np.full((5, 4), 0.25), redundancy=redundancy, evidence=np.full(5, 20.0))

    d = _read(tmp_path / "contig_decisions.tsv")
    assert dict(d.select("contig", "status").iter_rows()) == {
        "a": "kept", "a2": "kept", "b": "contaminant", "c": "redundant", "d": "included"}
    # b is removed from its bin and listed as a contaminant, whatever became of its move
    assert _read(tmp_path / "bin_contamination.tsv")["contig"].to_list() == ["b"]
    assert sorted(_read(tmp_path / "new_contig_bin.tsv").iter_rows()) == [("a", "x"), ("a2", "x"), ("d", "y")]
    assert sorted(_read(tmp_path / "decontaminated_contig_bin.tsv").iter_rows()) == [("a", "x"), ("a2", "x")]
    # include_contigs.tsv: candidates dropped as redundant are not listed with a confidence level
    inc = dict(_read(tmp_path / "include_contigs.tsv").select("contig", "confidence").iter_rows())
    assert inc == {"b": "dropped_redundant", "c": "dropped_redundant", "d": "high_confidence"}
