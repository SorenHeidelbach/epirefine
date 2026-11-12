"""Wrong inputs stop the run early with a message that names the problem."""
import pytest

from small_community import run_community, write_community
from epirefine import io as pio
from epirefine import runinfo
from epirefine.io import InputError


def _motif_table(path, rows):
    path.write_text("motif\tmod_type\tmod_position\n" + "".join("\t".join(map(str, r)) + "\n" for r in rows))
    return path


def test_contig_bin_with_crlf_line_ends(tmp_path):
    p = tmp_path / "cb.tsv"
    p.write_bytes(b"c1\tbinA\r\nc2\tbinB\r\n\r\n")
    assert pio.read_contig_bin(p) == {"c1": "binA", "c2": "binB"}


@pytest.mark.parametrize("text", ["c1 binA\nc2 binB\n", "c1,binA\nc2,binB\n"])
def test_contig_bin_not_tab_separated_is_an_error(tmp_path, text):
    p = tmp_path / "cb.tsv"
    p.write_text(text)
    with pytest.raises(InputError, match="tab-separated"):
        pio.read_contig_bin(p)


def test_contig_bin_malformed_rows_are_counted(tmp_path):
    p = tmp_path / "cb.tsv"
    p.write_text("c1\tbinA\nc2 binB\nc3\tbinA\n")
    msgs = []
    assert pio.read_contig_bin(p, log=msgs.append) == {"c1": "binA", "c3": "binA"}
    assert any("1 of 3 rows" in m for m in msgs)


def test_contig_bin_must_share_contigs_with_the_assembly():
    lengths = {"c1": 10, "c2": 10, "c3": 10}
    with pytest.raises(InputError, match="columns are swapped"):
        runinfo.match_contig_bin({"binA": "c1", "binB": "c2"}, lengths, "cb.tsv")  # columns swapped
    with pytest.raises(InputError, match="none of the 2 contigs"):
        runinfo.match_contig_bin({"y1": "binA", "y2": "binB"}, lengths, "cb.tsv")
    msgs = []
    kept = runinfo.match_contig_bin({"c1": "binA", "x1": "binA", "x2": "binB", "x3": "binB"}, lengths, "cb.tsv",
                                    log=msgs.append)
    assert kept == {"c1": "binA"}
    assert len(msgs) == 2 and "1 bins have none of their contigs in the assembly" in msgs[0] and "binB" in msgs[0]
    assert "3 of 4 contigs are not in the assembly" in msgs[1] and "1 of 2 bins" in msgs[1]
    assert "fewer than half" in msgs[1]


def test_contig_in_two_bins_is_an_error(tmp_path):
    p = tmp_path / "cb.tsv"
    p.write_text("c1\tbinA\nc2\tbinA\nc1\tbinB\nc2\tbinA\n")  # c2 twice in the same bin is harmless
    with pytest.raises(InputError, match="1 contigs are listed in more than one bin .e.g. c1 .binA, binB.."):
        pio.read_contig_bin(p)


def test_unknown_mod_type_is_rejected_with_the_accepted_ones(tmp_path):
    p = _motif_table(tmp_path / "m.tsv", [("GATC", "6mA", 1), ("CCWGG", "m", 1)])
    with pytest.raises(InputError) as e:
        pio.read_motifs(p)
    assert "'6mA'" in str(e.value) and "a, m, h, 21839" in str(e.value)


def test_invalid_motif_rows_are_counted_and_logged(tmp_path):
    # GATC with a 1-based position points at T; G.TC is not IUPAC; CCWGG is given twice
    p = _motif_table(tmp_path / "m.tsv", [("GATC", "a", 2), ("G.TC", "a", 1), ("CCWGG", "m", 1), ("CCWGG", "m", 1),
                                          ("GATC", "a", 1)])
    msgs = []
    motifs = pio.read_motifs(p, log=msgs.append)
    assert [m.key for m in motifs] == ["CCWGG_m-1", "GATC_a-1"]
    assert len(msgs) == 1 and "2 of 5 motif entries are invalid" in msgs[0] and "GATC_a-2" in msgs[0]


def test_motif_table_without_a_valid_motif(tmp_path):
    with pytest.raises(InputError, match="no valid motif"):
        pio.read_motifs(_motif_table(tmp_path / "m.tsv", [("GATC", "a", 2)]))
    with pytest.raises(InputError, match="no valid motif"):
        pio.read_motifs(_motif_table(tmp_path / "m.tsv", []))


def test_missing_input_is_named_before_the_assembly_is_read(small_community, tmp_path, monkeypatch):
    def no_read(*a, **k):
        raise AssertionError("the assembly was read before the inputs were checked")
    monkeypatch.setattr(pio, "fasta_lengths", no_read)
    monkeypatch.setattr(pio, "iter_fasta", no_read)
    with pytest.raises(InputError, match="pileup not found"):
        run_community({**small_community, "pileup": tmp_path / "missing.bed"}, tmp_path / "out")
    bad = _motif_table(tmp_path / "m.tsv", [("GATC", "6mA", 1)])
    with pytest.raises(InputError, match="mod_type"):
        run_community({**small_community, "motifs": bad}, tmp_path / "out")
    header = tmp_path / "header.bed"
    header.write_text("chrom\tstart\tend\tcode\tscore\tstrand\ts\te\tcolor\tNvalid\tfrac\tNmod\n"
                      + small_community["pileup"].read_text()[:2000])
    with pytest.raises(InputError, match="header"):
        run_community({**small_community, "pileup": header}, tmp_path / "out")
    gz = tmp_path / "plain.bed.gz"
    gz.write_text(small_community["pileup"].read_text()[:2000])
    with pytest.raises(InputError, match="not gzip-compressed"):
        run_community({**small_community, "pileup": gz}, tmp_path / "out")


def test_pileup_of_another_assembly_is_a_clear_error(small_community, tmp_path):
    other = tmp_path / "other.bed"
    other.write_text("".join("chr_" + ln for ln in small_community["pileup"].read_text().splitlines(keepends=True)))
    with pytest.raises(InputError, match="contig names do not match"):
        run_community({**small_community, "pileup": other}, tmp_path / "out")


def test_no_calibration_fragments_names_the_requirement(small_community, tmp_path):
    with pytest.raises(InputError) as e:  # no contig reaches this coverage
        run_community(small_community, tmp_path / "out", min_contig_coverage=1000.0)
    assert "--min-contig-coverage" in str(e.value) and "--frag-lengths" in str(e.value)
    with pytest.raises(InputError, match="0 of those that length"):  # no contig is this long
        run_community(small_community, tmp_path / "out2", frag_lengths=(100_000,))
