import numpy as np
import polars as pl

from pbinner.motifs import Motif, can_share_site, redundancy_weights


def test_can_share_site():
    assert can_share_site(Motif("GCWGC", 1, "m"), Motif("CGCWGCR", 2, "m"))     # nested variants
    assert can_share_site(Motif("GATC", 1, "a"), Motif("RGATCY", 2, "a"))
    assert not can_share_site(Motif("GATC", 1, "a"), Motif("CCWGG", 1, "a"))     # unrelated
    assert not can_share_site(Motif("GATC", 1, "a"), Motif("GATC", 1, "m"))      # other mod type


def test_variants_of_one_mtase_share_weight_unrelated_motifs_of_one_genome_do_not():
    motifs = [Motif("GCWGC", 1, "m"), Motif("CGCWGCR", 2, "m"), Motif("CCWGG", 1, "m"), Motif("GANTC", 1, "a")]
    rows = []
    for b in range(8):
        on01 = b in (0, 1, 2)          # groups 0, 1: one MTase in bins 0-2
        on23 = b in (3, 4)             # groups 2, 3: two unrelated MTases of genomes 3-4
        for g, on in ((0, on01), (1, on01), (2, on23), (3, on23)):
            rows.append((g, b, 10, 10 if on else 0, 100))
    cb = pl.DataFrame(rows, schema={"group": pl.Int32, "bin": pl.Int32, "n": pl.Int32, "k": pl.Int32, "w": pl.Int64}, orient="row")
    w, tab = redundancy_weights(cb, motifs, [0, 1, 2, 3], 4)
    assert np.allclose(w, [0.5, 0.5, 1.0, 1.0])
