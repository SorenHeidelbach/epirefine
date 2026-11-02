import polars as pl

from epirefine.motifs import noisy_groups


def test_noise_is_intermediate_sites_in_many_genomes(tmp_path):
    rows = []
    for c in range(30):  # 30 contigs in 6 bins of 5
        for pos in range(40):
            rows.append((c, 0, pos, 10, 3))                              # 30% of reads at every site, every genome: noise
            rows.append((c, 1, pos, 10, 10 if c % 2 else 0))             # on or off by contig: real
            rows.append((c, 2, pos, 10, 3 if c < 5 else 0))              # intermediate in one genome only: kept
            rows.append((c, 3, pos, 10, 10 if pos % 2 else 0))           # GATC in an RGATCY genome: half the sites
                                                                         # fully methylated, half not: kept
    pl.DataFrame(rows, schema={"contig": pl.Int32, "group": pl.Int32, "pos": pl.Int64, "n": pl.Int32, "k": pl.Int32},
                 orient="row").write_parquet(tmp_path / "p.parquet")
    cb = pl.DataFrame({"contig": list(range(30)), "bin": [c // 5 for c in range(30)]}, schema={"contig": pl.Int32, "bin": pl.Int32})
    g = noisy_groups([str(tmp_path / "p.parquet")], cb, list(range(30)))
    assert dict(g.select("group", "noisy").iter_rows()) == {0: True, 1: False, 2: False, 3: False}
