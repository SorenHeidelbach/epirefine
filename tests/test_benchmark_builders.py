"""The monoculture set builder must cut an isolate the same way in every process (referee 2026-11-10, 6.13: the
seed was the salted built-in hash() of the isolate name, so the cuts depended on PYTHONHASHSEED)."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parents[1] / "benchmarks"

SCRIPT = """
import sys
from pathlib import Path
sys.path.insert(0, {bench!r})
from monoculture_sets import cut_isolate
row = {{"isolate": "Synthetica-testii", "assembly": {asm!r}, "pileup": {pile!r}, "plasmids": ""}}
cache = Path({out!r}); cache.mkdir(parents=True, exist_ok=True)
pq, pieces = cut_isolate(row, cache, 2000, 20000)
print(pieces.select("contig", "start", "end").write_csv(separator="\\t"), end="")
"""


def _pileup_row(contig, pos):
    # modkit bedMethyl, 18 columns: chrom start end code score strand tstart tend color Nvalid frac Nmod Ncanon ...
    return "\t".join(map(str, [contig, pos, pos + 1, "a", 10, "+", pos, pos + 1, "255,0,0", 10, 50.0,
                               5, 5, 0, 0, 0, 0, 0]))


def _cut(tmp_path, hashseed, name):
    out = tmp_path / name
    code = SCRIPT.format(bench=str(BENCH), asm=str(tmp_path / "asm.fasta"), pile=str(tmp_path / "pileup.bed"),
                         out=str(out))
    env = {**os.environ, "PYTHONHASHSEED": str(hashseed)}
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert p.returncode == 0, p.stderr
    return p.stdout


def test_isolate_cuts_do_not_depend_on_the_hash_salt(tmp_path):
    seqs = {"contig_1": "ACGT" * 50_000, "contig_2": "GATC" * 7_500}
    (tmp_path / "asm.fasta").write_text("".join(f">{c}\n{s}\n" for c, s in seqs.items()))
    rows = [_pileup_row(c, pos) for c, s in seqs.items() for pos in range(0, len(s), 997)]
    (tmp_path / "pileup.bed").write_text("\n".join(rows) + "\n")
    first = _cut(tmp_path, 1, "a")
    assert first.count("\n") > 5  # contig_1 (200 kb) is cut into several pieces of at most 20 kb
    for seed in (2, 3, 12345):
        assert _cut(tmp_path, seed, f"s{seed}") == first


def test_isolate_seed_is_fixed():
    sys.path.insert(0, str(BENCH))
    from monoculture_sets import isolate_seed

    # SHA-256 based: a fixed value, so a set built today can be rebuilt from code
    assert isolate_seed("Synthetica-testii") == isolate_seed("Synthetica-testii")
    assert 0 <= isolate_seed("x") < 2**32
    import hashlib
    assert isolate_seed("x") == int.from_bytes(hashlib.sha256(b"x").digest()[:4], "little")
