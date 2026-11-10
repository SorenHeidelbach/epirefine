"""Readers for the inputs: assembly, contig->bin table, Nanomotif motifs, modkit pileup."""
from __future__ import annotations

import gzip
import io
from pathlib import Path

import polars as pl

from .motifs import Motif


def _open(path, mode="rt"):
    path = str(path)
    return gzip.open(path, mode) if path.endswith(".gz") else open(path, mode)


def iter_fasta(path):
    """(name, upper-case sequence) per record, one record in memory at a time."""
    name, parts = None, []
    with _open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(parts).upper()
                name, parts = line[1:].split()[0], []
            elif name is not None:
                parts.append(line.strip())
    if name is not None:
        yield name, "".join(parts).upper()


def read_fasta(path) -> dict[str, str]:
    """Whole assembly in memory; a repeated name keeps its first position and its last sequence."""
    return dict(iter_fasta(path))


def fasta_lengths(path) -> dict[str, int]:
    """Contig lengths in file order, without holding the sequences."""
    return {name: len(seq) for name, seq in iter_fasta(path)}


def fasta_subset(path, names) -> dict[str, str]:
    """Sequences of the named contigs, in the order of names."""
    want = set(names)
    got = {name: seq for name, seq in iter_fasta(path) if name in want}
    return {c: got[c] for c in names}


def read_contig_bin(path) -> dict[str, str]:
    """Two columns, contig and bin, no header (Nanomotif/DAS Tool style). A header line is tolerated."""
    out = {}
    with _open(path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2 or parts[0] in ("contig", "contig_id", "seq_name"):
                continue
            out[parts[0]] = parts[1]
    return out


def read_motifs(path) -> list[Motif]:
    """Read the motif set from Nanomotif's bin-motifs.tsv (or any TSV with motif, mod_type, mod_position).

    Only the *union* of motifs is used. Which bins carry which motif is re-estimated from the
    methylation data, so Nanomotif's per-bin presence calls are not trusted as ground truth.
    If `motif_complement`/`mod_position_complement` columns exist, the complement motif is added too.
    """
    df = pl.read_csv(path, separator="\t", infer_schema_length=0)
    need = {"motif", "mod_type", "mod_position"}
    if not need <= set(df.columns):
        raise ValueError(f"{path}: needs columns {sorted(need)}, found {df.columns}")
    seen, motifs = set(), []

    def add(seq, mod_type, pos):
        if seq is None or pos is None or seq in ("", "NA", "nan") or pos in ("", "NA", "nan"):
            return
        m = Motif(seq.upper(), int(float(pos)), mod_type)
        if m.key not in seen and m.is_valid():
            seen.add(m.key)
            motifs.append(m)

    has_comp = {"motif_complement", "mod_position_complement"} <= set(df.columns)
    for row in df.iter_rows(named=True):
        add(row["motif"], row["mod_type"], row["mod_position"])
        if has_comp:
            add(row["motif_complement"], row["mod_type"], row["mod_position_complement"])
    return motifs


# modkit bedMethyl: chrom start end code score strand tstart tend color Nvalid frac Nmod Ncanon ...
# Older modkit writes columns 10+ space separated, newer can write all tabs; both are handled by
# turning spaces into tabs before parsing.
_PILEUP_COLS = [0, 1, 3, 5, 9, 11]
_PILEUP_NAMES = ["contig", "pos", "mod_type", "strand", "n_valid", "n_mod"]


def _parse_pileup_block(block: bytes) -> pl.DataFrame:
    df = pl.read_csv(
        io.BytesIO(block.replace(b" ", b"\t")),
        separator="\t",
        has_header=False,
        comment_prefix="#",
        infer_schema=False,
        truncate_ragged_lines=True,
    )
    df = df.select([df.columns[i] for i in _PILEUP_COLS])
    df.columns = _PILEUP_NAMES
    return df.with_columns(
        pl.col("pos").cast(pl.Int64), pl.col("n_valid").cast(pl.Int64), pl.col("n_mod").cast(pl.Int64)
    )


def read_pileup_at_sites(path, records: pl.DataFrame, chunk_bytes: int = 256 << 20) -> pl.DataFrame:
    """Stream a modkit pileup and keep only rows at motif sites.

    `records` must have columns contig (str), pos (i64), strand (str), mod_type (str) and any extra
    annotation columns, which are carried through the inner join.
    """
    keys = ["contig", "pos", "strand", "mod_type"]
    wanted_contigs = set(records["contig"].unique().to_list())
    parts = []
    with _open(path, "rb") as fh:
        tail = b""
        while True:
            chunk = fh.read(chunk_bytes)
            if not chunk:
                block, tail = tail, b""
            else:
                chunk = tail + chunk
                cut = chunk.rfind(b"\n")
                if cut < 0:
                    tail = chunk
                    continue
                block, tail = chunk[: cut + 1], chunk[cut + 1 :]
            if block.strip():
                df = _parse_pileup_block(block).filter(pl.col("contig").is_in(wanted_contigs))
                if df.height:
                    parts.append(df.join(records, on=keys, how="inner"))
            if not chunk:
                break
    if not parts:
        return records.head(0).with_columns(pl.lit(0).alias("n_valid"), pl.lit(0).alias("n_mod"))
    return pl.concat(parts)
