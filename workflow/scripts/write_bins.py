"""Write one FASTA per bin from an assembly and a contig -> bin table.

The table may have a header (contig, bin: Nanomotif / epirefine new_contig_bin.tsv) or not (binner
output). With --lengths-only, write a table of contig lengths instead.
"""
import argparse
import os


def read_fasta(path):
    name, seq = None, []
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(seq)
                name, seq = line[1:].split()[0], []
            else:
                seq.append(line.strip())
    if name is not None:
        yield name, "".join(seq)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assembly", required=True)
    ap.add_argument("--contig-bin")
    ap.add_argument("--out")
    ap.add_argument("--min-length", type=int, default=0, help="skip bins shorter than this (bp)")
    ap.add_argument("--lengths-only")
    a = ap.parse_args()
    if a.lengths_only:
        with open(a.lengths_only, "w") as fh:
            fh.write("contig\tlength\n")
            for name, seq in read_fasta(a.assembly):
                fh.write(f"{name}\t{len(seq)}\n")
        return
    bin_of = {}
    with open(a.contig_bin) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2 or parts[0] in ("contig", "contig_id", "seq_name"):
                continue
            bin_of[parts[0]] = parts[1]
    seqs = {}
    for name, seq in read_fasta(a.assembly):
        if name in bin_of:
            seqs.setdefault(bin_of[name], []).append((name, seq))
    os.makedirs(a.out, exist_ok=True)
    n = 0
    for b, members in sorted(seqs.items()):
        if sum(len(s) for _, s in members) < a.min_length:
            continue
        with open(os.path.join(a.out, f"{b}.fa"), "w") as fh:
            for name, seq in members:
                fh.write(f">{name}\n")
                for i in range(0, len(seq), 80):
                    fh.write(seq[i:i + 80] + "\n")
        n += 1
    missing = set(bin_of) - {c for m in seqs.values() for c, _ in m}
    print(f"{n} bins written to {a.out}; {len(missing)} contigs in the table not found in the assembly")


if __name__ == "__main__":
    main()
