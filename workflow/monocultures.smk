"""Monoculture inputs and benchmark sets: modkit pileups and Nanomotif motifs where missing, then the
train / test / validation communities of benchmarks/monoculture_sets.py.

Every isolate folder under `root` with a Flye assembly (flye/assembly.fasta) and a modification BAM
(modcalls.aln.sorted.bam) is used. Existing outputs are reused: <isolate>/modkit.pileup.bed and
<isolate>/nanomotif_1.1.2/bin-motifs.tsv. Where they're missing, this runs `modkit pileup` on the BAM
and `nanomotif motif_discovery` with the whole isolate as one bin (settings as for the community
samples). Results for generated inputs go to <out>/inputs/<isolate>/.

    snakemake -s workflow/monocultures.smk --configfile workflow/config_monocultures.yaml --use-conda --cores 16
"""
from pathlib import Path

REPO = Path(workflow.basedir).parent
ROOT = Path(config["root"])
OUT = Path(config["out"])
ISO = sorted(d.name for d in ROOT.iterdir()
             if d.is_dir() and "-" in d.name and not d.name.startswith("_")
             and (d / "flye" / "assembly.fasta").exists() and (d / "modcalls.aln.sorted.bam").exists())


def asm(iso):
    return str(ROOT / iso / "flye" / "assembly.fasta")


def pileup_of(wc):
    have = ROOT / wc.iso / "modkit.pileup.bed"
    return str(have) if have.exists() else str(OUT / "inputs" / wc.iso / "modkit.pileup.bed")


def motifs_of(wc):
    have = ROOT / wc.iso / "nanomotif_1.1.2" / "bin-motifs.tsv"
    return str(have) if have.exists() else str(OUT / "inputs" / wc.iso / "nanomotif" / "bin-motifs.tsv")


rule all:
    input:
        config.get("suite", str(REPO / "benchmarks" / "suite_monocultures.json")),


rule modkit_pileup:
    input:
        bam=lambda wc: str(ROOT / wc.iso / "modcalls.aln.sorted.bam"),
    output:
        OUT / "inputs" / "{iso}" / "modkit.pileup.bed",
    log:
        OUT / "logs" / "modkit_{iso}.log",
    threads: 4
    conda:
        "envs/modkit.yaml"
    shell:
        "modkit pileup {input.bam} {output} --threads {threads} --log-filepath {log}"


rule contig_bin:
    input:
        lambda wc: asm(wc.iso),
    output:
        OUT / "inputs" / "{iso}" / "contig_bin.tsv",
    shell:
        # the whole isolate is one bin
        "grep '>' {input} | sed 's/^>//; s/ .*//' | awk -v b={wildcards.iso} '{{print $1\"\\t\"b}}' > {output}"


rule nanomotif:
    input:
        assembly=lambda wc: asm(wc.iso), pileup=pileup_of, contig_bin=OUT / "inputs" / "{iso}" / "contig_bin.tsv",
    output:
        OUT / "inputs" / "{iso}" / "nanomotif" / "bin-motifs.tsv",
    log:
        OUT / "logs" / "nanomotif_{iso}.log",
    threads: 4
    conda:
        "envs/nanomotif.yaml"
    shell:
        # thresholds as in the community runs (nanomotif 1.1.x defaults)
        "nanomotif motif_discovery {input.assembly} {input.pileup} -c {input.contig_bin} "
        "--out $(dirname {output}) -t {threads} --seed 1 > {log} 2>&1"


rule isolate_table:
    input:
        pileups=lambda wc: [pileup_of(type("W", (), {"iso": i})) for i in ISO],
        motifs=lambda wc: [motifs_of(type("W", (), {"iso": i})) for i in ISO],
    output:
        OUT / "isolates_inputs.tsv",
    run:
        rows = ["isolate\tassembly\tpileup\tmotifs\tplasmids"]
        for i, p, m in zip(ISO, input.pileups, input.motifs):
            g = next(iter((ROOT / i).glob("genomad/*_summary/*_plasmid_summary.tsv")), "")
            rows.append(f"{i}\t{asm(i)}\t{p}\t{m}\t{g}")
        Path(output[0]).write_text("\n".join(rows) + "\n")


rule build_sets:
    input:
        OUT / "isolates_inputs.tsv",
    output:
        config.get("suite", str(REPO / "benchmarks" / "suite_monocultures.json")),
    log:
        OUT / "logs" / "build_sets.log",
    params:
        python=str(REPO / ".venv" / "bin" / "python"),
        extra=config.get("build_args", ""),
    shell:
        "{params.python} {REPO}/benchmarks/monoculture_sets.py --isolates {input} --out {OUT} "
        "--suite {output} {params.extra} > {log} 2>&1"
