# Getting started

This guide takes you from nanopore reads to refined bins in five steps. Steps 1 and 2 make the two inputs that come
from the reads; if you already have a modkit pileup and Nanomotif's `bin-motifs.tsv`, go to step 3.

[1. Map the reads](#1-map-the-reads-with-their-methylation-tags) ·
[2. Pileup and motifs](#2-make-the-pileup-and-find-the-motifs) ·
[3. Run EpiRefine](#3-run-epirefine) ·
[4. Read the results](#4-read-the-results) ·
[5. Write the refined bins](#5-write-the-refined-bins)

## What you need

- **Nanopore reads basecalled with modified-base calling**, for example with
  [Dorado](https://github.com/nanoporetech/dorado) and a model that calls 6mA and 4mC/5mC. The calls are stored as
  `MM`/`ML` tags in the reads.
- **An assembly** of those reads (`assembly.fasta`) and **bins** from any binner, as a table `contig_bin.tsv`:
  one line per binned contig, `contig<TAB>bin`, no header. Contigs not in the table are unbinned.
- **Tools:** EpiRefine (`pip install epirefine`), [minimap2](https://github.com/lh3/minimap2) and
  [samtools](https://www.htslib.org/), [modkit](https://github.com/nanoporetech/modkit) and
  [Nanomotif](https://github.com/MicrobialDarkMatter/nanomotif).

## 1. Map the reads with their methylation tags

Map the reads to the assembly and keep the `MM`/`ML` tags (`samtools fastq -T` copies them into the read headers,
`minimap2 -y` copies them back into the alignments):

```bash
samtools fastq -T MM,ML reads.bam \
  | minimap2 -ax map-ont -y -t 16 assembly.fasta - \
  | samtools sort -@ 8 -o aligned.bam -
samtools index aligned.bam
```

## 2. Make the pileup and find the motifs

`modkit pileup` counts, at every position, how many reads were called modified and unmodified:

```bash
modkit pileup aligned.bam pileup.bed --threads 16
```

Nanomotif finds the motifs methylated in each bin:

```bash
nanomotif motif_discovery assembly.fasta pileup.bed -c contig_bin.tsv --out nanomotif -t 16
```

This writes `nanomotif/bin-motifs.tsv`. EpiRefine uses only the list of motifs from it; which bins methylate which
motif it measures again from the reads.

## 3. Run EpiRefine

```bash
epirefine run \
    --assembly assembly.fasta \
    --contig-bin contig_bin.tsv \
    --pileup pileup.bed \
    --motifs nanomotif/bin-motifs.tsv \
    --out epirefine_out \
    --workers 16
```

What happens, in order (the log shows each step):

1. The pileup is read once and reduced to the motif sites (a 330 GB pileup became a 2.7 GB site cache). The cache
   is kept in `epirefine_out/cache`; a rerun with the same inputs skips this step.
2. EpiRefine learns, per motif, how the basecaller behaves at methylated and unmethylated sites.
3. It builds a methylation fingerprint for every bin, robust to the bin's own contaminants.
4. It calibrates its probabilities on pieces cut from your own bins.
5. It scores every contig against every bin and against "a genome without a bin", and decides.

**Time and memory.** A community of 25-35 Mb with a 4-6 GB pileup takes 0.5-3 minutes and at most 2.2 GB. A
complex fecal sample (2.6 Gb assembly, 490 bins, 330 GB pileup) took about 1.5-2 hours and 41 GB with
`--workers 16`. Most of the time on large samples is the first read of the pileup.

## 4. Read the results

Start with **`epirefine_out/contig_decisions.tsv`**, one row per contig. The columns to look at first:

| Column | Meaning |
|---|---|
| `status` | what EpiRefine decided (table below) |
| `given_bin`, `new_bin` | the bin before and after refinement (empty = unbinned) |
| `p_contaminant` | for a binned contig: probability that it does not belong to its bin |
| `best_alternative`, `p_best_alternative` | the most likely other home and its probability (`unknown` = a genome without a bin) |
| `coverage` | median reads per motif site; below 5 no decision is made |

| Status | Meaning | In the refined bins? |
|---|---|---|
| `kept` | methylation fits its bin | yes, in its bin |
| `contaminant` | methylation contradicts its bin | no |
| `moved` | contaminant that clearly fits another bin | yes, in the new bin |
| `mixed` | long contig, partly its bin's and partly another genome's (`foreign_segments` says where) | yes, in its bin |
| `included` | unbinned; fits one bin | yes, in that bin |
| `unbinned` | unbinned; no bin fits well enough | no |
| `redundant` | unbinned; would be included, but its sequence is already in the bin | no |
| `low_coverage`, `too_short`, `no_methylation_evidence` | not judged | unchanged |
| `sole_contig_of_bin` | the only contig of its bin; nothing to compare it with | unchanged |

A quick summary of what happened:

```bash
cut -f5 epirefine_out/contig_decisions.tsv | sort | uniq -c
```

The flagged contigs, most certain first (columns: contig, length, given bin, status, new bin, p_contaminant):

```bash
awk -F'\t' '$5=="contaminant" || $5=="moved"' epirefine_out/contig_decisions.tsv \
  | sort -t$'\t' -k7,7gr | cut -f1,2,4,5,6,7 | head
```

All files and columns: [outputs.md](outputs.md).

## 5. Write the refined bins

`epirefine_out/new_contig_bin.tsv` is the refined binning (`contig<TAB>bin`, with a header). To write one FASTA file
per bin:

```bash
mkdir -p refined_bins
awk 'NR==FNR { if (FNR > 1) bin[$1] = $2; next }
     /^>/    { name = substr($1, 2); out = (name in bin) ? "refined_bins/" bin[name] ".fa" : "" }
     out     { print > out }' epirefine_out/new_contig_bin.tsv assembly.fasta
```

Contigs that were only flagged are not in any bin; `decontaminated_contig_bin.tsv` has the bins with contaminants
removed but nothing added, if you want to judge the two steps separately (for example with CheckM2).

## Next

- Change what EpiRefine decides, how strictly, or how fast: [usage.md](usage.md).
- Understand why a contig was flagged or included: [how-it-works.md](how-it-works.md).
- Something looks odd: [faq.md](faq.md).
