# Getting started

This guide takes you from nanopore reads to refined bins in five steps. Steps 1 and 2 make the two inputs that come
from the reads; if you already have a modkit pileup and Nanomotif's `bin-motifs.tsv`, go to step 3.

[Try the example data](#try-it-on-the-example-data) ·
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

## Try it on the example data

The repository (and the source distribution) holds a small example in `tests/data/example_community`: a
**synthetic** community made by the test suite's generator (`tests/synth.py`, seed 1), not real reads. Five
simulated genomes (1.1 Mb in 56 contigs); four of them have bins (`binA`, `binB`, `binC`, and `binE` of a single
contig) and one has none. Each genome methylates its own combination of the 12 motifs in `bin-motifs.tsv` (the
binless genome also methylates a motif that no bin has, so the table lacks it, as Nanomotif's would), and the pileup
holds simulated read counts at the motif sites only. The binning has planted errors: four contigs put in the bin of
another binned genome, one contig of the binless genome put in a bin, six members of `binA`, `binB` and `binC` left
unbinned, an unbinned exact copy of a binned contig and a second copy of it in a wrong bin, an unbinned contig below
1 kb and a binned one at about 2x coverage. The four files take 1.6 MB.

```bash
git clone https://github.com/SorenHeidelbach/epirefine.git && cd epirefine
epirefine run \
    --assembly tests/data/example_community/assembly.fasta.gz \
    --pileup tests/data/example_community/pileup.bed.gz \
    --contig-bin tests/data/example_community/contig_bin.tsv \
    --motifs tests/data/example_community/bin-motifs.tsv \
    --out example_out --workers 1
```

It takes a few seconds on one core (about 5 s and 0.2 GB on a development server). Near its end the log reads

```
refinement: p_unknown=0.417, contamination rate=0.102; 6 of 42 testable binned contigs flagged, 6 of 13 unbinned included, 4 contaminants moved to another bin
```

and `cut -f5 example_out/contig_decisions.tsv | sort | uniq -c` should count 35 `kept`, 6 `included`, 4 `moved`,
2 `contaminant`, 5 `unbinned`, 1 each of `redundant`, `low_coverage`, `too_short` and `sole_contig_of_bin`. The
six flagged contigs are the six planted ones: four are moved back to their own genome's bin, and two stay out of
every bin (the contig of the binless genome, and the misplaced copy, whose way back is closed because its sequence
is already in the right bin). The six missing members are included, the unbinned copy is `redundant`, and the
binless genome's five unbinned contigs stay `unbinned`. A log line `WARNING: 1 bins show two methylation states`
about `binA` is expected: the planted contaminants make it look like two genomes. `tests/test_example_data.py`
checks this run. Being synthetic, these data are easy: they show the files and the workflow, not the accuracy to
expect on real samples.

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

1. The pileup is read once and reduced to the motif sites (a 330 GB pileup became 2.7 GB of motif sites; with an encoded copy of the assembly the cache was 5.1 GB). The cache
   is kept in `epirefine_out/cache`; a rerun with the same inputs skips this step.
2. EpiRefine learns, per motif, how the basecaller behaves at methylated and unmethylated sites.
3. It builds a methylation fingerprint for every bin, robust to the bin's own contaminants.
4. It calibrates its probabilities on pieces cut from your own bins.
5. It scores every contig against every bin and against "a genome without a bin", and decides.

**Time and memory.** A community of 25-35 Mb with a 4-6 GB pileup takes a few minutes at most and about 1-2 GB.
For a complex fecal sample (2.6 Gb assembly, 490 bins, 330 GB pileup), the first read of the pileup took 2.4 hours
with 2 workers (0.10.1) and everything after it 13.5 minutes with 12 workers (0.11.0), with 25.5 GB in the main
process plus the workers. Measurements per version are in the [FAQ](faq.md#how-long-does-it-take-and-how-much-memory).

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
