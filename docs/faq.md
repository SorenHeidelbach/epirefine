# FAQ and troubleshooting

**Inputs:** [Which basecaller and model?](#which-basecaller-and-model-do-i-need) ·
[Must the motifs come from Nanomotif?](#must-the-motifs-come-from-nanomotif) ·
[Which binner?](#does-it-matter-which-binner-i-used) · [Gzipped pileup?](#can-the-pileup-be-gzipped)

**Results:** [Many contigs not judged](#why-did-so-many-contigs-get-no-decision) ·
[Nothing included](#why-was-nothing-included) · [Plasmids left unbinned](#why-are-some-plasmids-or-phages-left-unbinned) ·
[Close strains](#can-it-separate-close-strains) · [Bins without methylation](#what-about-bins-without-detected-methylation) ·
[Why was this contig flagged?](#why-was-this-contig-flagged) · [Should I trust a flag?](#should-i-trust-every-flag)

**Running:** [Time and memory](#how-long-does-it-take-and-how-much-memory) ·
[Rerunning with other thresholds](#can-i-rerun-with-other-thresholds-without-reading-the-pileup-again) ·
[Reproducibility](#are-results-reproducible) · [Reporting a problem](#something-went-wrong)

## Inputs

### Which basecaller and model do I need?

Any nanopore basecall with modified-base calls in the `MM`/`ML` tags, for example Dorado with a model that calls
6mA and 4mC/5mC (`modkit pileup` reports all of them). The more modification types are called, the more motifs can
tell genomes apart. EpiRefine learns the basecaller's error rates from your sample, so no model-specific settings
are needed.

### Must the motifs come from Nanomotif?

No. EpiRefine needs only the list of motifs: any TSV with the columns `motif`, `mod_type` and `mod_position` works.
Nanomotif's `bin-motifs.tsv` is the usual source. Motifs that no bin methylates are harmless; motifs missing from the
list cannot be used, so a genome whose motifs are all missing looks unmethylated.

### Does it matter which binner I used?

No. EpiRefine refines any binning given as `contig<TAB>bin`. The bins are used as examples of what each genome's
methylation looks like, so more complete bins give better fingerprints, but bins with their own contamination are
handled: their foreign pieces are down-weighted when the fingerprint is built.

### Can the pileup be gzipped?

Yes, but a gzipped pileup is read by one worker. For large pileups, unpack it first: an uncompressed pileup is read
in parallel by `--workers` processes. The pileup is read only once per set of inputs; reruns use the site cache.

## Results

### Can I run it on a single bin (an isolate or one MAG)?

Yes, but only contig inclusion is done. A contaminant is recognised by fitting other bins, or a genome without a
bin, better than its own bin, and both the contamination rate and the calibration are built on several bins. With
one bin in the contig-bin table EpiRefine says so in the log (`WARNING: ... one bin only`), keeps every binned
contig untested (`p_contaminant` is `NaN`, `best_alternative` is `unknown`, `bin_contamination.tsv` is empty) and
writes `"contamination_detection": "skipped: one bin only"` and `"contamination_rate": null` to `params.json`.
Unbinned contigs are still tested for inclusion into the bin.

### Why did so many contigs get no decision?

EpiRefine declines rather than guesses. A contig gets no decision (`low_coverage`, `too_short` or
`no_methylation_evidence`) when it has fewer than 5 reads per motif site, is shorter than 1 kb, or carries too few
informative motif sites to tell bins apart. On a complex fecal sample that was about half of the assembly, mostly
short, low-coverage contigs. Binned contigs without a decision stay in their bin.

You can lower `--min-contig-coverage`, `--min-contig-length` or `--min-evidence`, but decisions there are much less
reliable: against Hi-C, flags below 5x coverage were right only 12.5% of the time.

### Why was nothing included?

An unbinned contig is included only when one bin has probability ≥ 0.9. That fails when:

- the contig has no decision (see above; check its `status`);
- several bins methylate the same motifs, so `credible_set_95` lists more than one;
- its genome has no bin, and `p_unknown` is high: this is the right answer for contigs of unbinned genomes;
- it would be `redundant`: its sequence is already in the bin.

`include_contigs.tsv` lists the candidates with probability ≥ 0.5, which shows how close the near misses were.

### Why are some plasmids or phages left unbinned?

Most are short and at low coverage, so they get no decision. Others carry their own methyltransferase and methylate
a motif their host does not; that contradicts the host's fingerprint, so EpiRefine does not place them with their
host. Both cases are reported as not placed rather than placed wrongly.

### Can it separate close strains?

Only when they methylate different motifs. Strains with the same methyltransferases have the same fingerprint, so
methylation cannot tell their contigs apart; EpiRefine then lists both bins in `credible_set_95` and does not move or
include the contig. Bins that look alike are listed in `calibration/lookalikes.tsv`.

### What about bins without detected methylation?

A bin in which no listed motif is methylated (`no_methylation_detected` in `bin_evidence.tsv`) matches every contig
without methylation at the listed motifs, including contigs of genomes whose motifs Nanomotif never saw. Inclusions
into such bins therefore need probability ≥ 0.95 (`--include-threshold-unmethylated`).

### Why was this contig flagged?

Run with `--motif-fit` and look up the contig in `motif_fit.tsv`: it shows, per motif group, the contig's methylated
fraction with its 95% interval next to its bin's. A flag needs at least one motif group where the two are clearly
apart (`methylome_difference` ≥ 0.4 in `contig_decisions.tsv`), typically a motif the bin methylates and the
contig does not, or the reverse. `best_alternative` says which bin, or `unknown`, fits better.

### Should I trust every flag?

On benchmarks with known truth every flag was a true contaminant. On a fecal sample, 6 of 7 flags and 23 of 24
inclusions that Hi-C could check (at ≥ 10x) agreed with Hi-C. Two cases deserve a look:

- **`p_contaminant` just above 0.5**: the evidence is real but not overwhelming. Sort by `p_contaminant` to review
  the least certain flags first.
- **Very long contigs**: evidence grows with length faster than calibration fully corrects, so long contigs are
  somewhat overconfident. A long contig that is only partly foreign is reported as `mixed` instead of being flagged.

## Running

### How long does it take, and how much memory?

| Sample | Pileup | Time | Peak memory |
|---|---|---|---|
| Community of 25-35 Mb | 4-6 GB | 0.5-3 min | ≤ 2.2 GB |
| Fecal sample, 2.6 Gb assembly, 490 bins | 330 GB | 1.5-2 h | 41 GB |

With `--workers 16`. Most of the time on large samples is the first read of the pileup; a rerun from the cache is
much faster.

### Can I rerun with other thresholds without reading the pileup again?

Yes. The site cache is kept in `<out>/cache`. Point a new run at it with `--cache-dir`:

```bash
epirefine run ... --out strict --contamination-threshold 0.9 --cache-dir epirefine_out/cache
```

The cache is reused when the assembly, pileup and motifs are the same.

### Are results reproducible?

Yes. The same inputs and `--seed` (default 1) give identical outputs.

### Something went wrong

Check the log first (standard output): it starts with the version, the command line and the inputs, and each step
reports what it read and fitted. Every line begins with the seconds since the start and the peak memory of the main
process so far, e.g. `[  312.4s   2.1 GB]`; the last line also gives the peak memory of the largest worker process.
Errors are written to the same stream (`ERROR: ...`, with the traceback for an unexpected failure) and in one line to
standard error; the exit code is 2 for a problem with the inputs or options and 1 otherwise. Common causes:

- **Few contigs have sites** (`n_sites` is 0): contig names in the pileup, the assembly and
  `contig_bin.tsv` must match exactly, and the pileup must be made against the same assembly. The inputs are
  checked before the run starts its work: a missing or unreadable file, a contig-bin table that is not
  `contig<TAB>bin` or shares no contig with the assembly, a motif table with an unknown `mod_type`, and a pileup
  that is not a headerless bedMethyl table stop the run with a message naming the problem. Rows of the contig-bin
  table whose contig is not in the assembly, and invalid motifs, are dropped with a `WARNING` line in the log that
  gives their number.
- **"no pileup row lies at a motif site of an assembly contig"**: the pileup's contig names differ from the
  assembly's (the message shows one of each), or the pileup was made with other modification codes than
  `a`, `m`, `h`, `21839`.
- **"unknown mod_type"**: the motif table must use the modkit codes `a` (6mA), `m` (5mC), `21839` (4mC) or `h`
  (5hmC), as Nanomotif writes them; `mod_position` is 0-based.
- **"no calibration fragments"**: no binned contig has both enough coverage (`--min-contig-coverage`, 5) and the
  length of the shortest calibration piece (`--frag-lengths`, 2 kb). Lower one of them, or run with
  `--no-calibrate`.
- **"pileup rows of contig … are not contiguous"**: the rows of each contig must be together, as `modkit pileup`
  writes them. Sort a re-ordered pileup by contig (`sort -k1,1 -k2,2n pileup.bed > sorted.bed`).

If that doesn't help, open an [issue](https://github.com/SorenHeidelbach/epirefine/issues) with the command, the
EpiRefine version (`epirefine --version`) and the end of the log.
