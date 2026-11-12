# FAQ and troubleshooting

**Inputs:** [Which basecaller and model?](#which-basecaller-and-model-do-i-need) ·
[Must the motifs come from Nanomotif?](#must-the-motifs-come-from-nanomotif) ·
[Which binner?](#does-it-matter-which-binner-i-used) · [Gzipped pileup?](#can-the-pileup-be-gzipped)

**Results:** [Many contigs not judged](#why-did-so-many-contigs-get-no-decision) ·
[Nothing included](#why-was-nothing-included) · [Plasmids and phages](#why-are-some-plasmids-or-phages-left-unbinned-or-flagged) ·
[Close strains](#can-it-separate-close-strains) · [Bins without methylation](#what-about-bins-without-detected-methylation) ·
[Why was this contig flagged?](#why-was-this-contig-flagged) · [Should I trust a flag?](#should-i-trust-every-flag) ·
[Known limitations](#known-limitations)

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
Nanomotif's `bin-motifs.tsv` is the usual source. Motifs missing from the list cannot be used, so a genome whose
motifs are all missing looks unmethylated. Extra motifs that no bin methylates are not harmless: on the development
benchmarks, adding motifs from another sample's motif table left flag precision at 99.5% or more but lowered the
share of removed contigs put back on the training communities from 89.3% to 77.1% (100 added motifs), 40.4% (400) and
6.7% (all 715), because the calibration then chose a model that keeps unbinned contigs out of the bins (see
[known limitations](#known-limitations)).

### Does it matter which binner I used?

No. EpiRefine refines any binning given as `contig<TAB>bin`. The bins are used as examples of what each genome's
methylation looks like, so more complete bins give better fingerprints. A bin's own contaminants are down-weighted
when its fingerprint is built, which works when they differ from the bin in several motifs. A bin that is a merge of
two genomes differing in a single motif is not handled: see [known limitations](#known-limitations).

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
informative motif sites to tell bins apart. On a complex fecal sample (ZymoFecal, a development sample) that was
76% of the contigs and 31% of the assembled sequence, mostly short, low-coverage contigs. Binned contigs without a
decision stay in their bin.

You can lower `--min-contig-coverage`, `--min-contig-length` or `--min-evidence`, but decisions there are much less
reliable: on that sample, Hi-C confirmed about a quarter of the flags below 5x coverage (12 of 46, 26%, 95% interval
16-40%, under the current Hi-C reference; 4 of 41 under an earlier one), and most flags there have no Hi-C verdict.

### Why was nothing included?

An unbinned contig is included only when one bin has probability ≥ 0.9. That fails when:

- the contig has no decision (see above; check its `status`);
- several bins methylate the same motifs, so `credible_set_95` lists more than one;
- its genome has no bin, so "no bin" is the most probable answer (`p_unknown` high). That is expected for contigs of
  unbinned genomes, but when `p_unknown` is near 1 and nothing at all is included, see
  [known limitations](#known-limitations);
- it would be `redundant`: its sequence is already in the bin.

`include_contigs.tsv` lists the candidates with probability ≥ 0.5, which shows how close the near misses were.

### Why are some plasmids or phages left unbinned, or flagged?

Most are short and at low coverage, so they get no decision: on a fecal development sample (ZymoFecal) half of the
unbinned plasmids and 63% of the unbinned viruses were below 5x, and only 5 of 116 binned plasmids and 3 of 67 binned
viruses got a decision. Most elements with enough coverage share their host's methylation, but some do not:
high-copy plasmids can be partly unmethylated at a motif their host methylates, and phages can bring their own
methyltransferases (on that sample a 190 kb phage carrying three methyltransferase genes had four motifs its host
lacks). An unbinned element like that is left unbinned rather than placed with its host. A binned one can be flagged
as a contaminant even though it lives in the bin's cells: 4 of the 8 binned plasmids and viruses with a decision on
that sample were flagged, and Hi-C could check none of them. Flags on contigs annotated as plasmid or virus (for
example by geNomad) deserve a second look.

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

Most flags are right, but not all. On held-out benchmark communities with known truth (monoculture isolates not used
during development), 311 of 316 flags (98.4%) were true contaminants; on the development benchmarks every flag was.
On a fecal sample with Hi-C (ZymoFecal, version 0.8.0 with a Nanomotif 1.1.3 motif table), 7 of 8 flags (88%, 95%
interval 53-98%) and 23 of 24 inclusions (96%, 80-99%) that Hi-C could check at ≥ 10x agreed with it; with the
Nanomotif 1.3.0 table and version 0.11.1 it is 7 of 8 flags and 24 of 26 inclusions. Hi-C could check only these 8 of
the 139 flagged contigs and 24 of the 485 inclusions, the flag figure moves between 67% and 88% with the version of
the Hi-C reference (the inclusion figure stays at 92-100%), and the same sample was used to choose several defaults (the 0.4 methylome difference, the 0.9
inclusion threshold, the 5x cut-off), so it is not an independent check. Things to keep in mind:

- **A high `p_contaminant` is no guarantee**: the 5 wrong flags on the held-out communities all had `p_contaminant`
  ≥ 0.9999. Sorting by `p_contaminant` shows the least certain flags first, but wrong flags are not only among them.
- **Very long contigs**: evidence grows with length faster than one temperature corrects, so long contigs are
  overconfident (on the fecal sample, the temperature that would fit each piece length alone rose from 2.7 at 2 kb
  to 14.7 at 500 kb). A long contig that is only partly foreign is reported as `mixed` instead of being flagged.
- **Plasmids and phages** in their host's bin can be flagged (see [above](#why-are-some-plasmids-or-phages-left-unbinned-or-flagged)).

## Known limitations

- **With some motif tables nothing is included.** EpiRefine describes a genome without a bin either as a draw from
  the whole community or as a relative of the binned genomes, and chooses between the two during calibration. With
  some motif tables it chooses the relatives, which then absorb every unbinned contig: the estimated share of
  unbinned contigs from genomes without a bin (`p_unknown`) becomes 1, and no contig is included or moved; flags
  are still made. On a fecal development sample this happened with a Nanomotif 1.3.0 motif table but not with a
  1.1.3 table of the same bins (0 against 485 inclusions). It shows in the log as

  ```
  unknown model: relatives (eta=0.05, lam=0.9) vs community only: log-loss gain 0.1418 +- 0.0459 -> relatives
  refinement: p_unknown=1.000, contamination rate=0.302; 150 of 9129 testable binned contigs flagged, 0 of 25600 unbinned included, 0 contaminants moved to another bin
  ```

  and in `params.json` as `p_unknown` near 1 with `unknown_lam` below 1. A run in this state has no usable
  inclusion result. Since version 0.11.1 the relatives model is kept only when its gain is significant over bins, not
over single calibration pieces, and the sample above then chooses the community-only model (about 500 inclusions).
The choice still rests on a narrow margin there (0.9-1.9 standard errors over bins against a limit of 2, on five
sets of calibration pieces), so check these two log lines on every sample; `--calibration legacy` gives the earlier rule. See the
  [changelog](https://github.com/SorenHeidelbach/epirefine/blob/main/CHANGELOG.md).
- **Bins merged from two genomes that differ in one motif are not detected.** On development sets built by merging
  two isolates' bins, a second genome that differed in several motifs had all its contigs flagged up to at least 40%
  of the bin; one that differed in a single motif was mostly missed (at most half of its contigs flagged at any
  share, and 0-1% once it held 40% or more of the bin).
  `bin_evidence.tsv` marks such bins in its `warning` column (`two_methylation_states`).
- **Real-data accuracy is known from one development sample only.** The Hi-C comparison above is on a sample that
  was used to choose defaults; no real sample held out from development has been evaluated yet.

## Running

### How long does it take, and how much memory?

| Sample | Pileup | Time | Peak memory |
|---|---|---|---|
| Benchmark community of 25-35 Mb | 4-6 GB | 0.5-3 min (0.8.0); about 16 s from the site cache (0.11.0, 4 workers) | ≤ 2.2 GB (0.8.0); 0.6-1.0 GB (0.11.0) |
| Fecal sample, 2.6 Gb assembly, 490 bins | 330 GB | first read of the pileup: 2.4 h with 2 workers (0.10.1); everything after it: 13.5 min with 12 workers (0.11.0, from the site cache) | 25.5 GB in the main process (0.11.0), plus the workers |

Memory is that of the largest single process; the worker processes come on top (an early version needed up to 54 GB
in total on the fecal sample with 8 workers). Most of the time on large samples is the first read of the pileup; a
rerun from the cache is much faster.

### Can I rerun with other thresholds without reading the pileup again?

Yes. The site cache is kept in `<out>/cache`. Point a new run at it with `--cache-dir`:

```bash
epirefine run ... --out strict --contamination-threshold 0.9 --cache-dir epirefine_out/cache
```

The cache is reused when the assembly, pileup and motifs are the same (and the motif-grouping options; see
`docs/usage.md`). `epirefine cache check` tells beforehand whether a run would reuse it.

### Are results reproducible?

Yes. The same inputs and seeds (`--seed` and `--calibration-seed`, both 1 by default) give identical outputs, also
with another number of workers (`--workers`) or threads (cores, `OMP_NUM_THREADS`), with the same installation on the
same kind of machine. `--calibration-seed` draws other calibration pieces and leaves the motif groups and the site
cache alone, so running a few values of it shows which decisions depend on the pieces drawn.

Two things can change the last digits of the probabilities without changing the decisions: the optional compiled
kernels (package `epirefine-rs`) against the numpy code, and another machine or BLAS library. Both change only how
floating-point sums and exponentials are rounded. On the development benchmarks (monoculture training communities,
ZymoOral pieces, simulated look-alikes) runs with and without the kernels made the same calls, with probabilities
that agreed to 1e-10 or better.

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
