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
Nanomotif's `bin-motifs.tsv` is the usual source; the benchmarks of 0.12 used tables from Nanomotif 1.3.0, and
another motif table can change the results a lot. Motifs missing from the list cannot be used, so a genome whose
motifs are all missing looks unmethylated. Extra motifs that no bin methylates are not harmless: on the development
benchmarks, adding motifs from another sample's motif table left flag precision at 99.5% or more but lowered the
share of removed contigs put back on the training communities from 89.3% to 77.1% (100 added motifs), 40.4% (400) and
6.7% (all 715), because the calibration then chose a model that keeps unbinned contigs out of the bins. That was
measured with 0.11; since 0.12 that model choice is removed (see [known limitations](#known-limitations)), and the
effect of extra motifs has not been measured again.

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
informative motif sites to tell bins apart. On a complex fecal sample (ZymoFecal, a development sample, version 0.8.0) that was
76% of the contigs and 31% of the assembled sequence, mostly short, low-coverage contigs. Binned contigs without a
decision stay in their bin: a no call never changes a bin. On samples with an unbinned genome that shares a bin's methylation, the share of
`no_methylation_evidence` can be much larger (see [known limitations](#known-limitations)).

You can lower `--min-contig-coverage`, `--min-contig-length` or `--min-evidence`, but decisions there are much less
reliable: on that sample, Hi-C confirmed about a quarter of the flags below 5x coverage (12 of 46, 26%, 95% interval
16-40%, under the current Hi-C reference; 4 of 41 under an earlier one), and most flags there have no Hi-C verdict.

### Why was nothing included?

An unbinned contig is included only when one bin has probability ≥ 0.9. That fails when:

- the contig has no decision (see above; check its `status`);
- several bins methylate the same motifs, so `credible_set_95` lists more than one;
- its genome has no bin, so "no bin" is the most probable answer (`p_unknown` high). That is expected for contigs of
  unbinned genomes. (Up to 0.11.1 a "relatives" model could push `p_unknown` to 1 so that nothing at all was
  included; since 0.12 that model is no longer used, see [known limitations](#known-limitations));
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

Most flags are right, but not all. On the monoculture test and validation communities with known truth (spent: they
were looked at during development, so they are no longer an unseen evaluation; version 0.10.1), 311 of 316 flags
(98.4%) were true contaminants; on the development benchmarks every flag was.
On a fecal sample with Hi-C (ZymoFecal, version 0.8.0 with a Nanomotif 1.1.3 motif table), 7 of 8 flags (88%, 95%
interval 53-98%) and 23 of 24 inclusions (96%, 80-99%) that Hi-C could check at ≥ 10x agreed with it; with the
Nanomotif 1.3.0 table and version 0.11.1 it is 7 of 8 flags and 24 of 26 inclusions. Hi-C could check only these 8 of
the 139 flagged contigs and 24 of the 485 inclusions of the 0.8.0 run, the flag figure moves between 67% and 88% with the version of
the Hi-C reference (the inclusion figure stays at 92-100%), and the same sample was used to choose several defaults (the 0.4 methylome difference, the 0.9
inclusion threshold, the 5x cut-off), so it is not an independent check. With 0.12.0 (Nanomotif 1.3.0 table, calibration seed 1) the
sample gets 147 flags and 589 inclusions (0.11.1: 147 and 503), and the share of them that Hi-C confirms is
unchanged. Things to keep in mind:

- **A high `p_contaminant` is no guarantee**: the 5 wrong flags on the test and validation communities (spent) all had `p_contaminant`
  ≥ 0.9999. Sorting by `p_contaminant` shows the least certain flags first, but wrong flags are not only among them.
- **Very long contigs**: evidence grows with length faster than one temperature corrects, so long contigs are
  overconfident (on the fecal sample, the temperature that would fit each piece length alone rose from 2.7 at 2 kb
  to 14.7 at 500 kb). A long contig that is only partly foreign is reported as `mixed` instead of being flagged.
- **Plasmids and phages** in their host's bin can be flagged (see [above](#why-are-some-plasmids-or-phages-left-unbinned-or-flagged)).

## Known limitations

- **Unbinned genomes with a bin's methylation are placed into that bin.** Since 0.12 EpiRefine models a genome
  without a bin as a random draw from the community, on every run. Contigs from an unbinned genome that has the
  same, or nearly the same, methylation as a binned genome (a strain or a close sister that methylates the same
  motifs) look like members of that bin, and are placed into it. Methylation alone cannot tell the two apart. On
  simulated development sets built this way ("look-alikes"), 112 contigs of unbinned genomes went into a sister
  genome's bin over three sets (precision of the placements 67-81%); all were exact or near copies of the bin's
  methylation (differing only in a phase-variable motif, or in a motif with at most 2 sites on the contig). The
  relatives statistic in the log and in `params.json` (`relatives_statistic`, see below) warns of such a sample.
- **On such samples more short contigs get no call.** The calibration pieces of a bin with an unbinned
  same-methylome sister are confidently scored to the wrong genome, so the calibrated temperature rises (look-alikes:
  3.7-4.3, against 1.3 under 0.11.1). The evidence check is `expected_evidence` = evidence / T below
  `--min-evidence` (3), so more short contigs get the status `no_methylation_evidence`: about 12-35% of the contigs,
  depending on where the temperature lands (look-alike set s1: 245, 396 and 619 no calls of 1,849 contigs at T 3.7,
  4.3 and 5.0; development data). On the fecal development sample the temperature did not change (5.0) and
  neither did the no calls. A no call never changes a bin: the status is written last, and flags, inclusions and
  the refined bins do not depend on it.
- **A calibration that ends at the temperature cap gives no calls.** The temperature is fitted on pieces cut from
  the bins (a grid from 0.2 to 20; never below 1). If it ends at the cap of 20, no tempering made the pieces
  predictable, and the fit says nothing about how far the scores can be trusted. EpiRefine then makes no call at all
  (no flag, inclusion or move; the posteriors are still written, for inspection only), says so in one log line

  ```
  WARNING: calibration guard: the calibrated temperature is at the grid cap (20; 1 calibration pieces, 0 in test B): ...
  ```

  and records `calibration_guard` in `params.json`. On development data this happened only on two-bin stress
  samples with a single calibration piece holding almost no motif sites (or one scored confidently to the wrong
  bin) or with half of the methylation knocked out; never on the monocultures, the mock communities or the two
  fecal samples, where the temperature stayed between 1.08 and 5.9. More or longer binned contigs (more
  calibration pieces) are the remedy. `--temperature` overrides the guard with an uncalibrated temperature; on one
  development stress run, `--temperature 1` placed 25 contigs, one of them wrongly.
- **The relatives statistic.** Up to 0.11.1 EpiRefine chose between the community model and a "relatives" model
  (a genome without a bin resembles one of the bins) by a test on the calibration pieces, and the relatives model
  could take every unbinned contig ("nothing is included"). The test is still run and reported, but decides
  nothing:

  ```
  relatives statistic: 1.85 (log-loss gain of relatives over community only 0.1418 +- 0.0767 over bins; the 2-SE switch of 0.11 would choose community only; report only, the switch is removed)
  ```

  (the example is from the fecal development sample, calibration seed 1). `statistic` is the mean gain in log loss per calibration piece of the best model with relatives over the best
  community-only model, divided by its standard error (resampled over bins). Near 0 or below: nothing in the
  pieces points to unbinned relatives. Above about 2 (the old switch's limit): some bins have pieces that an
  unbinned genome with the same methylation would explain; expect the placements and no calls described above for
  those bins. On development data it was 0.4-1.9 on a fecal sample and 4.4-5.4 on the simulated look-alikes.
  `--relatives switch` restores the switch of 0.11.1 for comparison (deprecated).
- **The temperature moves in steps.** It is chosen from a grid of 31 values from 0.2 to 20, each about 17% above
  the one before. The number of inclusions can move with one step: on the fecal development sample, when the bins
  of the calibration were resampled, about one resample in six landed one step away from the full fit, and its
  inclusions moved by about 17%. A finer search was tried and is not in 0.12. Running a few values of
  `--calibration-seed` shows how much a sample's inclusions depend on it.
- **Confident errors on planted fecal scenarios.** On planted scenarios built from the fecal development sample
  (ZymoFecal s4-s8: binned contigs hidden or swapped), 0.12 returned more hidden contigs to their bin than the
  0.11.1-equivalent model (56.3% against 52.4%, 534 against 497 of 949), but its inclusion precision was 95.018% (534 of 562)
  against 95.029% (497 of 523), and it made 30 wrong placements at P ≥ 0.99 against 28. Each difference comes from
  one contig (m2_u19042189ctg placed in a wrong bin twice; m2_u10058973ctg placed in the same wrong bin at P 0.996-0.997
  where 0.11.1 placed it at 0.984-0.985), and for both the planted truth is contradicted by the runs without planting.
  Details: [CHANGELOG](../CHANGELOG.md), 0.12.0, Known limitations.
- **Some small samples cannot be calibrated.** A run stops with "no calibration fragments" when no calibration
  piece can be cut from the bins (see [Something went wrong](#something-went-wrong)). On a stress set of 690 runs
  on development genomes (replicate seeds included), 37 runs stopped this way, all on contigs cut into 1-2 kb
  pieces, the same 37 with 0.12 as with 0.10.1.
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

The rows give the version each was measured with; this table has no row for 0.12 yet. Since
0.12 the two largest tables are kept on disk (below), which lowers the main process's peak memory on large samples.
Memory is that of the largest single process; the worker processes come on top (an early version needed up to 54 GB
in total on the fecal sample with 8 workers). Most of the time on large samples is the first read of the pileup; a
rerun from the cache is much faster. On large samples EpiRefine keeps two large tables in temporary files while it
runs (about 12 GB on the fecal sample above), in the output directory or in the folder `--temp-dir` names, so leave
that much free space there. Do not put that folder in memory (tmpfs, `/dev/shm`; the log warns): the files then take
memory instead of disk. `--tables-in-memory` keeps the tables in memory instead, as before 0.12: no disk wait, about
6 GB more memory on that sample, the same results.

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

Two things can change the last digits of the probabilities without changing the decisions: the compiled
kernels against the numpy code (the pure wheel, or `EPIREFINE_RS=0`), and another machine or BLAS library. Both change only how
floating-point sums and exponentials are rounded. On the development benchmarks (monoculture training communities,
ZymoOral pieces, simulated look-alikes) runs with and without the kernels made the same calls, with probabilities
that agreed to 1e-10 or better. The kernels' two CPU paths (`avx2_fma` on x86-64 CPUs with AVX2 and FMA, `generic`
elsewhere; `params.json` names the one used) give the same bits.

### Does EpiRefine use the compiled kernels on my machine?

The third line of the log says so: `kernels: compiled (avx2_fma; built from <commit>)`, or `kernels: numpy
(<reason>)`. When the numpy code runs without `EPIREFINE_RS=0`, the log also has one warning line. The reasons:
no compiled wheel for the platform (pip then installs the pure wheel: Windows, free-threaded Python, PyPy, glibc
older than 2.17, other CPUs), a compiled module that does not load, or one too old for this version of EpiRefine.
The numpy code takes the same decisions and is slower, most in the pass over the pileup and in the model. To compile the kernels for such a
platform, install from the source distribution with a Rust toolchain and a C compiler:
`pip install --no-binary epirefine epirefine`.

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
