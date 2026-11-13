# Options

All options of `epirefine run`, grouped by what they control. The defaults were chosen during development, on
monoculture benchmarks with known truth and on one fecal sample with Hi-C (ZymoFecal). Several thresholds (the 0.4
methylome difference, the 0.9 inclusion threshold, the 5x coverage cut-off) were set on that fecal sample, so its Hi-C
agreement is not an independent check of them. Most users only set the inputs, `--out` and `--workers`.

[Inputs and output](#inputs-and-output) · [Decisions](#decisions) · [What gets judged](#what-gets-judged) ·
[Motifs](#motifs) · [Long contigs](#long-contigs) · [Model and calibration](#model-and-calibration) ·
[Speed and caching](#speed-and-caching) · [Examples](#examples)

## Inputs and output

| Option | Required | Description |
|---|---|---|
| `--assembly` | yes | Assembly FASTA (plain or gzipped). |
| `--contig-bin` | yes | The bins to refine: `contig<TAB>bin`, no header. Contigs not listed are unbinned. |
| `--pileup` | yes | `modkit pileup` bedMethyl (`.bed` or `.bed.gz`). Both the older space-separated and the all-tab formats work. The rows of each contig must be together, as modkit writes them. |
| `--motifs` | yes | Nanomotif `bin-motifs.tsv` (or any TSV with `motif`, `mod_type`, `mod_position`). Only the list of motifs is used, plus their complements when given. `mod_type` must be a modkit code (`a` = 6mA, `m` = 5mC, `21839` = 4mC, `h` = 5hmC), `mod_position` is 0-based; other mod types stop the run, invalid motifs are skipped and counted in the log. |
| `--out` | yes | Output folder. |
| `--motif-fit` | no | Also write `motif_fit.tsv`: each contig's methylated fraction per motif group next to its bin's. |
| `--seed` | no (1) | Random seed of the motif-grouping sample (part of the site-cache key) and, unless `--calibration-seed` is given, of the calibration pieces; the same seeds give identical outputs. |
| `--calibration-seed` | no (`--seed`) | Random seed of the calibration pieces only. Another value draws other pieces, and so shows how much the calibration depends on them, without changing the motif groups or invalidating the site cache. At the default the pieces are those of 0.11.1. |

## Decisions

| Option | Default | What it does | When to change it |
|---|---|---|---|
| `--contamination-threshold` | 0.5 | Flag a binned contig when the probability that it doesn't belong is at least this. | Raise it to flag only the most certain contaminants. |
| `--min-methylome-difference` | 0.4 | A flag also needs one motif group where the contig's 95% interval for its methylated fraction lies at least this far from the bin's. Keeps many small, consistent shifts (variation within a genome) from adding up to a flag. 0 turns it off. | Lower it to flag subtler differences; expect more false flags. |
| `--include-threshold` | 0.9 | Include an unbinned contig into a bin (or move a flagged one) when that bin's probability is at least this. | Raise it for fewer, safer inclusions. |
| `--include-threshold-unmethylated` | 0.95 | The same for bins without detected methylation, which any contig whose own motifs are not in the motif list matches. | Rarely. |
| `--max-redundancy` | 0.3 | Don't include a contig when this share of its 21-mers is already in the bin (an assembly duplicate). 1 turns the check off. | Raise it if you want duplicated sequence included. |
| `--p-unknown` | estimated | Prior probability that an unbinned contig's genome has no bin. Estimated from the unbinned contigs by default. | Set it when you know how complete the binning is. |

## What gets judged

| Option | Default | What it does |
|---|---|---|
| `--min-contig-coverage` | 5 | Contigs whose coverage (the median of the valid calls per motif site, both strands summed at sites of palindromic motifs; see [outputs.md](outputs.md#coverage-and-5x)) is below this get no decision (status `low_coverage`). On a fecal development sample, Hi-C confirmed about a quarter of the flags below 5x (12 of 46). 0 judges all. |
| `--min-contig-length` | 1000 | Shorter contigs get no decision (status `too_short`). 0 judges all. |
| `--min-evidence` | 3.0 | Contigs whose expected methylation evidence (in nats) is below this get status `no_methylation_evidence`: they carry too little methylation to tell bins apart. |
| `--min-cov` | 1 | Motif sites with fewer reads than this are ignored. |

## Motifs

| Option | Default | What it does |
|---|---|---|
| `--min-jaccard` | 0.9 | Motifs whose sites overlap this much are merged into one motif group. A site matched by several groups counts in the most specific one. |
| `--group-sample-mb` | 50 | Mb of sequence sampled to measure the overlap between motifs. |
| `--shared-sites` | `sample` | Which group gets a site that motifs of several groups match: the group with the fewest occurrences in the sampled sequence (`sample`), or the one whose motif is rarest in random sequence (`motif`; does not depend on the sample or the seed). |
| `--complement-pairs` | `sequence` | A motif and its reverse complement in the motif list: one two-strand site per occurrence, paired by sequence (`sequence`); paired only when each has a single modified position (`position`; the same sites unless a sequence is listed with several modified positions); or kept as two motifs with single-strand sites (`separate`). |
| `--strand-sites` | `sum` | Reads of the two strands at one occurrence of a palindrome or complement pair: summed into one site (`sum`); one site per strand (`split`); or summed unless the strands can differ, i.e. complement pairs and occurrences where one strand is also a site of another motif group (`auto`). |
| `--noise-max-off-share` | 1 | Call a motif group noise only when at most this share of the bins that measure it leave it unmethylated (1 = not used). |
| `--no-motif-noise-filter` | off | Keep motif groups that look like basecaller noise (partly methylated on many contigs of many bins; see `motif_noise.tsv`). By default they are dropped. |
| `--no-group-weights` | off | Count every motif group fully. By default, groups that can share sites and are methylated in the same bins share one weight, so variants of one enzyme don't vote several times (see `group_weights.tsv`). |

## Long contigs

| Option | Default | What it does |
|---|---|---|
| `--chunk-size` | 50000 | Binned contigs are split into pieces of this size to build robust bin fingerprints and to judge long contigs along their length. |
| `--no-regional-model` | off | Judge long binned contigs only as a whole. By default a long contig that is partly its bin's and partly another genome's is reported as `mixed`, with its foreign segments, instead of being flagged. |

## Model and calibration

EpiRefine calibrates its probabilities on pieces of the sample's own bins; these options override what it
estimates. Leave them unset unless you are testing the model.

| Option | Default | What it does |
|---|---|---|
| `--temperature` | calibrated | Divides all evidence before it becomes probabilities (higher = less confident). |
| `--allow-sharpening` | off | Allow a calibrated temperature below 1 (by default probabilities are never made sharper than the raw evidence). |
| `--variation` | beta | How a contig's methylated fraction varies around its bin's: `beta` or `logit` (logit-normal). |
| `--tau` | calibrated | The variation parameter (beta concentration, or logit-normal sigma). |
| `--no-calibrate` | off | Skip calibration: temperature 1 and a fixed variation (beta concentration 100) unless set. Probabilities are then overconfident; for testing only. |
| `--relatives` | `community-only` | Model of a genome without a bin: a random draw from the community on every run (since 0.12). The relatives test of 0.11 is reported as a number (log line `relatives statistic`, `params.json` `relatives_statistic`) and decides nothing. The switch of 0.11.1 (`switch`) is deprecated and kept for comparison only; it is used automatically with `--no-calibrate`. |
| `--unknown-share` | `t1` | How the share of unbinned contigs from genomes without a bin (`p_unknown`) is estimated: at temperature 1 (`t1`, since 0.12; it does not rise with the temperature), or at the calibrated temperature (`deployed`, 0.11; the default with `--relatives switch`). |
| `--temperature-law` | `grid` | Experimental. `sites`: a temperature that grows smoothly with a contig's number of motif sites (log T = a + b log(sites / 1000), b between 0 and 1, T at least 1.082) instead of one temperature from the grid. |
| `--evidence-temperature` | `median` | Experimental. With `--temperature-law sites`, `contig` divides each contig's methylation evidence by its own temperature instead of the median one. |
| `--frag-lengths` | `2000,5000,10000,20000,50000,100000` | Lengths of the calibration pieces in bp, comma-separated. |
| `--frag-budget` | 6000 | Total number of calibration pieces, spread over bins and lengths. |
| `--frags-per-bin` | 30 | At most this many pieces per bin and length. |
| `--contam-eps` | 0.05 | Prior share of foreign pieces in a bin when building its fingerprint (0 = trust every piece; the regional model for long contigs is then off too, and the log says so; must be below 1). |
| `--unknown-prior` | `bins` | `unbinned`: a genome without a bin is assumed to carry each motif as often as the unbinned contigs do, not as often as the bins do. On development sets, far fewer contigs of genomes without a bin were placed into bins with no or one common motif (13 -> 0 and 233 -> 16 in two test sets), and fewer removed contigs were put back (training communities 89.3% -> 81.8%). Not tested on held-out data or a metagenome. |
| `--unknown-prior-strength` | 5 | With `--unknown-prior unbinned`: pseudo-contigs at the bins' frequency (large values return to `bins`). |
| `--redundancy-order` | `independent` | `sequential`: inclusion candidates of a bin are compared with the bin plus the candidates already accepted (most probable first), so two unbinned copies of one sequence are not both included. |
| `--regional-leave-out`, `--regional-withdraw`, `--regional-tempered` | `chunk`, `vs_foreign`, off | Variants of the regional model: `contig` scores a long contig's pieces against the bin without the whole contig; `largest` withdraws a flag as `mixed` only when mixed is the most probable of member / foreign / mixed. |
| `--gate-leave-out`, `--gate-unweighted`, `--eps-gated` | `none`, off, off | Variants of the methylome-difference requirement: against the bin's 95% interval without the contig; contig intervals from unweighted sites; contamination rate from contigs that pass it. |
| `--siteless-bin-guard` | off | Include nothing into a bin without a single motif site. |
| `--grid-size` | 64 | Points on the grid of methylated fractions. |
| `--two-state-warning` | 0.2 | Share at which `bin_evidence.tsv` warns about a bin with two methylation states, or with a large flagged share (reporting only). |

`epirefine --version` prints the version (also the first line of every log and `version` in `params.json`).

Removed in 0.12 (they lost on the development benchmarks; see `CHANGELOG.md`): `--calibration`, `--robust-fit`,
`--max-flagged-share`, `--tau-support`, `--unlisted-genome`, `--inclusion-support`, `--regional-inclusion`,
`--tau-select`, `--npmle-start`, `--relatives per-bin` and `slot` with `--slot-objective`, `--slot-test-b` and
`--slot-kappa`, and the hidden `--calibration-exclusion`.

## Speed and caching

| Option | Default | What it does |
|---|---|---|
| `--workers` | number of CPUs, at most 16 | Processes for reading the pileup and building the tables. Uncompressed pileups are read in parallel; a gzipped pileup is read by one worker, so unpack large pileups first. |
| `--temp-dir` | `<out>` | Folder for the temporary files of the two large tables (contig tables and chunk tables) on large samples: about 12 GB on a 35,000-contig fecal sample, removed when the run ends, also when it fails. Created if missing; a folder in which no file can be made stops the run at the start. Use a folder on disk: on tmpfs (`/dev/shm`, often an HPC job's `$TMPDIR`) the files take memory, and the log warns. |
| `--tables-in-memory` | off | Keep the contig tables in memory and the chunk tables' file in the page cache, as before 0.12: no wait for the disk, but about 6 GB more memory on a 35,000-contig fecal sample. The results are the same. |
| `--cache-dir` | `<out>/cache` | Where the site cache goes. A rerun with the same assembly, pileup and motifs reuses it and skips reading the pileup, so reruns with other thresholds are fast. Point several runs at the same cache to share it: runs that read it share a lock on it and run at the same time; a run that has to rebuild it waits until they are done (and they wait for a rebuild). |

### The site cache

The cache is reused when its key matches: the assembly and the pileup (path, size and modification time, plus a
fingerprint of a few 1 MiB blocks at the start, middle and end of each file; small files are hashed whole), the
motif list, the motif groups (which depend on `--seed`, `--group-sample-mb` and `--min-jaccard`) and the site
rules `--shared-sites`, `--complement-pairs` and `--strand-sites`. `--calibration-seed` and every decision
option leave it valid. Caches written before 0.12 have no fingerprint (format 1); they are reused on path, size and
time as before, and the log says so.

```bash
# would a run reuse this cache? (exit status 0: reused, 1: would rebuild); reads no more of the pileup than the
# fingerprint blocks
epirefine cache check --cache-dir run1/cache --assembly asm.fasta --pileup pileup.bed --motifs bin-motifs.tsv
# add the fingerprints to a cache written before 0.12 (only if it was built from these very files)
epirefine cache upgrade --cache-dir run1/cache --assembly asm.fasta --pileup pileup.bed
```

## Repeating a run

`params.json` records the inputs, every option, the seeds, the commit, the library versions, the kernels and the
threads. `epirefine rerun` repeats the run it describes, reading the inputs from their recorded paths and reusing the
recorded run's site cache:

```bash
epirefine rerun --params first_run/params.json --out repeat [--workers N] [--cache-dir DIR]
```

The log warns when an input file's size or modification time, the version or the commit differs from the record.
Decisions do not depend on the worker count.
A run recorded before 0.12 that used an option removed in 0.12 (below) cannot be repeated by this version: `rerun`
stops with one line naming the option; repeat it with the version the record names. Such options at the value that
was their default (for example `"calibration": "default"` in a 0.11.1 record) are accepted, since that behaviour
remains.

## Examples

Only the most certain decisions:

```bash
epirefine run --assembly asm.fasta --contig-bin bins.tsv --pileup pileup.bed --motifs bin-motifs.tsv \
    --out strict --contamination-threshold 0.9 --include-threshold 0.99
```

Rerun with other thresholds, reusing the first run's site cache:

```bash
epirefine run ... --out strict --cache-dir first_run/cache
```

Also judge contigs at low coverage (decisions there are much less reliable):

```bash
epirefine run ... --out lowcov --min-contig-coverage 2
```
