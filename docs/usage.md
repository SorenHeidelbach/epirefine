# Options

All options of `epirefine run`, grouped by what they control. The defaults were chosen on benchmarks with known
truth and checked against Hi-C on a fecal sample; most users only set the inputs, `--out` and `--workers`.

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
| `--seed` | no (1) | Random seed; the same seed gives identical outputs. |

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
| `--min-contig-coverage` | 5 | Contigs with a median below this many reads per motif site get no decision (status `low_coverage`). Against Hi-C, flags below 5x were right only 12.5% of the time. 0 judges all. |
| `--min-contig-length` | 1000 | Shorter contigs get no decision (status `too_short`). 0 judges all. |
| `--min-evidence` | 3.0 | Contigs whose expected methylation evidence (in nats) is below this get status `no_methylation_evidence`: they carry too little methylation to tell bins apart. |
| `--min-cov` | 1 | Motif sites with fewer reads than this are ignored. |

## Motifs

| Option | Default | What it does |
|---|---|---|
| `--min-jaccard` | 0.9 | Motifs whose sites overlap this much are merged into one motif group. A site matched by several groups counts in the most specific one. |
| `--group-sample-mb` | 50 | Mb of sequence sampled to measure the overlap between motifs. |
| `--shared-sites` | `sample` | Which group gets a site that motifs of several groups match: the group with the fewest occurrences in the sampled sequence (`sample`), or the one whose motif is rarest in random sequence (`motif`; does not depend on the sample or the seed). |
| `--complement-pairs` | `sequence` | A motif and its reverse complement in the motif list: one two-strand site per occurrence, paired by sequence (`sequence`); paired only when each has a single modified position (`position`); or kept as two motifs with single-strand sites (`separate`). |
| `--strand-sites` | `sum` | Reads of the two strands at one occurrence of a palindrome or complement pair: summed into one site (`sum`); one site per strand (`split`); or summed unless the strands can differ, i.e. complement pairs and occurrences where one strand is also a site of another motif group (`auto`). |
| `--noise-max-off-share` | 1 | Call a motif group noise only when at most this share of the bins that measure it leave it unmethylated (1 = not used). |
| `--no-motif-noise-filter` | off | Keep motif groups that look like basecaller noise (partly methylated on many contigs of many bins; see `motif_noise.tsv`). By default they are dropped. |
| `--no-group-weights` | off | Count every motif group fully. By default, groups that can share sites and are methylated in the same bins share one weight, so variants of one enzyme don't vote several times (see `group_weights.tsv`). |

## Long contigs

| Option | Default | What it does |
|---|---|---|
| `--chunk-size` | 50000 | Binned contigs are split into pieces of this size to build robust bin fingerprints and to judge long contigs along their length. |
| `--no-regional-model` | off | Judge long binned contigs only as a whole. By default a long contig that is partly its bin's and partly another genome's is reported as `mixed`, with its foreign segments, instead of being flagged. |
| `--regional-inclusion` | off | Also include long unbinned contigs from their 50 kb windows. Experimental; it added inclusions the whole-contig model didn't support. |

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
| `--calibration` | `legacy` | Calibration design: `legacy` (as in 0.10), `coherent`, or a comma-separated list of features; see `docs/development/method-notes.md` §7.6. |
| `--frag-lengths` | `2000,5000,10000,20000,50000,100000` | Lengths of the calibration pieces in bp, comma-separated. |
| `--frag-budget` | 6000 | Total number of calibration pieces, spread over bins and lengths. |
| `--frags-per-bin` | 30 | At most this many pieces per bin and length. |
| `--contam-eps` | 0.05 | Prior share of foreign pieces in a bin when building its fingerprint (0 = trust every piece; the regional model for long contigs is then off too, and the log says so; must be below 1). |
| `--robust-fit` | `alternating` | `damped`: iterate the bin fingerprints in damped steps until they settle (slower; same results on every benchmark so far). |
| `--unknown-prior` | `bins` | `unbinned`: a genome without a bin is assumed to carry each motif as often as the unbinned contigs do, not as often as the bins do. Far fewer contigs of genomes without a bin are placed into bins with no or one common motif; fewer contigs without a site of their bin's motifs are put back. |
| `--unknown-prior-strength` | 5 | With `--unknown-prior unbinned`: pseudo-contigs at the bins' frequency (large values return to `bins`). |
| `--redundancy-order` | `sequential` | Inclusion candidates of a bin are compared with the bin plus the candidates already accepted (most probable first), so two unbinned copies of one sequence are not both included. `independent`: each candidate against the given bin only (the behaviour up to 0.10). |
| `--regional-leave-out`, `--regional-withdraw`, `--regional-tempered` | `chunk`, `vs_foreign`, off | Variants of the regional model: `contig` scores a long contig's pieces against the bin without the whole contig; `largest` withdraws a flag as `mixed` only when mixed is the most probable of member / foreign / mixed. |
| `--gate-leave-out`, `--gate-unweighted`, `--eps-gated` | `none`, off, off | Variants of the methylome-difference requirement: against the bin's 95% interval without the contig; contig intervals from unweighted sites; contamination rate from contigs that pass it. |
| `--max-flagged-share` | 1 | Apply no flag of a bin whose flagged contigs hold at least this share of it (not recommended: see the dev log proposal of 2026-11-10). |
| `--tau-support` | `all` | `leave_bin_out`: a motif group gets between-contig variation only if the calibration supports it without its best bin. |
| `--unlisted-genome`, `--inclusion-support` | off, `none` | Two rejected inclusion designs, kept for comparison. |
| `--grid-size` | 64 | Points on the grid of methylated fractions. |

## Speed and caching

| Option | Default | What it does |
|---|---|---|
| `--workers` | number of CPUs, at most 16 | Processes for reading the pileup and building the tables. Uncompressed pileups are read in parallel; a gzipped pileup is read by one worker, so unpack large pileups first. |
| `--cache-dir` | `<out>/cache` | Where the site cache goes. A rerun with the same assembly, pileup and motifs reuses it and skips reading the pileup, so reruns with other thresholds are fast. Point several runs at the same cache to share it. |

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
