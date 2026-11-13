# Outputs

Everything EpiRefine writes into `--out`. For most uses you need only the first two files.

| File | Use it for |
|---|---|
| [`new_contig_bin.tsv`](#new_contig_bintsv) | **The refined binning.** |
| [`contig_decisions.tsv`](#contig_decisionstsv) | **Every contig's decision, with probabilities.** |
| [`decontaminated_contig_bin.tsv`](#nanomotif-compatible-files) | The bins with contaminants removed, nothing added. |
| [`bin_contamination.tsv`, `include_contigs.tsv`](#nanomotif-compatible-files) | Nanomotif-format flags and inclusion candidates. |
| [`bin_evidence.tsv`, `host_profiles.tsv`, `motif_fit.tsv`](#bins-and-motifs) | What each bin's methylation looks like (`motif_fit.tsv` with `--motif-fit`). |
| [`motif_groups.tsv`, `motif_noise.tsv`, `group_weights.tsv`, `detection_model.tsv`](#bins-and-motifs) | How the motifs were grouped, filtered and modelled. |
| [`host_chunk_outliers.tsv`](#diagnostics) | Pieces of binned contigs that look foreign to their bin. |
| [`calibration/`](#the-calibration-folder) | Eleven tables on the calibration of the probabilities. |
| [`params.json`](#paramsjson) | What the run was made with and what it fitted. |
| [`unbinned_prior.tsv`](#outputs-of-non-default-options) | Only with `--unknown-prior unbinned`. |
| `cache/` | The site cache, reused by reruns (see `--cache-dir`). `manifest.json` holds its key (`inputs`), its part files (`parts`), `format` (2 since 0.12; missing in older caches) and `fingerprints` of the assembly and the pileup; `.lock` is the lock file runs share. |

All tables are tab-separated with a header. Empty text fields are written as `""`, missing numbers as `NaN`, and
decimals with 4 digits after the point.

## `new_contig_bin.tsv`

The refined binning, `contig`, `bin`: every binned contig that was kept (including `mixed` ones), plus the
`included` and `moved` contigs in their new bin. Contigs flagged as contaminants without a new bin are not listed.
To write FASTA files per bin, see [getting started, step 5](getting-started.md#5-write-the-refined-bins).

## `contig_decisions.tsv`

One row per contig of the assembly.

| Column | Meaning |
|---|---|
| `contig`, `length` | contig name and length (bp) |
| `n_sites` | motif sites with reads on the contig |
| `given_bin` | its bin in the input (empty = unbinned) |
| `status` | the decision ([table below](#statuses)) |
| `new_bin` | its bin after refinement (empty = none) |
| `p_contaminant` | binned contigs: probability that it does not belong to its bin |
| `methylome_difference` | binned contigs: the largest gap between the contig's 95% interval for a motif group's methylated fraction and the bin's level; a flag needs at least `--min-methylome-difference` (0.4) |
| `redundancy_with_bin` | share of the contig's 21-mers already in the bin it would join (checked for inclusions) |
| `best_alternative`, `p_best_alternative` | the most probable home other than its own bin, and its probability among the alternatives; `unknown` = a genome without a bin |
| `map`, `p_map` | the most probable explanation overall (own bin, another bin or `unknown`) and its probability |
| `p_unknown` | probability that the contig comes from a genome without a bin |
| `credible_set_95` | the smallest set of bins (and `unknown`) that together hold 95% of the probability; one entry = a clear answer, several = methylation cannot decide between them |
| `expected_evidence` | how much methylation evidence (nats) the contig is expected to carry for its bin (its input bin; for an unbinned contig, its best-scoring bin) against a random genome of the community, computed from its sites and read counts only and divided by the run's temperature; below `--min-evidence` (3), a contig that would be kept or unbinned gets no call (status no_methylation_evidence) |
| `coverage` | the contig's coverage as EpiRefine counts it: the median, over all its motif sites, of the valid calls at a site (["5x" below](#coverage-and-5x)); 0 for a contig without motif sites; below `--min-contig-coverage` (5) it is not judged |
| `p_regional_foreign`, `p_regional_mixed` | long binned contigs: probability that it is foreign along its whole length, and that it is partly its bin's and partly foreign |
| `foreign_segments`, `segment_bins` | for `mixed` contigs: the foreign stretches (`start-end` in bp, `;`-separated) and the bin each fits best |

### Statuses

| Status | Meaning | In `new_contig_bin.tsv` |
|---|---|---|
| `kept` | binned; its methylation fits its bin | yes, in its bin |
| `contaminant` | binned; its methylation contradicts its bin (`p_contaminant` ≥ 0.5 and a clear difference in at least one motif group). Also a contaminant that fitted another bin but was not moved there because that bin already holds its sequence (`redundancy_with_bin` ≥ 0.3) | no |
| `moved` | a contaminant whose best alternative bin has probability ≥ 0.9 | yes, in that bin |
| `mixed` | a long binned contig that is partly its bin's and partly another genome's; nothing is split | yes, in its bin |
| `included` | unbinned; one bin has probability ≥ 0.9 (≥ 0.95 for a bin without detected methylation) | yes, in that bin |
| `unbinned` | unbinned; no bin is probable enough | no |
| `redundant` | unbinned; would be included, but ≥ 30% of its 21-mers are already in that bin (an assembly duplicate) | no |
| `sole_contig_of_bin` | the only contig of its bin, so there is no profile without it to compare it with | yes, unchanged |
| `no_methylation_evidence` | a no call: the contig would be `kept` or `unbinned`, but its `expected_evidence` is below `--min-evidence`, so methylation cannot support that. Set last, after every flag and inclusion: it never changes a bin | binned: yes, unchanged |
| `low_coverage` | `coverage` below `--min-contig-coverage` (5): no call | binned: yes, unchanged |
| `too_short` | shorter than `--min-contig-length` (1 kb): no call | binned: yes, unchanged |

`low_coverage` and `too_short` contigs take no part in any estimate; `no_methylation_evidence` is a label only. When the [calibration guard](#paramsjson) fires, no contig is `contaminant`, `moved` or `included`.

### Coverage and "5x"

`coverage` and `--min-contig-coverage` count valid calls per motif site, not read depth. For each motif site the
pileup's valid calls (`N_valid`, column 10 of modkit's bedMethyl) are summed over the rows that belong to the site;
the contig's `coverage` is the median of these sums over all its motif sites (every motif group, including groups
the noise filter drops). With the default `--strand-sites sum`, a site of a palindromic motif (such as GATC), or of
a motif listed with its reverse complement, holds both strands, so its count is the valid calls of both strands
summed; a site of any other motif is one strand. "5x" (the default cut-off) therefore means a median of 5 valid
calls per site: at a given read depth a palindromic site counts about twice what a one-strand site counts, and the
median mixes the two kinds in the proportions the contig's motifs have. Valid calls exclude reads that modkit
filtered or could not call at the site, so they are below the read depth. With `--strand-sites auto` the sites
stored as single strands count double here, so the coverage does not depend on how the strands were split.

## Nanomotif-compatible files

Same names and columns as Nanomotif's `detect_contamination` and `include_contigs`, so tools built for those
outputs can read EpiRefine's.

- **`bin_contamination.tsv`**: one row per flagged contig: `contig`, `bin`, `method` (`epirefine`), `cluster` (the
  best alternative), `bin_cluster`, `bin_length`, `n_contigs_bin`, `fraction_contigs`, `fraction_length`,
  `p_contaminant`, `p_cluster`.
- **`include_contigs.tsv`**: inclusion candidates with probability ≥ 0.5: `contig`, `bin` (its current bin or
  `unbinned`), `assigned_bin`, `method`, `prob`, `mean_prob`, `confidence` (`high_confidence` ≥ 0.95,
  `medium_confidence` ≥ 0.8, `low_confidence` ≥ 0.5; `dropped_redundant` for a candidate that reached the inclusion
  threshold but was not added because the bin already holds its sequence). This lists candidates, not decisions:
  inclusion is applied at 0.9 (0.95 for a bin without detected methylation), so some `medium_confidence` rows are
  applied and the contigs actually added are those with status `included` or `moved` in `contig_decisions.tsv`.
  When the calibration guard fires (`params.json` `calibration_guard.fired`), the file has its header and no row.
- **`decontaminated_contig_bin.tsv`**: `contig`, `bin`: the input bins minus the flagged contigs.

## Bins and motifs

- **`bin_evidence.tsv`**, one row per bin:
  - `bin`; `status`: `methylated` (some motif group's 95% interval lies above 0.5), `no_methylation_detected` (every
    group with sites lies confidently below 0.5; such a bin needs the stricter inclusion threshold) or
    `no_methylation_evidence` (neither: too few sites or reads to say);
  - `groups_with_sites`, `groups_methylated`, `groups_confidently_unmethylated`, `groups_undetermined`: motif groups
    with sites on the bin's contigs, and how many of them are methylated, confidently unmethylated (95% interval
    below 0.5) or neither; `sites`: the motif sites on the bin's contigs;
  - `robust_fit`: `converged`, or `not_converged` when the bin's fingerprint did not settle;
  - `flagged_share`: share of the bin's judged sequence that was flagged;
  - `two_state_share` with `two_state_group`: the motif group in which the bin's pieces fall into two methylation
    states, one below and one above 0.5, and the share of the smaller state (0 when the pieces agree);
  - `warning`: `;`-separated, empty when none: `two_methylation_states` (`two_state_share` ≥ `--two-state-warning`,
    0.2), `large_flagged_share` (`flagged_share` ≥ the same 0.2), and `profile_not_converged`.

  A `two_state_share` or `flagged_share` of 0.2 or more means the bin probably holds two methylomes: two genomes or
  strains merged by the binner, or one genome with part of its sequence in another state. When the two differ in a
  single sparse motif nothing is flagged in such a bin (the fingerprint settles between them), so this column is
  then the only sign; look at the bin before trusting it.
- **`host_profiles.tsv`**: each bin's fingerprint, one row per bin and motif group (`bin`, `group`, `motifs`): the
  mean methylated fraction (`phi_mean`), its 95% interval (`phi_lo95`, `phi_hi95`) and the probability that it is
  above 0.5 (`p_phi_gt_0.5`). The rows with `bin` = `_population_prior_` describe the community: the distribution of
  methylated fractions over all bins, which is also the model of a genome without a bin.
- **`motif_fit.tsv`** (with `--motif-fit`): one row per contig and motif group with sites (`contig`, `group`,
  `motifs`, `n_sites`): the contig's methylated fraction (`phi_observed`) with its 95% interval
  (`phi_observed_lo95`, `phi_observed_hi95`), next to the mean of `reference_bin` (`phi_bin`). The reference bin is
  the contig's own bin, or for an unbinned contig its best alternative bin (empty when that is `unknown`). Useful to
  see why a contig was flagged.
- **`motif_groups.tsv`**: which Nanomotif motifs were merged into which motif group (`group`, `mod_type`, `motifs`,
  written as `MOTIF_modtype-position`), with the group's motif sites with at least `--min-cov` calls on all contigs
  (`n_sites`) and the valid calls at them (`n_reads`); both empty for a group the noise filter dropped.
- **`motif_noise.tsv`**: the noise filter, one row per motif group measured on binned contigs of at least 10x
  (`group`, `mod_type`, `motifs`). A contig is noise-like for a group when it has at least 20 sites with at least 5
  calls, their mean modified fraction of reads lies between 0.15 and 0.6, and at least 20% of them lie between 0.2
  and 0.8; a bin is noise-like when at least half its measured contigs are. `bins`: bins that measure the group;
  `noise_bins` and `noise_bin_share`: how many of them are noise-like, and their share; `median_level`: over bins,
  the median of the bin's level (the median over its contigs of the mean modified fraction); `off_bin_share`: share of the bins
  where the group is unmethylated (level below 0.05; a dropped group that is off in many bins may be a real, weakly
  detected motif, see `--noise-max-off-share`); `noisy`: dropped as basecaller noise (at least 3 noise-like bins
  making up at least 20% of the bins).
- **`group_weights.tsv`**: groups that behave as one enzyme and share one weight (`group`, `motifs`, `cluster`,
  `cluster_size`, `weight` = 1 / `cluster_size`).
- **`detection_model.tsv`**: the basecaller model per motif group (`group`): the mean share of reads called modified
  at unmethylated (`mu0`) and at methylated sites (`mu1`), their concentrations (`kappa0`, `kappa1`; small = more
  spread between sites), `pi1`, the share of the labelled sites (those of bins clearly unmethylated or clearly
  methylated in the group) that come from methylated bins, and `n_sites`, the group's sites with at least
  `--min-cov` calls on binned contigs.

## Diagnostics

- **`host_chunk_outliers.tsv`**: every 50 kb piece of every binned contig that has motif sites (`contig`, `start`,
  `bin`) with its probability of being foreign to its bin (`p_foreign`). Pieces likely to be foreign are
  down-weighted when the bin's fingerprint is built.

### The calibration folder

`calibration/` describes how the probabilities were calibrated on pieces cut from the binned contigs (written
unless `--no-calibrate`). Each piece is scored as if it were unbinned, against all bins with its own bin rebuilt
without its source contig (**test A**, the right answer is its bin), and against the bins without its own bin
(**test B**, the right answer is "unknown", a genome without a bin). Test B uses only the pieces whose bin can be
told from its nearest competitor ("distinct": the bin has no look-alike and the piece's expected log Bayes factor
for its bin over the nearest other bin is at least 3). Pieces that overlap a 50 kb chunk the robust fit calls foreign
to its bin are left out of the calibration (`foreign_excluded`). Lengths are in bp. `fragments.tsv`, `summary.tsv`,
`reliability.tsv`, `temperature_by_length.tsv` and `assignability.tsv` are computed at the temperature and with the
model of a genome without a bin that the decisions use: the community-only fit, also on runs where `params.json`
holds `unknown_model_forced` (the 0.11 switch would have chosen relatives; with the deprecated `--relatives switch`
they describe the switch's choice). `false_flag_rate.tsv` and `reliability_deployed.tsv` use the settings of the
decisions.

| Table | One row per | Columns |
|---|---|---|
| `fragments.tsv` | calibration piece | `unit` (piece number), `contig` (source contig), `bin`, `tile` (position of the piece on the contig, in piece lengths), `length`, `start`, `end`; `foreign_excluded` (true: left out, see above); for the kept pieces: `expected_log_bf_vs_nearest` and `expected_log_bf_vs_unknown` (the evidence the piece is expected to give for its bin over the nearest other bin and over a community genome, from its sites and calls only, divided by the temperature), `posterior_true` (the test-A probability of its bin at the calibrated settings), `map_correct` (its most probable label is its bin); `NaN` for excluded pieces |
| `tau_fit.tsv` | value of the variation parameter tau tried, plus `per-group` when the groups disagree | `tau` (`NaN` for the per-group row), `label`, `loo_loglik` (summed log-likelihood of the kept pieces under their own bin, built without their contig), `n_fragments` (pieces kept by that fit) |
| `tau_groups.tsv` | motif group | `group`, `tau` (the value chosen for it; `inf` = no variation between contigs and their bin), `loo_loglik_gain_vs_no_variation` (gain of that choice over tau = inf, in nats) |
| `temperature_fit.tsv` | model of a genome without a bin and temperature on the grid | `eta`, `lam` (`lam` = 1: community draw, the model in use by default; `lam` < 1: a mixture, with weight 1 - `lam`, with an unbinned relative of a bin, which takes each motif group's level from that bin with probability 1 - `eta`; by default only scored, for the relatives statistic), `temperature`, `log_loss` (`log_loss_A` + `log_loss_B_distinct`), `log_loss_A` (mean log loss of test A), `log_loss_B_distinct` (of test B, `NaN` when no piece is distinct) |
| `temperature_by_length.tsv` | piece length | `length`, `fragments` (kept pieces), `temperature` (the grid temperature that would fit those pieces alone, any value of the grid, also below 1), `median_sites` (motif sites per piece); a diagnostic, not used |
| `summary.tsv` | scaling and piece length (`all` = every length) | `scaling`: `uncalibrated` (temperature 1), `fitted_temperature` (the best grid value, also below 1) or `calibrated` (the temperature in use); `length`, `n`; test A: `A_accuracy` (share whose most probable label is right), `A_mean_confidence` (mean probability of the most probable label), `A_ece` (expected calibration error over 10 probability bins), `A_log_loss`, `A_set95_coverage` (share whose 95% credible set holds the right answer), `A_set95_mean_size`, `A_mean_p_unknown` (mean probability of "unknown"); test B over all kept pieces: `B_unknown_rate` (share with "unknown" most probable), `B_false_confident_rate` (share with some bin at probability ≥ 0.95), `B_log_loss`; and over the distinct pieces: `B_distinct_n`, `B_distinct_unknown_rate`, `B_distinct_false_confident_rate`. Probabilities at the reference share of unknown genomes, 0.1 (or `--p-unknown`) |
| `reliability.tsv` | test and probability bin of width 0.1 | `test` (`A` or `B`, all kept pieces, calibrated temperature, reference share 0.1), `conf_lo`, `conf_hi` (the bin), `n`, `mean_confidence` (mean probability of the most probable label), `accuracy` (share whose most probable label is right) |
| `assignability.tsv` | class of expected evidence | `expected_log_bf_lo`, `expected_log_bf_hi` (the class of `expected_log_bf_vs_nearest`: below 0.5, 0.5-1, 1-2, 2-3, 3-5, 5-10, 10 and more), `n`, `accuracy`, `mean_confidence`, `set95_mean_size` (test A), `mean_observed_log_bf_true_vs_nearest` (the observed log Bayes factor of the right bin over the best other bin, divided by the temperature). Shows whether the expected evidence, computed without looking at methylation, predicts accuracy |
| `lookalikes.tsv` | bin | `bin`, `nearest_bin` (the bin with the closest fingerprint), `max_phi_difference` (the largest difference of mean methylated fraction between the two over motif groups), `has_lookalike` (that difference is below 0.15: the two cannot be told apart, and the bin's pieces are left out of test B) |
| `false_flag_rate.tsv` | piece length (`all` = every length) and set of pieces | `length`, `fragments` (`all` pieces, or those `kept` by the calibration), `n`, `median_sites`, `mean_p_foreign` (mean probability of being foreign when each piece is judged as a binned contig of its own bin, at the run's share of unknown genomes, contamination rate and temperature), `rate_p_foreign_above_threshold` (share at or above `--contamination-threshold`), `rate_gate_passed` (share that clears `--min-methylome-difference`), `false_flag_rate` (share meeting both: true members that would be flagged), `false_flags` (their number) |
| `reliability_deployed.tsv` | test, piece length and probability bin of width 0.1 | `test`: `A_any_candidate` (every candidate's probability in test A; observed = it is the right one), `A_map` (the most probable label; observed = it is right), `B_unknown` (the probability of "unknown" in test B on the distinct pieces; observed = 1); `length`, `p_lo`, `p_hi` (the bin), `n`, `mean_p` (mean predicted probability), `observed` (observed share). At the share of unknown genomes and the temperature used for the decisions |

`false_flag_rate.tsv` brackets the false-flag rate: the `kept` row leaves out the pieces of chunks the robust fit
calls foreign (a lower bound), the `all` row counts them (an upper bound: where a bin really holds foreign contigs,
their pieces are correctly flagged and counted here as false). On the development benchmarks 0 of 5,723 kept member
pieces of the training communities would have been flagged (development data), so a clearly higher rate means flags
in this sample deserve a second look; no threshold for it has been validated.

## `params.json`

What the run was made with and what it fitted. Written when the run starts (`finished` is `false`) and completed at
the end, so an interrupted run still shows its settings; a run that stops with an error adds `finished: false`,
`error` (its type and message) and the kernel counts so far (`kernels.ran`). The file is strict JSON: an infinite value is written as the
string `"inf"` (a `tau` of `"inf"` means no variation between a contig and its bin), a missing one as `null`.
`epirefine rerun --params params.json --out <dir>` repeats the run from this file (see `docs/usage.md`).

**Provenance.**

- `version`, `command_line`, `start_time` (UTC), `finished`, `python` (its version), `out`, `cache_dir` (`null`:
  `<out>/cache`), `workers` (the number used), `temp_dir` (the folder of the temporary table files, `--temp-dir`;
  `null`: `<out>`);
- `inputs`: for each of `assembly`, `pileup`, `contig_bin` and `motifs`, its `path`, `size` in bytes and
  modification time `mtime` (seconds since 1970);
- `commit` (`commit`, `source`: `git` with `dirty` true when tracked files differ from the commit, or the
  environment variable `EPIREFINE_COMMIT`, or a file `COMMIT` next to `src` as a snapshot made with `git archive`
  can carry; `null` for an installed release, whose `version` names it);
- `libraries` (versions of `python`, `numpy`, `scipy`, `polars`);
- `kernels` (`available`: the install has the compiled kernels; `used`: the run chose them (decided once at its
  start, for the main process and every worker); `reason` the numpy code was chosen, or `null`; `EPIREFINE_RS`,
  `EPIREFINE_RS_NO_SIMD`; for a compiled install also `commit` the module was built from, `cpu_path` (`avx2_fma` or
  `generic`), `api`, `features` and `module_sha256` of the loaded compiled module; at the end of the run `ran`: per
  kernel (`pileup_reader`, `motif_scan`, `npmle`, `bin_loglik`, `chunk_terms`, `cell_sums`, `score_sums`) the calls
  that took the compiled code (`compiled`) and the calls that took the numpy code (`numpy`), the workers' included
  (some options and small tables take the numpy code also when `used` is true), and `missing_features`);
- `platform` (`system`, `release`, `machine`, `libc` and `libc_version`: the C library, whose maths functions set the
  last bits of exp and log);
- `threads` (`workers`, `cores` this process may use, `model_threads`, `polars_threads`, and the variables
  `omp_num_threads`, `openblas_num_threads`, `mkl_num_threads`, `polars_max_threads` as set, or `null`);
- `seeds` (`seed`, `calibration_seed`, and `bootstrap`, the fixed seed of the resampling in the calibration tests);
  `seed` and `calibration_seed` also at the top level (`calibration_seed` is the seed the pieces were drawn with:
  `--calibration-seed`, or `seed` when it was not given);
- at the end: `elapsed_s` (wall time in seconds), `peak_rss_gb` (peak memory of the main process, GB) and
  `peak_rss_worker_gb` (of the largest worker process).

**Every option as given**, under the name of the command-line option with underscores. Decisions:
`contamination_threshold`, `min_methylome_difference`, `include_threshold`, `include_threshold_unmethylated`,
`max_redundancy`, and `p_unknown_given` (`--p-unknown`). Which contigs are judged: `min_evidence`,
`min_contig_coverage`, `min_contig_length`, `min_cov`. Motifs: `group_weights` (false with `--no-group-weights`),
`motif_noise_filter` (false with `--no-motif-noise-filter`), `min_jaccard`, `noise_max_off_share`, `shared_sites`,
`complement_pairs`, `strand_sites`, `group_sample_bp` (`--group-sample-mb` in bp). Long contigs: `regional_model`
(false with `--no-regional-model`), `chunk_size`. Model and calibration: `variation`,
`tau_given` and `temperature_given` (`--tau` and `--temperature`; `null` when not given), `allow_sharpening`,
`do_calibrate` (false with `--no-calibrate`), `frag_lengths`,
`frags_per_bin`, `frag_budget`, `contam_eps`, `grid_size`. Extra output: `write_motif_fit` (`--motif-fit`).
Memory: `tables_in_memory` (`--tables-in-memory`). The rule
variants: `regional_leave_out`, `regional_withdraw`, `regional_tempered`, `gate_leave_out`, `gate_unweighted`,
`eps_gated`, `unknown_prior`, `unknown_prior_strength`, `two_state_warning`, `redundancy_order`,
`siteless_bin_guard`. The calibration-core options: `temperature_law`, `evidence_temperature`.
`relatives` and `unknown_share` hold the values used (`community-only` and `t1` by default), not `null`.

**Fitted values.**

- `tau` (the variation parameter used: one value, or one per motif group), `temperature` (used) and
  `fitted_temperature` (the best value of the grid, also below 1), `n_fragments` (calibration pieces drawn);
- `unknown_eta`, `unknown_lam`: the model of a genome without a bin in use (`unknown_lam` = 1: a community draw);
- `p_unknown`: the share of unbinned contigs from genomes without a bin used in the decisions (estimated, or
  `--p-unknown`); `p_unknown_t1`: the share estimated at temperature 1 (the same value by default; `null` when
  `--p-unknown` is given; absent with `--relatives switch`);
- `contamination_rate`: the estimated share of foreign contigs among the testable binned contigs, estimated before
  the methylome-difference requirement withdraws flags (so it can be well above the share flagged), `null` with
  one bin; `contamination_detection`: `done`, or `skipped: one bin only`;
- `temperature_by_sites`: `null`, unless the temperature depends on the number of sites (`--temperature-law
  sites`): then the law (`law`, `a`, `b`, `n_ref`, `t_min` and the temperatures `T_at_300_sites` ...
  `T_at_30000_sites`);
- `relatives_statistic` (since 0.12; `null` without calibration or with one bin): the relatives test of 0.11 as a
  report: `gain` (mean log-loss gain per calibration piece of the best model with relatives over the best
  community-only one, each at its own temperature), `se` (its standard error, `se_over_bins` true when resampled over
  bins), `statistic` = `gain` / `se` (`null` when `se` is 0), `relatives_eta`, `relatives_lam`,
  `relatives_temperature`, `community_temperature`, and `switch_choice` (what the 2-SE switch of 0.11.1 would have
  chosen). It changes no score or call;
- `unknown_model_forced` (only when the switch would have chosen the relatives): its `switch_eta`, `switch_lam`,
  `switch_temperature` and the `community_temperature` used;
- `calibration_guard` (since 0.12): `fired` (true when the calibrated temperature is at the cap of the grid; the run
  then makes no call), `reason`, `temperature`, `cap` (20), `pieces` (calibration pieces used), `test_b_pieces`
  (pieces in test B) and `fallback` (`"no calls"` when fired); `null` without calibration, with `--temperature` or
  with the deprecated `--relatives switch`. With `--temperature-law sites` the guard reads the law, whose temperature
  the decisions use: `fired` when the law's temperature at 1,000 sites is at the upper bound of its search,
  `temperature` is that value and `cap` 100. See the FAQ, "Known limitations".

## Outputs of non-default options

- **`unbinned_prior.tsv`** (with `--unknown-prior unbinned`): per motif group (`group`, `motifs`), how often it is
  methylated among the bins (`methylated_among_bins`) and among the unbinned contigs that get a decision
  (`methylated_among_unbinned`), and how many of those unbinned contigs have sites in the group
  (`unbinned_contigs_with_sites`).
