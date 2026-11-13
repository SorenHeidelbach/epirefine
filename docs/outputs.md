# Outputs

Everything EpiRefine writes into `--out`. For most uses you need only the first two files.

| File | Use it for |
|---|---|
| [`new_contig_bin.tsv`](#new_contig_bintsv) | **The refined binning.** |
| [`contig_decisions.tsv`](#contig_decisionstsv) | **Every contig's decision, with probabilities.** |
| [`decontaminated_contig_bin.tsv`](#nanomotif-compatible-files) | The bins with contaminants removed, nothing added. |
| [`bin_contamination.tsv`, `include_contigs.tsv`](#nanomotif-compatible-files) | Nanomotif-format flags and inclusion candidates. |
| [`bin_evidence.tsv`, `host_profiles.tsv`, `motif_fit.tsv`](#bins-and-motifs) | What each bin's methylation looks like. |
| [`motif_groups.tsv`, `motif_noise.tsv`, `group_weights.tsv`, `detection_model.tsv`](#bins-and-motifs) | How the motifs were grouped, filtered and modelled. |
| [`host_chunk_outliers.tsv`, `calibration/`, `params.json`](#diagnostics) | Diagnostics and fitted parameters. |
| `cache/` | The site cache, reused by reruns (see `--cache-dir`). `manifest.json` holds its key (`inputs`), its part files (`parts`), `format` (2 since 0.12; missing in older caches) and `fingerprints` of the assembly and the pileup; `.lock` is the lock file runs share. |

All tables are tab-separated with a header. Empty fields are written as `""`.

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
| `expected_evidence` | how much methylation evidence (nats) the contig is expected to carry to tell bins apart; below `--min-evidence` (3) it is not judged |
| `coverage` | median reads per motif site; below `--min-contig-coverage` (5) it is not judged |
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
| `no_methylation_evidence` | too little informative methylation to tell bins apart | binned: yes, unchanged |
| `low_coverage` | median coverage below 5 reads per motif site | binned: yes, unchanged |
| `too_short` | shorter than 1 kb | binned: yes, unchanged |

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
- **`decontaminated_contig_bin.tsv`**: `contig`, `bin`: the input bins minus the flagged contigs.

## Bins and motifs

- **`bin_evidence.tsv`**: per bin: `status` (`methylated`, or `no_methylation_detected` for bins without any
  methylated motif group, which need the stricter inclusion threshold), motif groups with sites, methylated,
  confidently unmethylated and undetermined, and the number of sites. Then four columns about the bin as a whole:
  `robust_fit` (`converged`, or `not_converged` when the bin's fingerprint did not settle), `flagged_share` (share of
  the bin's judged sequence that was flagged), `two_state_share` with `two_state_group` (the motif group in which
  the bin's pieces fall into two methylation states, one below and one above 0.5, and the share of the smaller state;
  0 when the pieces agree) and `warning`. A `two_state_share` or `flagged_share` of 0.2 or more means the bin
  probably holds two methylomes: two genomes or strains merged by the binner, or one genome with part of its
  sequence in another state. When the two differ in a single sparse motif nothing is flagged in such a bin (the
  fingerprint settles between them), so this column is then the only sign; look at the bin before trusting it.
- **`unbinned_prior.tsv`** (with `--unknown-prior unbinned`): per motif group, how often it is methylated among the
  bins and among the unbinned contigs.
- **`host_profiles.tsv`**: each bin's fingerprint: per bin and motif group, the methylated fraction (`phi_mean`),
  its 95% interval (`phi_lo95`, `phi_hi95`) and the probability that it is above 0.5.
- **`motif_fit.tsv`** (with `--motif-fit`): per contig and motif group, the contig's methylated fraction with its 95%
  interval next to its bin's. Useful to see why a contig was flagged.
- **`motif_groups.tsv`**: which Nanomotif motifs were merged into which motif group (`group`, `mod_type`, `motifs`),
  with the number of sites and reads.
- **`motif_noise.tsv`**: per motif group, whether it was dropped as basecaller noise (`noisy`), with the bins it
  was measured in, its typical level, and the share of those bins where it is unmethylated (`off_bin_share`, level
  below 0.05; a dropped group that is off in many bins may be a real, weakly detected motif).
- **`group_weights.tsv`**: groups that behave as one enzyme and share one weight (`cluster`, `cluster_size`,
  `weight`).
- **`detection_model.tsv`**: the basecaller model per motif group: the share of reads called modified at
  unmethylated (`mu0`) and methylated (`mu1`) sites, their spread (`kappa0`, `kappa1`), and the methylated share
  across all sites (`pi1`).

## Diagnostics

- **`host_chunk_outliers.tsv`**: every 50 kb piece of every binned contig (`contig`, `start`, `bin`) with its
  probability of being foreign to its bin (`p_foreign`). Pieces likely to be foreign are down-weighted when the bin's
  fingerprint is built.
- **`calibration/`**: how the probabilities were calibrated on pieces cut from the bins: accuracy, calibration error
  and credible-set coverage per piece length before and after calibration (`summary.tsv`, `reliability.tsv`), the
  fitted temperature, overall and by length (`temperature_fit.tsv`, `temperature_by_length.tsv`), the variation
  parameter (`tau_fit.tsv`, `tau_groups.tsv`), the pieces themselves (`fragments.tsv`), how well each bin can be told
  apart (`assignability.tsv`) and bins whose methylation looks alike (`lookalikes.tsv`). Two tables describe the
  probabilities as they are used for the decisions: `false_flag_rate.tsv`, per piece length the share of pieces
  of true members (scored without their own contig) that would be flagged at the run's settings, for all pieces
  and for those kept by the calibration (`false_flag_rate`, `false_flags`; on the development benchmarks
  0 of 5,723 kept member pieces of the training communities would have been flagged, so a clearly higher rate means
  flags in this sample deserve a second look; no threshold for it has been validated), and `reliability_deployed.tsv`, predicted against observed
  probabilities at the estimated priors and the temperature in use.
- **`params.json`**: what the run was made with and what it fitted. Written when the run starts (`finished` is
  `false`) and completed at the end, so an interrupted run still shows its settings.
  - Provenance: `version`, `command_line`, `start_time` (UTC), `python`, `inputs` (path, size in bytes and
    modification time of the assembly, pileup, contig-bin table and motif table), `out`, `cache_dir`, `workers`,
    `seed`, `calibration_seed` (the seed the calibration pieces were drawn with: `--calibration-seed`, or `seed`
    when it was not given).
  - The code and where it ran: `commit` (`commit`, `source`: `git` with `dirty` true when tracked files differ from
    the commit, or the environment variable `EPIREFINE_COMMIT`, or a file `COMMIT` next to `src` as a snapshot made
    with `git archive` can carry; `null` for an installed release, whose `version` names it), `libraries` (python,
    numpy, scipy, polars versions), `kernels` (`available`: the install has the compiled kernels; `used`;
    `reason` the numpy code ran, or `null`; `EPIREFINE_RS`, `EPIREFINE_RS_NO_SIMD`; for a compiled install also
    `commit` the module was built from, `cpu_path` (`avx2_fma` or `generic`), `api`, `features` and `module_sha256`
    of the compiled module), `threads` (`workers`, `cores` this process
    may use, `model_threads`, `polars_threads`, and the variables `omp_num_threads`, `openblas_num_threads`,
    `mkl_num_threads`, `polars_max_threads` as set, or `null`), `seeds` (`seed`, `calibration_seed`, and `bootstrap`,
    the fixed seed of the resampling in the calibration tests).
  - `epirefine rerun --params params.json --out <dir>` repeats the run from this file (see `docs/usage.md`).
  - Every option as given, under the name of the command-line option with underscores (`contamination_threshold`,
    `include_threshold`, `include_threshold_unmethylated`, `min_methylome_difference`, `max_redundancy`,
    `min_evidence`, `min_contig_coverage`, `min_contig_length`, `min_cov`, `chunk_size`, `grid_size`, `contam_eps`,
    `min_jaccard`, `variation`, `frag_lengths`, `frags_per_bin`, `frag_budget`, `allow_sharpening`, and the switches
    `motif_noise_filter`, `group_weights`, `regional_model`, `regional_inclusion`, `do_calibrate`,
    `write_motif_fit`; `group_sample_bp` is `--group-sample-mb` in bp). `p_unknown_given`, `tau_given` and
    `temperature_given` hold `--p-unknown`, `--tau` and `--temperature` (`null` when not given).
  - Fitted values: `temperature` (used) and `fitted_temperature`, `tau` (variation), `unknown_eta`, `unknown_lam`,
    `n_fragments`, `p_unknown` (share of unbinned contigs from genomes without a bin) and `contamination_rate`
    (share of contaminants among binned contigs).
  - The file is strict JSON: an infinite value is written as the string `"inf"` (a `tau` of `"inf"` means no
    variation between a contig and its bin), a missing one as `null`.