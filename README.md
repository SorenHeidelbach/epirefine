# pbinner (v2)

Probabilistic plasmid–host assignment from nanopore methylation, built on Nanomotif output.

For each plasmid contig it reports:
- a posterior over host bins **plus an explicit "unknown host" class**, with a 95% credible set
- a status: `assigned` / `ambiguous` / `host_not_binned` / `uninformative`
- which motifs the plasmid methylates with **its own MTase**
- motifs its assigned host fails to explain (misfit)
- an **assignability** score: how well this contig *could* be assigned at all, computed from site positions and coverage only

Calibration is estimated inside each sample from pseudo-plasmids cut from host contigs. Host
profiles are robust to contaminating contigs in bins.

The full method, with equations and the reasoning behind each choice, is in [docs/method.md](docs/method.md).
The design history, roadmap and session log are in [docs/conversation.md](docs/conversation.md).

## Install

```bash
cd plasmid_binner
uv venv && uv pip install -e ".[dev]"
.venv/bin/pytest -q
```

## Inputs

| Flag | File | Notes |
|---|---|---|
| `--assembly` | assembly FASTA | |
| `--pileup` | `modkit pileup` bedMethyl (`.bed`/`.bed.gz`) | Space-separated (older modkit) or all-tab formats. Only rows at motif sites are kept while streaming. |
| `--contig-bin` | `contig<TAB>bin`, no header | Host bins |
| `--motifs` | Nanomotif `bin-motifs.tsv` | Only the **union** of motifs is used (+ `motif_complement`). Which bin carries which motif is re-estimated from the data. |
| `--plasmids` | one contig per line, or geNomad `*_plasmid_summary.tsv` | Plasmid candidates: scored with plasmid parameters and removed from host bins. Default: all unbinned contigs. |
| `--mtases` (optional) | TSV: `contig`, `mod_type`, `motif` | MTase genes, e.g. from MTase-linker. Column names are matched flexibly (`gene_id`, `detected_motif`, `motif_pred`, `6mA/5mC/4mC`, ...). **Annotate host contigs too**, so the reliability of annotations (ρ) can be estimated on the bins. |

```bash
pbinner run --assembly asm.fasta --pileup pileup.bed.gz --contig-bin contig_bin.tsv \
    --motifs nanomotif/bin-motifs.tsv --plasmids genomad/asm_plasmid_summary.tsv \
    --mtases mtase_linker/mtases.tsv --out pbinner_out
```

**Options:**
- `--score-all` scores every contig (binned ones leave-one-out).
- `--allow-sharpening` lets calibration use a temperature < 1; by default it is clipped at 1.
- `--delta`, `--tau-p`, `--tau` and `--temperature` override fitted values.
- `--workers N` sets the processes for the pileup and table passes (default: CPUs, at most 16).
- `--cache-dir` holds the site cache (default `<out>/cache`). A rerun with the same assembly, pileup
  and motifs reuses it and skips the pileup pass.
- `--frag-budget` (default 6000) is the total number of calibration fragments, spread over bins and
  lengths with at most `--frags-per-bin` each.

The pileup must have each contig's rows together (as modkit writes it). Uncompressed pileups are read
in parallel byte ranges; gzipped ones by a single worker.

## Report

```bash
pbinner report --run pbinner_out [--truth truth.tsv] [--title "My sample"]
```

Writes `pbinner_out/report.html`, a single self-contained interactive page (open it in any browser):
- summary tiles
- a filterable, sortable plasmid table
- the methylation fingerprint matrix: host bins with their assigned plasmids, own-MTase and misfit marks
- the posterior matrix (true host outlined when `--truth` is given)
- calibration: reliability, performance by fragment length, assignability vs accuracy, τ fit
- model diagnostics: caller μ0/μ1 per motif group, the δ × τ_p likelihood surface, bin-contamination strips

Every chart has hover details and a table view, and the page follows light/dark mode.

## Outputs

| File | Content |
|---|---|
| `assignments.tsv` | Per contig: `status`, MAP host, posterior, `p_unknown`, 95% credible set, runner-up, observed log BFs; `expected_log_bf_vs_nearest_bin` / `nearest_competitor` / `expected_log_bf_vs_unknown` (assignability of the MAP host); `detectability` (would a binned host have been recognised?); `own_mtase_motifs`; `misfit_motifs`; `mtase_annotated` |
| `motif_fit.tsv` | Per contig × motif group: observed φ (95% CI), φ expected under the assigned host (95% CI), misfit log BF, P(own MTase), annotation level |
| `posteriors.tsv` | Full posterior per contig × candidate |
| `host_profiles.tsv` | Per bin × motif group: φ posterior mean, 95% interval, P(φ > 0.5). `_population_prior_` is the learned prior. |
| `host_chunk_outliers.tsv` | P(chunk is foreign to its bin) for every 50 kb host chunk. Works as bin-contamination QC. |
| `detection_model.tsv` | Learned caller behaviour per motif group (μ0, μ1, κ0, κ1) |
| `motif_groups.tsv` | Which motifs were merged into which feature, with site and read counts |
| `plasmid_param_fit.tsv` | Marginal likelihood over the (δ, τ_p) grid |
| `calibration/summary.tsv` | Pseudo-plasmid accuracy, ECE, log loss, credible-set coverage per length. Rows: uncalibrated / fitted temperature / used. Test A = host binned; test B = host removed (should become `unknown`); `B_distinct_*` = host removed and no lookalike bin present. |
| `calibration/assignability.tsv` | Accuracy of pseudo-plasmids binned by their *expected* log BF. Validates the assignability score. |
| `calibration/lookalikes.tsv` | Bins whose methylation repertoire matches another bin |
| `calibration/fragments.tsv`, `tau_fit.tsv`, `temperature_fit.tsv`, `reliability.tsv` | Details |
| `params.json` | τ, temperature (used and fitted), δ, τ_p, ρ estimates, settings |

### Status

| Status | Meaning |
|---|---|
| `assigned` | The 95% credible set is a single bin |
| `ambiguous` | The MAP is a bin, but the 95% set contains other bins and/or `unknown` (e.g. lookalike hosts) |
| `host_not_binned` | MAP is `unknown`, and detectability ≥ 3: if any binned host were the host, it would have been recognised |
| `uninformative` | MAP is `unknown`, and detectability < 3: the contig has too few or too uninformative sites |

## Model

```
k_i | n_i, s_i        ~ BetaBinomial(n_i, mu_{s,m}, kappa_{s,m})             caller model per motif group
s_i | phi_um         ~ Bernoulli(phi_um)                                    site methylated?
w_um                 ~ Bernoulli(rho_um)                                    unit has its own MTase for m
level_um             = phi_own ~ q_m          if w_um = 1                   'present' part of pi_m
                     = phi_hm                 if w_um = 0
phi_um | level       ~ Beta(d*level*tau, (1 - d*level)*tau)                 chunks: d = 1, tau = tau_c
                                                                            plasmids: d = delta, tau = tau_p
phi_hm               ~ pi_m        (grid NPMLE across bins, shrunk to mod type, uniform floor)
chunk k of bin h     ~ (1 - eps) host h + eps random community host         robust profiles (EM)
host                 ~ (1 - p_unknown)/H per bin, p_unknown for "unknown"
P(unit | unknown)    = prior predictive under pi_m
```

**Fitting:**
- **Caller model:** empirical Bayes, a two-component beta-binomial mixture per group, shrunk to a mod-type-level fit.
- **Host profiles:** robust EM over 50 kb chunks. Foreign chunks are down-weighted using leave-chunk-out profiles.
- **τ_c:** leave-one-out marginal likelihood of pseudo-plasmids.
- **Temperature:** log loss on test A plus distinct test B, clipped at ≥ 1.
- **δ and τ_p:** type-II ML over plasmid candidates (≥ 5 with ≥ 20 sites), marginalising over the host.
- **ρ:** estimated on bins from annotation vs observed profiles (motif-level, mod-type-level). The base rate for unannotated groups is a parameter.

**Sites:**
- A palindrome's two strands form one site.
- A motif and its reverse complement are canonicalised.
- Motifs of the same mod type with ≥ 50% site containment are merged into one group.
- Records matching motifs of **different** groups (e.g. the A of TCGANTC) are dropped, so one MTase's signal can't leak into another group.

**Assignability:** expected log Bayes factors on the "Asimov" dataset, where read counts are replaced by their expectation under the hypothesised host. The expected per-site log-likelihood is linear in φ, so two tables per unit suffice for any host.

## Synthetic benchmark

```bash
pbinner simulate --out sim/data [--seed N]
pbinner run --assembly sim/data/assembly.fasta --pileup sim/data/pileup.bed --contig-bin sim/data/contig_bin.tsv \
    --motifs sim/data/bin-motifs.tsv --plasmids sim/data/plasmids.txt --mtases sim/data/mtases.tsv --out sim/out
pbinner evaluate --assignments sim/out/assignments.tsv --truth sim/data/truth.tsv
```

**The simulated community (9 hosts, 31 plasmids) includes:**
- nested motifs (GATC/RGATCY)
- two hosts with identical repertoires
- a host with no methylation
- an unbinned host
- four plasmids with their own MTase: annotated with motif, with mod type only, and unannotated
- plasmid under-methylation (0.85 vs 0.96) with per-plasmid variation
- two contaminating contigs in bins
- a Type I motif with its complement
- MTase annotations with missing motifs and one inactive MTase

**Results (seeds 1 and 2):**

| | Seed 1 | Seed 2 |
|---|---|---|
| Plasmids: true host in 95% set | 31/31 | 31/31 |
| Plasmids: confident (≥ 0.95) wrong | 0 | 0 |
| Plasmids: MAP correct | 27/31 | 26/31 |
| Own-MTase plasmids assigned to the right host, extra motif flagged | 4/4 | 4/4 |
| Contaminating chunks flagged | 4/4 | 5/5 |
| Fitted δ / τ_p (truth ≈ 0.885 / 60) | 0.90 / 30 | 0.85 / 30 |
| Pseudo-plasmid accuracy / 95% set coverage | 0.85 / 1.00 | 0.87 / 1.00 |
| Host removed, no lookalike → `unknown` | 94% | 94% |

**MAP misses:**
- The H3/H4 lookalike plasmids: status `ambiguous`, both bins in the set, expected log BF vs nearest ≈ 0.
- Short H9 (unbinned host) plasmids whose only distinguishing motif has no sites: status `ambiguous`, with `unknown` in the set.

**Assignability is validated:** pseudo-plasmids with expected log BF < 0.5 are about 50% accurate; those above 10 are about 99.9% accurate.

## Known limitations, and next steps

- **Temperature clipped at 1:** the pseudo-plasmid posteriors are slightly underconfident (ECE about 0.04 vs 0.02 with the fitted temperature). This is deliberate (see `docs/conversation.md`).
- **One host per contig.** Mixtures and read-level deconvolution come in v4.
- **One δ and τ_p for all plasmids and groups.** Mod-type-specific δ is a possible extension.
- **Robust profiles assume a contaminant looks like a "random community host".** A bin that is a 50/50 merge of two genomes is not resolved.
- **No coverage, co-abundance or taxonomy priors yet** (v3).
- **Not yet run on real data.** The Zymo mocks are the next step.
- **Dropped constant:** the binomial coefficient is omitted from likelihoods, so reported log-likelihoods are relative. BFs and posteriors are unaffected.
