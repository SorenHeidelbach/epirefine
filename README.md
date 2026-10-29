# pbinner

Bin refinement from nanopore methylation, built on Nanomotif output:

- **contamination detection:** flags binned contigs whose methylation doesn't fit their bin;
- **contig inclusion:** places unbinned contigs into the bin they belong to.

Every decision comes with a calibrated posterior, including an explicit "unbinned genome" class, so
contigs of genomes that have no bin stay out of bins. Outputs include Nanomotif-compatible files
(`bin_contamination.tsv`, `include_contigs.tsv`, `new_contig_bin.tsv`,
`decontaminated_contig_bin.tsv`). Calibration is estimated inside each sample from pseudo-contigs
cut from binned contigs.

- Method: [docs/method.md](docs/method.md).
- Performance work: [docs/scalability.md](docs/scalability.md).
- Changes per version: [CHANGELOG.md](CHANGELOG.md).
- Development rules: [CLAUDE.md](CLAUDE.md).

## Repository layout

| Folder | Content |
|---|---|
| `src/pbinner/` | The package: methylation-based contamination detection and contig inclusion, nothing else |
| `tests/` | Unit tests (`.venv/bin/python -m pytest -q tests`) |
| `workflow/` | Snakemake workflow: pbinner, Nanomotif, CheckM2 and, for samples with Hi-C reads, the Hi-C validation; R comparison script |
| `hic/` | Hi-C validation, separate from pbinner: contig contacts, Hi-C bin calls, scoring of refinement decisions ([hic/README.md](hic/README.md)) |
| `benchmarks/` | Benchmark suite (`benchmark_suite.py`, `suite.json`) and the scripts that build its scenarios |
| `dev/` | Development tooling: git hooks, speed prototypes |
| `docs/` | Method and performance notes |

## Install

```bash
cd plasmid_binner
uv venv && uv pip install -e ".[dev]"
.venv/bin/pytest -q
```

## Running pbinner

| Flag | File | Notes |
|---|---|---|
| `--assembly` | assembly FASTA | |
| `--pileup` | `modkit pileup` bedMethyl (`.bed`/`.bed.gz`) | Space-separated (older modkit) or all-tab formats. Only rows at motif sites are kept while streaming. |
| `--contig-bin` | `contig<TAB>bin`, no header | The bins to refine |
| `--motifs` | Nanomotif `bin-motifs.tsv` | Only the **union** of motifs is used (+ `motif_complement`). Which bin carries which motif is re-estimated from the data. |

```bash
pbinner run --assembly asm.fasta --pileup pileup.bed --contig-bin contig_bin.tsv \
    --motifs nanomotif/bin-motifs.tsv --out pbinner_out --workers 16
```

**Decision thresholds:**
- `--contamination-threshold` (0.5): flag a binned contig when P(foreign) is at least this,
- `--min-methylome-difference` (0.3): ... and its 95% credible interval for some motif group's
  methylated fraction lies at least this far from its bin's level (an effect size, so that many
  weak site-level differences don't add up to a flag);
- `--include-threshold` (0.95): include an unbinned contig when one bin's posterior is at least this;
- `--max-redundancy` (0.3): don't include a contig when this share of its k-mers is already in the
  bin (assembly duplicates; status `redundant`).

**Model options:** `--variation beta|logit` (how a contig's methylation varies around its bin's;
`--tau` overrides the calibrated value), `--temperature`, `--allow-sharpening` (default: the
temperature is clipped at ≥ 1), `--p-unknown`, `--contam-eps`, `--chunk-size`.
`--regional-inclusion` (experimental, off by default) also includes long unbinned contigs whose
50 kb windows match a bin when the whole contig doesn't.

**Running:**
- `--workers N` sets the processes for the pileup and table passes (default: CPUs, at most 16).
- `--cache-dir` holds the site cache (default `<out>/cache`). A rerun with the same assembly, pileup
  and motifs reuses it and skips the pileup pass.
- `--frag-budget` (default 6000) is the total number of calibration fragments, spread over bins and
  lengths with at most `--frags-per-bin` each.
- `--motif-fit` also writes each contig's methylated fraction per motif group next to its bin's.

The pileup must have each contig's rows together (as modkit writes it). Uncompressed pileups are read
in parallel byte ranges; gzipped ones by a single worker.

## Outputs

| File | Content |
|---|---|
| `contig_decisions.tsv` | Per contig: `status`, given and new bin, `p_contaminant`, `methylome_difference`, `redundancy_with_bin`, best alternative bin and its posterior, MAP, `p_unknown`, 95% credible set |
| `bin_contamination.tsv`, `include_contigs.tsv`, `new_contig_bin.tsv`, `decontaminated_contig_bin.tsv` | Nanomotif-compatible results |
| `host_profiles.tsv` | Per bin × motif group: methylated fraction (posterior mean, 95% interval) |
| `host_chunk_outliers.tsv` | P(chunk is foreign to its bin) for every 50 kb chunk of a binned contig |
| `detection_model.tsv` | Learned caller behaviour per motif group (μ0, μ1, κ0, κ1) |
| `motif_groups.tsv` | Which motifs were merged into which feature, with site and read counts |
| `motif_fit.tsv` | With `--motif-fit`: per contig × motif group, observed methylated fraction (95% CI) and the bin's |
| `calibration/` | Pseudo-contig calibration: accuracy, ECE, credible-set coverage per length, temperature and τ fits |
| `params.json` | Fitted parameters and settings |

**Status** in `contig_decisions.tsv`:

| Status | Meaning |
|---|---|
| `kept` | Binned contig that fits its bin |
| `contaminant` | Binned contig flagged as foreign; removed from its bin |
| `moved` | Flagged contaminant that confidently belongs to another bin; moved there |
| `sole_contig_of_bin` | Only contig of its bin: no leave-one-out profile, not judged |
| `included` | Unbinned contig placed into a bin |
| `redundant` | Would be included, but its sequence is already in the bin |
| `unbinned` | Unbinned contig left out (uncertain, or its genome has no bin) |

## Workflow: refinement checked with CheckM2 and Hi-C

`workflow/Snakefile` runs, per sample:
1. Nanomotif (`detect_contamination` + `include_contigs`) and pbinner, on the same bins, motifs and
   pileup;
2. writes the original and refined bin sets as FASTA;
3. scores each set with CheckM2;
4. compares them in R (`workflow/scripts/compare_bins.R`): `comparison/summary.tsv`,
   `comparison/bins.tsv` and `comparison/bin_refinement.pdf` (bins per quality tier, sequence moved,
   an arrow per changed bin from original to refined quality);
5. for samples with Hi-C reads: maps them with chromap, calls each contig's Hi-C bin and scores both
   methods' flags and inclusions against it (`hic_validation/summary.tsv`); see
   [hic/README.md](hic/README.md).

```bash
# once: micromamba + a snakemake environment (see the dev log's DATA.md for this server)
snakemake -s workflow/Snakefile --configfile workflow/config.yaml --use-conda --cores 16
```

- Samples go in `workflow/config.yaml`: assembly, modkit pileup, the binner's contig-to-bin table and
  Nanomotif's `bin-motifs.tsv`. Optional per sample: `cache_dir` (pbinner site cache), `methods`
  (bin sets to compare), `hic_r1`/`hic_r2`, `hic_index`, `hic_max_pairs`.
- The first run creates the conda environments (Nanomotif 1.1.2, CheckM2 1.1.0, R, chromap) and
  downloads the CheckM2 database (3 GB) to `checkm2_db_dir`.
- pbinner itself runs from the repository's `.venv`.

## Benchmarks

`benchmarks/benchmark_suite.py --note "<change>"` runs the scenarios in `benchmarks/suite.json`
(ZymoOral with planted binning errors at two fragment lengths, a simulated community; `--all` adds
human_gut) and records accuracy, wall time and peak memory in the dev log.

## Known limitations

- **One bin per contig.** A contig assembled across two genomes is not split.
- **Robust profiles assume a contaminant looks like a "random community genome".** A bin that is a
  50/50 merge of two genomes is not resolved.
- **Motif sites shared by two motif groups are dropped** (about 20% of positions on human_gut).
- **No coverage, co-abundance or taxonomy evidence**: methylation only.
- **Dropped constant:** the binomial coefficient is omitted from likelihoods, so reported
  log-likelihoods are relative. Bayes factors and posteriors are unaffected.
