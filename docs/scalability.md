# pbinner: scalability evaluation and speed-up plan

This document evaluates how pbinner v2 scales, measured on this server (16 cores, 62 GB RAM), and
lists concrete ways to make the critical steps faster: by lowering their complexity or by more
efficient code. Each speed-up is estimated either from a prototype timed here or from a scaling
argument; the table says which.

## 1. Target size

| Dimension | Target | ZymoOral (for reference) |
|---|---|---|
| Contigs | 50,000 | 63 |
| Assembly | 1.7 Gb (real human-gut assembly used for timing) | 25 Mb |
| Motifs (union over bins) | 1,000 (≈ 800 groups after merging) | 43 (33 groups) |
| Bins | 400 (and 1,000) | 12 |
| Plasmid candidates | 5,000 | 3 |
| Pileup | ~120 GB | 4 GB |

## 2. Measurements

**Benchmark script:** `scripts/scale_components.py` (removed in 0.5.0; the benchmark suite now measures speed). It times pbinner's real functions at the
target size, scaling linearly from a timed fraction where a full run would exceed the machine.

**Inputs:**
- *Sequence-dependent stages:* timed on the real 1.73 Gb human-gut assembly (18,351 contigs).
- *Motif set:* grown from ZymoOral's Nanomotif 1.1.3 motifs to 1,000, keeping their length,
  modification-type and bipartite mix.
- *Pileup throughput:* measured on ZymoOral's real 4 GB pileup.

Small end-to-end runs: 10 bins ran in 92 s at 1.2 GB peak; 20 bins used 3.3 GB peak.

| Stage | Time at target (400 bins) | Peak memory | Source |
|---|---|---|---|
| Read assembly | 15 s | 2 GB | measured |
| **Motif search** (regex) | **15.8 h** | — | 5 motifs × 150 Mb measured, scaled |
| **Per-(motif, contig) overhead** | **3.1 h** | — | 220 µs per pair measured, × 50k × 1,000 |
| **Records table** | — | **≈ 200 GB** | 2.88 records/kb/motif × 40 B, measured |
| **Motif grouping** (pairwise containment) | **≈ 10 h** | — | 0.15 s per pair measured, × ~250k pairs |
| Pileup stream + join | 15 min per 120 GB | records table | 0.14 GB/s measured |
| Unit likelihoods (30k chunks) | 42 min | 12 GB ($K \cdot M \cdot G$, ×2 copies) | 200 chunks measured, scaled |
| Model build (population prior EM + robust EM) | ≈ 30 min per build | — | measured on 1/3 of chunks |
| ↳ population prior EM, one call | 2–5 min (×45 calls) | — | measured |
| **Calibration** (8 τ × 72k fragments) | **11–18 h** | **369 GB** of $A, P_w$ tensors | measured parts, combined |
| Plasmid scoring (5k × 400 bins) | 3 min | 26 GB of tensors | 390 plasmids measured, scaled |
| Detectability | 1.7 h | — | 5 hosts measured, × 400 |
| δ × τ_p grid (42 points) | 2 h | as scoring | 42 × scoring |

**Verdict.** At the target size the current code would need about two days of compute and several
hundred GB of RAM, so it is infeasible. The bottlenecks are not where big-O alone points, though:
- **Cheap despite bad big-O:** caller-model fitting, robust responsibilities, and the plasmid-scoring
  matrix products. Each takes minutes.
- **Expensive:** motif search, the in-memory records table, pairwise motif grouping, and a
  calibration whose size grows with the number of bins.

## 3. Prototype timings of the alternatives

Timed with `dev/speedup_prototypes.py` on this machine (single thread):

| Operation | Current | Alternative | Measured speed-up |
|---|---|---|---|
| Motif matching, per motif and strand, 100 Mb | Python regex 1.38 s | 2-bit encoded sequence, one vectorised pass per *motif shape* with `isin` over all motifs of that shape: 0.12 s per motif-strand | **11×** (more when many motifs share a shape) |
| Pairwise motif containment | `intersect1d` 0.15 s per pair × 250k pairs | Shared-record counting: one pass over records found in a sequence sample, counting motif pairs per record | O(M²·R/M) → O(R_sample): **10 h → seconds** |
| Population prior EM, one call (800 groups, 400 bins, 300 iterations) | 121 s (Python loop over groups) | Vectorised over groups (`einsum`) 13 s; stopping at convergence (~50 iterations) **2.2 s** | **9× / 55×** |
| Site likelihood per site × 64 grid points | `logaddexp` float64 1.65 µs | Lookup in a precomputed (group, n, k) table, float32: 0.10 µs | **16×** |

## 4. Speed-up plan, by stage

Each change is labelled **[C]** if it reduces complexity or the size of the problem, or **[E]** if it
is more efficient code for the same computation.

### 4.1 Motif search: 19 h → ≈ 10 min

1. **[E] Scan once per motif shape, not once per motif.** Encode the assembly once as 2 bits per base.
   - *Shape* = the positions of a motif's non-N bases. Motifs that share a shape (all 5-mers; all
     `XXXNNNNNXXXX` bipartites) are matched in one vectorised pass: build the packed integer of the
     bases at the shape's positions for every genome position, then `isin` against the codes of all
     motifs with that shape.
   - Degenerate IUPAC bases (R, Y, W, …) expand into their concrete variants, which adds codes but
     not passes.
   - With ~50–150 distinct shapes for 1,000 motifs, about 100–300 passes over 1.7 Gb replace 2,000
     regex scans.
   - *Estimate:* 1–3 h single-threaded (measured 11× per motif-strand plus the shape sharing).
2. **[E] Concatenate contigs** (with separators) before searching and map positions back with
   `searchsorted`. This removes the 3.1 h of per-(motif, contig) Python overhead.
3. **[E] Process contig batches in parallel** (16 processes). *Estimate:* ≈ 5–12 min.

### 4.2 Records table: 200 GB → a few GB (streaming)

4. **[C] Never materialise all records.** Process the assembly in contig batches of ~50 Mb. For each
   batch: find its records, read the matching pileup rows, aggregate to sites, and keep only compact
   sufficient statistics.
   - Pileups are sorted by contig, so they can be read sequentially. A bgzip + tabix-indexed pileup
     (as in ZymoHMW) also allows random access and parallel reading.
5. **[C] Two light passes instead of one heavy one.**
   - *Pass 0:* fit the caller model on a ~5% random sample of contigs. That's millions of sites per
     group, plenty for four parameters.
   - *Pass 1:* stream everything once, adding each site's term directly into the unit
     log-likelihood arrays.
   - Sites are never stored, except for the sampled calibration fragments (§4.5). Peak memory is
     then dominated by the chunk log-likelihoods: $K \cdot M \cdot G \cdot 4$ B ≈ 6 GB in float32 at
     the target.

### 4.3 Motif grouping: 10 h → seconds

6. **[C] Count shared records instead of intersecting all pairs.**
   - Containment is a property of the motif sequences, so estimating it on a sample (e.g. 20 Mb of
     contigs) is accurate.
   - In that sample, sort records by position, find records matched by ≥ 2 motifs, and add 1 to each
     such motif pair's counter. That's O(records in the sample) instead of O(M²) intersections.

### 4.4 Unit likelihoods: 42 min → ≈ 3–5 min

7. **[E] Table lookup instead of transcendental functions.** A site's 64-point log-likelihood depends
   only on (group, n, k). Precompute it once for every distinct (group, n, k) present (a few thousand
   per group; cap $n$ at e.g. 1,000), then gather and add per unit. Measured 16× for the per-site
   term.
8. **[E] float32 and parallel accumulation** (numba, or chunked numpy across processes). This also
   folds naturally into the streaming pass of §4.2.

### 4.5 Model building and calibration: 11–18 h → ≈ 15–25 min, 369 GB → < 2 GB

9. **[E] Vectorise the population-prior EM across groups and stop at convergence**: 121 s → 2.2 s per
   call (measured).
10. **[E] Warm-start the EM** from the previous robust iteration or τ value. Later calls then need
    only a few iterations.
11. **[C] A fixed calibration budget instead of 180 fragments per bin.**
    - Today the fragment count grows with the number of bins (72k at 400 bins, 180k at 1,000).
      Calibration quality depends on the number of fragments, not bins.
    - About 6,000 fragments (1,000 per length), stratified across bins, measures accuracy and ECE to
      about ±0.01.
    - This reduces the work 12× at 400 bins and 30× at 1,000.
12. **[C] Score without per-group tensors.** Calibration needs only $S_{uh}$, not
    $A_{uhm}$ and $P_w$. Accumulating $S$ group by group (the own-MTase mixture is applied inside the
    loop) cuts memory by a factor of $M$: 369 GB → 6,000 × 400 × 8 B ≈ 20 MB.
13. **[C] Fewer τ evaluations.** Use a golden-section search over $\log\tau$ (≈ 5 evaluations) and
    reuse the robust chunk weights across τ, since they hardly depend on it. That replaces eight full
    rebuilds.
14. **[E] Vectorised leave-one-out.** Fragments overlap 1–3 chunks, so compute each chunk's
    leave-one-out bin profile once ($K \times M \times G$) and score all fragments of that chunk
    together. This replaces a Python loop over fragments.

### 4.6 Plasmid scoring: 3 min / 26 GB → ≈ 1 min / < 1 GB

15. **[C] Two-stage scoring with host pruning.**
    - *Stage 1:* score all bins without the own-MTase term; this is one matrix product per group and
      cheap.
    - *Stage 2:* keep only hosts within a wide margin of the best (e.g. 50 nats) plus "unknown", and
      compute own-MTase terms, $P_w$, misfit and per-group outputs for those only. Typically 1–10
      hosts survive.
    - Memory goes from $U \cdot H \cdot M$ to $U \cdot K_{\text{top}} \cdot M$.
    - The pruning margin can be made safe with an upper bound on how much the own-MTase term can
      raise a host: at most $\sum_m \max(0, \log\rho_{um} + O_{um} - A_{uhm})$.

### 4.7 Detectability: 1.7 h → ≈ 1 min

16. **[C] Compute it only where it's used.** Detectability only decides between `host_not_binned`
    and `uninformative`, i.e. for units whose MAP is "unknown". These are typically a small minority.
17. **[C] Restrict to the hosts that matter.** The maximum over hosts is attained by bins whose
    methylomes differ most from the community prior at the unit's informative motifs. A cheap
    per-group divergence score (without the τ_p marginalisation) picks the top ~20 bins, and only
    those get the exact calculation.

### 4.8 δ × τ_p fit: 2 h → ≈ 1 min

18. **[C] Reuse stage-1 pruning** (each plasmid's top hosts), subsample to ≤ 1,000 informative
    plasmids, and search coordinate-wise rather than over the full 42-point grid.

### 4.9 Pileup reading: 15 min per 120 GB → ≈ 3–8 min

19. **[E] Smaller parsing costs and parallel reading.** Skip the space-to-tab copy when the file is
    already all-tab (detect from the first line), let polars project only the six needed columns, and
    read contig ranges in parallel when the pileup is bgzipped and tabix-indexed. This stage is
    limited by disk I/O, so 2–4× is a realistic ceiling.

### 4.10 Report

20. **[C] Cap the matrices.** At thousands of plasmids and hundreds of bins, the SVG matrices
    (plasmids × bins/motifs) exceed what a browser renders. Show columns only for bins that appear
    in some credible set, page through plasmids, and keep the full data in the table view.

## 5. Expected result

| Stage | Now (target size) | After the plan | Main changes |
|---|---|---|---|
| Motif search | 19 h | 5–12 min | 1–3 |
| Records memory | 200 GB | streaming, < 1 GB | 4–5 |
| Motif grouping | 10 h | seconds | 6 |
| Pileup | 15 min | 3–8 min | 19 |
| Unit likelihoods | 42 min | 3–5 min | 7–8 |
| Model builds | 30 min × 9 | 2–5 min, then ~1 min warm | 9–10, 13 |
| Calibration | 11–18 h, 369 GB | 15–25 min, < 2 GB | 11–14 |
| Scoring | 3 min, 26 GB | ≈ 1 min, < 1 GB | 15 |
| Detectability | 1.7 h | ≈ 1 min | 16–17 |
| δ × τ_p | 2 h | ≈ 1 min | 18 |
| **Total** | **≈ 2 days, several hundred GB RAM** | **≈ 45–75 min on 16 cores, ≈ 10–15 GB RAM** | |

These are estimates. Only the four prototype speed-ups in §3 were measured; the rest follow from the
measured per-unit costs. After implementation, the end-to-end benchmark (`scripts/scale_bench.py`, removed in 0.5.0)
should be rerun at increasing bin counts to confirm them.

**Scaling after the plan:**
- linear in assembly size and pileup size;
- linear in the number of motif shapes (not motifs) for search;
- linear in $U \cdot H \cdot M$ only for the cheap stage-1 matrix products;
- independent of the number of bins for calibration.

### Suggested implementation order (largest gain per effort first)

1. Prior-EM vectorisation + warm start (§4.5, items 9–10): small change, removes hours.
2. Fixed calibration budget + score-only path (11–12): small change, removes the 369 GB.
3. Shared-record motif grouping on a sample (6): small change, removes 10 h.
4. Shape-based motif search + contig concatenation + parallelism (1–3): moderate change.
5. Streaming two-pass architecture (4–5, 7–8): largest change; required for > ~500 Mb assemblies
   with many motifs, because of memory.
6. Pruned scoring, detectability and δ × τ_p (15–18).
7. Report caps (20).

Items 1–3 alone remove the multi-hour model and calibration costs at a few hundred bins. Datasets
whose assembly × motif records exceed memory still need items 4–5.

## 6. Implemented and measured (2026-10-20)

The plan was implemented and run end to end on the real human-gut sample: 1.73 Gb assembly,
18,351 contigs, 369 bins, 775 motifs (367 groups), 14,008 unbinned targets, a 235 GB pileup with
376 M covered motif sites. Server: 16 cores, 62 GB RAM, disk reading at about 50 MB/s per stream
and about 280 MB/s in parallel.

| Stage | Measured | Was (projection, §2) | What changed |
|---|---|---|---|
| Read assembly | 15 s (124 s cold cache) | 15 s | — |
| Motif groups (50 Mb sample) | 2.1 min | ≈ 10 h | items 1, 6: shape matcher, shared-record pair counts |
| Pileup pass → site cache | 12.7 min (disk-bound) | 19 h search + 200 GB records | items 1–5, 19: parallel byte ranges at contig boundaries, motif search per batch, compact Parquet cache (1.3 GB) |
| Caller (detection) model | 8.0 min | — | parallel per group; dominated by 3 pooled mod-type fits |
| Unit tables (second pass) | 3.4 min | 42 min | items 7–8: lookup-table site terms, float32, 16 processes |
| Calibration (8 τ, 6,573 fragments) | 13.8 min (≈ 100 s per τ) | 11–18 h, 369 GB | items 9–12: fixed budget, vectorised NPMLE, summed scores, threaded outlier step |
| Final model | 1.2 min | 30 min | as above |
| δ × τ_p fit (1,000 candidates) | 3.0 min | 2 h | item 18 (subsample) |
| Scoring 14,008 targets × 369 bins | 37 s | 3 min, 26 GB | item 12 (no U×H×M tensors) |
| Assignability | 9.6 min | 1.7 h | items 16–17 (unknown-MAP units, 20-bin shortlist) |
| Outputs | 51 s | — | vectorised tables |
| **Total** | **≈ 56 min from scratch; 45 min reusing the site cache** | **≈ 2 days** | |

**Memory.** The run peaked at 38.9 GB. The cause was found afterwards: the threaded outlier step
used 1,000-chunk batches on 16 threads, about 1 GB of temporaries each. At 200-chunk batches a
human-gut-sized model build peaks at 9.1 GB instead of 28 GB (synthetic arrays of the same shape),
and calibration now holds at most two models. The end-to-end peak with these fixes has not been
re-measured.

**Checks that results did not change.**
- Motif matcher: identical records to the regex version on all 775 human-gut motifs (and a test).
- Streaming sites: identical site table to the in-memory path on ZymoOral (1,212,430 sites).
- Summed scores and vectorised NPMLE: equal to the full-tensor and per-group versions (tests).
- ZymoOral end to end: every MAP unchanged; posteriors differ only through the fragment sample.

**Bugs found while doing this.**
- *Caller-model fit depended on row order.* For two ZymoOral 5mC groups, L-BFGS reached the
  degenerate optimum (μ₁ ≈ 0.5) or the real one (μ₁ ≈ 0.99, log-likelihood higher by 900 to 3,400)
  depending on summation order. Fixed with sorted inputs and extra starting points; degenerate
  groups on ZymoOral went from 10 to 7 of 33.
- *Run-to-run differences with the same seed.* Group numbering and fragment selection depended on
  polars' group order; both are now deterministic (tests), and cache reuse compares motif partitions,
  not labels.

**Still open.**
- Assignability (9.6 min) runs its 20-host detectability loop single-threaded.
- The caller fit (8 min) is limited by the 3 pooled mod-type fits.
- Report caps (item 20) are not done: on human gut, `motif_fit.tsv` is 530 MB, too big to embed in
  the HTML report.
