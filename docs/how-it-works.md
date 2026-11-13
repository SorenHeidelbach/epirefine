# How it works

EpiRefine turns the reads at motif sites into a probability for every contig and every bin, in six steps. Every
model parameter is estimated from your own sample; nothing is trained elsewhere. The decision thresholds (for
example 0.5 to flag, 0.9 to include, 5 reads per site) are fixed defaults that were chosen during development.

[The idea](#the-idea) · [1. Motif sites](#1-motif-sites) · [2. Basecaller noise](#2-basecaller-noise) ·
[3. A contig's methylation](#3-a-contigs-methylation) · [4. Bin fingerprints](#4-bin-fingerprints) ·
[5. Evidence](#5-evidence) · [6. Calibration](#6-calibration) · [Decisions](#decisions) ·
[Long contigs](#long-contigs) · [Limits](#limits)

## The idea

A bacterium's methyltransferases modify their target motif (for example GATC or CCWGG) on the copies of it in the
cell: on the chromosome, and usually on the plasmids and phages it carries as well (exceptions are under
[Limits](#limits)). Which motifs are methylated is therefore the same along a whole genome and usually differs
between genomes. Nanopore reads record it, base by base.

So every contig carries a methylation signature, and a contig that belongs to a bin should have the bin's. A
contig whose signature contradicts its bin is probably from another genome; an unbinned contig whose signature
matches one bin probably belongs there. This signal is measured separately from GC content, k-mer composition and
coverage, which are what binners use, so it can find errors that binning cannot see. It is not fully independent of
them: decisions need enough coverage, and in bins without methylated motifs the remaining signal (how often the
basecaller calls unmethylated sites modified) follows GC content.

## 1. Motif sites

EpiRefine reads the modkit pileup once and keeps only the motif sites: each occurrence of a motif from Nanomotif's
list, with the number of reads at it (*n*) and the number called modified (*k*). The two strands of a palindrome
form one site. No site is turned into a yes/no call; the counts themselves are used.

Motifs that cover almost the same sites (at least 90% shared, `--min-jaccard`: for example a motif and its reverse
complement, or two spellings of one motif) are merged into **motif groups**. A motif and a more specific variant of
it, such as GATC and RGATCY, share too few sites for that and stay separate groups; a site matched by several groups
counts once, in the most specific one (an RGATCY site counts for RGATCY, not for GATC). Groups whose calls look like
basecaller noise rather than methylation are dropped.

## 2. Basecaller noise

The basecaller is not perfect: at an unmethylated site some reads are called modified, and at a methylated site
some are not. For each motif group EpiRefine learns how many reads to expect called modified at an unmethylated and
at a methylated site (two beta-binomial distributions), using the bins themselves as examples: bins where almost no
reads are modified at the group's sites show the unmethylated pattern, bins where most are show the methylated one.

With this, every site says how much more likely its counts are if it is methylated. For example, at RGATCY on a
fecal sample about 4% of reads are called modified at unmethylated sites and 85% at methylated ones. A site with 9
of 10 reads modified is then strong evidence for "methylated", one with 1 of 10 strong evidence against, and one
with 4 of 10 says little either way.

## 3. A contig's methylation

A contig is many sites. If a fraction φ of its sites in a motif group is methylated, the counts at all of them
together give a curve over φ: how likely the contig's reads are for each possible fraction. The curve peaks near the
contig's true fraction and narrows as it has more sites; a contig with no sites in a group gives a flat curve and
that group says nothing about it.

## 4. Bin fingerprints

A bin's **fingerprint** is, for every motif group, the range of methylated fractions that fit all of its sequence.
It is built from the bin's contigs cut into 50 kb pieces. Bins are themselves contaminated, so each piece is first
checked against the rest of its bin, and pieces that do not fit are down-weighted. When a binned contig is judged,
its own pieces are left out of its bin's fingerprint.

EpiRefine also learns, per motif group, how methylated fractions are spread across all bins of the sample (the
**community**). This is the description of "a genome without a bin": a contig from an unbinned genome has unknown
methylation, which looks like a random draw from the community.

## 5. Evidence

Every contig is scored against every bin and against a genome without a bin: how probable are its reads if its
methylation is drawn from that bin's fingerprint? The scores add up over motif groups, and the difference between
two scores is the evidence for one explanation over the other (in nats; 2.3 nats = 10:1).

- A motif that neither the contig nor the bin methylates adds almost nothing, because most genomes don't methylate
  it either.
- A motif the bin methylates but the contig clearly does not costs a lot, and more with every such site: one motif on
  a 1.1 Mb contig gave −685 nats against its bin.
- The "genome without a bin" explanation keeps a contig from being forced into the least bad bin.

## 6. Calibration

Raw evidence is too confident, because sites on a contig are not independent (neighbouring sites share reads, and
methylation varies a little along a genome). EpiRefine calibrates it on the sample itself: it cuts pieces of 2-100 kb
from binned contigs, scores each as if it were unbinned (with the piece removed from its bin), and also with its bin
removed, where the right answer is "no bin". It then chooses the **temperature**, a single factor that divides all
evidence, that makes these known answers come out with honest probabilities on average. On a fecal development
sample (with a Nanomotif 1.1.3 motif table) the temperature was 5.0, so 100 nats of raw evidence count as 20. The
temperature that would fit each piece length alone rises with length (there from 2.7 at 2 kb to 14.7 at 500 kb), so
one temperature leaves very long contigs overconfident. The temperature is chosen from a grid of 31 values between
0.2 and 20, each about 17% above the one before, and is never below 1 (the pieces are easier to place than real
unbinned contigs, so they can show that the evidence is too confident, but not that it is too timid). The
calibration results are in `calibration/`.

Two checks come with the calibration:

- **The calibration guard.** If the temperature ends at the top of the grid (20), no temperature made the pieces
  come out right, and the run has nothing to calibrate with. EpiRefine then makes no call at all (no flag,
  inclusion or move), says so in a `WARNING` line of the log and records it in `params.json`
  (`calibration_guard`).
- **The relatives statistic.** Every run also scores the pieces under a second description of "a genome without a
  bin": an unbinned relative of one of the bins, with (nearly) the same methylation. The log and `params.json`
  (`relatives_statistic`) report how much better that description explains the pieces, in standard errors. Near 0
  or below, nothing points to such relatives; above about 2, some bins seem to have an unbinned sister genome (see
  [Limits](#limits)). The number is a report: since 0.12 it decides nothing.

## Decisions

From the calibrated evidence EpiRefine computes, for every contig, the probability of each bin and of "no bin". The
share of unbinned contigs that come from genomes without a bin is estimated from the unbinned contigs (at
temperature 1, where it does not move with the temperature), and the share of binned contigs that are
contaminants from the binned contigs. Then:

| Decision | When |
|---|---|
| **Flagged** (`contaminant`) | P(not its bin) ≥ 0.5, and in at least one motif group the contig's 95% interval for its methylated fraction lies ≥ 0.4 from the bin's |
| **Moved** | flagged, and one other bin has P ≥ 0.9 |
| **Included** | unbinned, and one bin has P ≥ 0.9 (≥ 0.95 for a bin without detected methylation) |
| **Not placed** (`redundant`) | 30% or more of the contig's 21-mers are already in that bin |
| **No call** | coverage below 5 (valid calls per motif site, [counted as in outputs.md](outputs.md#coverage-and-5x)), shorter than 1 kb, or too little methylation to tell bins apart (`no_methylation_evidence`) |

The second condition for a flag keeps many small, consistent shifts (variation within one genome) from adding up to
a flag: a different genome usually differs by a whole motif, methylated in one and not in the other. The stricter
threshold for bins without methylation exists because any contig without methylation at the listed motifs matches
such a bin, including contigs of genomes whose motifs Nanomotif never saw.

A **no call** for too little methylation (`no_methylation_evidence`) is decided last, on contigs that would
otherwise be kept in their bin or left unbinned: it says that methylation cannot support that outcome. It never
changes a bin: such a binned contig stays in its bin, an unbinned one stays unbinned, and no other decision depends
on it. The check divides the expected evidence by the temperature, so a sample with a high temperature has more no
calls.

All thresholds can be changed; see [usage.md](usage.md#decisions).

## Long contigs

An assembler sometimes joins two genomes into one contig. For binned contigs of two or more 50 kb windows, EpiRefine
also compares three explanations along the contig: all of it is the bin's, all of it is foreign, or it switches
between the two. A flagged contig that is more likely mixed is kept in its bin with status `mixed`, and the foreign
stretches are reported (`foreign_segments`). Nothing is split.

## Limits

- **Coverage and length.** Decisions need about 5 reads per motif site and enough sites; on a complex fecal sample
  76% of the contigs and 31% of the sequence got no decision, including half of the unbinned plasmids and most
  binned plasmids and viruses.
- **Genomes with the same methylation**, such as very close strains, cannot be told apart. When both have bins,
  EpiRefine keeps several bins in the credible set or leaves the contig where it is. When only one has a bin, the
  contigs of the other are placed into that bin, because a genome without a bin is modelled as a random genome of
  the community, and such contigs fit the bin better (on simulated development sets, 112 such placements over three
  sets). On such samples the temperature also rises, so more short contigs get no call. A relatives statistic
  above about 2 warns of this.
- **The temperature moves in steps** of about 17%, and the number of inclusions can move with it: when the
  calibration bins of a fecal development sample were resampled, about one resample in six moved by one step, and
  its inclusions by about 17%.
- **Bins without methylation** match every unmethylated contig; inclusions into them need P ≥ 0.95.
- **Mobile elements methylated unlike their host**, such as phages with their own methyltransferases or high-copy
  plasmids that are partly unmethylated, are left unbinned rather than placed with their host when unbinned, and can
  be flagged as contaminants when binned (on the fecal sample, 4 of the 8 binned plasmids and viruses with a
  decision).
- **Results depend on the motif table**: extra motifs that no bin methylates lowered how many contigs were put back
  (measured with 0.11; see the [FAQ](faq.md#must-the-motifs-come-from-nanomotif)).
- **Motif groups are treated as independent.** One enzyme whose sites spread over several partly methylated groups
  can count more than once. On the fecal sample, dropping such groups entirely reversed or weakened below 3 nats
  the evidence of 2 of 101 flags and 17 of 485 inclusions, none of them contradicted by Hi-C.

The statistical details, with formulas and the reasoning behind each choice, are in the developer notes
([development/method-notes.md](development/method-notes.md)) and in the manuscript (in preparation).
