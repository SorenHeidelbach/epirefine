# Hi-C validation

Hi-C (proximity ligation) links DNA molecules that were in the same cell, so it measures what
EpiRefine infers from methylation, through an independent signal. This folder turns raw Hi-C reads
into contig contacts and scores bin refinement against them. It is separate from EpiRefine and runs
from the workflow (`workflow/Snakefile`, rules `hic_*`) for samples with `hic_r1` / `hic_r2` in the
config.

1. **Mapping** (`chromap --preset hic`): read pairs to the assembly; chromap keeps pairs whose
   two reads both map uniquely (pair type `UU`) and removes PCR duplicates. `contacts.py` also keeps
   only `UU` pairs, in case another mapper's .pairs file is given. `hic_max_pairs` in the config limits the number of read pairs
   used (the first N).
2. **`contacts.py`**: contacts between different contigs, per contig and bin, against what random
   ligation between cells would give for that bin's overall Hi-C signal (abundance x restriction
   sites). A contig's Hi-C bin is the most enriched bin with at least 10 contacts and at least 60%
   of the contig's contacts to bins.
3. **`validate.py`**: for each refinement method, contamination flags (confirmed / false / missed)
   and inclusions (correct / wrong / missed) against the confident Hi-C calls.

4. **`refine.py`**: a Hi-C-only refinement (move binned contigs to their Hi-C bin, add unbinned
   contigs with a confident Hi-C bin), scored with CheckM2 as bin set `hic`.
5. **`bin_quality.py`**: Hi-C quality per bin and bin set: share of contacts that stay inside the
   bin, sequence Hi-C places in another bin (foreign) or in this bin from outside (missing), next to
   CheckM2 completeness and contamination.
6. **`marker_visibility.py`**: whether CheckM2 can see the contigs Hi-C calls foreign: do they carry
   single-copy marker genes (derived from the sample's high-quality bins), and are those markers
   duplicated in the bin (reads as contamination) or new to it (reads as completeness)?

Caveat: the Hi-C library's restriction enzymes may be blocked by some DNA modifications, which would
make methylated genomes under-represented in contacts. Check contacts per site against the bins'
methylation before trusting low-contact calls.
