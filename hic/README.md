# Hi-C validation

Hi-C (proximity ligation) links DNA molecules that were in the same cell, so it measures what
pbinner infers from methylation, through an independent signal. This folder turns raw Hi-C reads
into contig contacts and scores bin refinement against them. It is separate from pbinner and runs
from the workflow (`workflow/Snakefile`, rules `hic_*`) for samples with `hic_r1` / `hic_r2` in the
config.

1. **Mapping** (`chromap --preset hic`): read pairs to the assembly; pairs with MAPQ ≥ 20 on two
   positions, PCR duplicates removed. `hic_max_pairs` in the config limits the number of read pairs
   used (the first N).
2. **`contacts.py`**: contacts between different contigs, per contig and bin, against what random
   ligation between cells would give for that bin's overall Hi-C signal (abundance x restriction
   sites). A contig's Hi-C bin is the most enriched bin with at least 10 contacts and at least 60%
   of the contig's contacts to bins.
3. **`validate.py`**: for each refinement method, contamination flags (confirmed / false / missed)
   and inclusions (correct / wrong / missed) against the confident Hi-C calls.

Caveat: the Hi-C library's restriction enzymes may be blocked by some DNA modifications, which would
make methylated genomes under-represented in contacts. Check contacts per site against the bins'
methylation before trusting low-contact calls.
