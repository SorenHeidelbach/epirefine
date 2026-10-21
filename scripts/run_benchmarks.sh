#!/usr/bin/env bash
# Run pbinner on every benchmark scenario and score it (Nanomotif results are reused).
# Usage: scripts/run_benchmarks.sh <label>   -> writes <scenario>/pbinner_<label>/ and appends to the results tables
set -euo pipefail
label=${1:?label}
PB=/home/ubuntu/plasmid_binner/.venv/bin
S=/home/ubuntu/plasmid_binner/scripts
Z=/home/ubuntu/data/ZymoOral/bench
SIM=/home/ubuntu/data/sim_bench

run() {  # assembly pileup contig_bin motifs out cache truth table
  local log=$5.log
  $PB/pbinner run --assembly "$1" --pileup "$2" --contig-bin "$3" --motifs "$4" --workers 16 --out "$5" \
    ${6:+--cache-dir "$6"} > "$log" 2>&1 || { echo "FAILED $5 (see $log)"; return; }
  $PB/python $S/evaluate_refinement.py --truth "$7" --contamination "$5/bin_contamination.tsv" \
    --new-bins "$5/new_contig_bin.tsv" --name "pbinner_$label" --table "$8" > /dev/null
  echo "$5: $(grep -E 'refinement' "$log" | cut -c12-200)"
}

for s in 1 2 3; do
  run $Z/frag/assembly.fasta $Z/frag/pileup.bed $Z/scen$s/contig_bin.tsv $Z/../ZymoOral/nanomotif_1.1.3/bin-motifs.tsv \
      $Z/scen$s/pbinner_$label $Z/pbcache $Z/scen$s/truth_actions.tsv $Z/results_s$s.tsv
done
for s in 4 5 6; do
  run $Z/frag_short/assembly.fasta $Z/frag_short/pileup.bed $Z/scen$s/contig_bin.tsv $Z/../ZymoOral/nanomotif_1.1.3/bin-motifs.tsv \
      $Z/scen$s/pbinner_$label $Z/pbcache_short $Z/scen$s/truth_actions.tsv $Z/results_s$s.tsv
done
for s in 1 2 3; do
  d=$SIM/sim1/scen$s
  run $SIM/sim1/assembly.fasta $SIM/sim1/pileup.bed $d/contig_bin.tsv $d/bin-motifs.tsv \
      $d/pbinner_$label "" $d/truth_actions.tsv $d/results.tsv
done
