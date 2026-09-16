#!/bin/bash
# Thread-scaling study (weak scaling): the file count grows with the worker count, so each
# worker always gets the same amount of work and the wall time per point stays comparable.
# Three files (~25k events) per worker; with the partitioning and batch-size fixes in place
# that is still comfortably above process start-up for every configuration.
set -u
OUT=${1:-/tmp/scaling.jsonl}
FILES_PER_WORKER=${2:-3}
: > "$OUT"
for JOBS in ${3:-1 5 10 15 20}; do
  FILES=$((JOBS * FILES_PER_WORKER))
  bash /exp/mu2e/app/users/mmackenz/main/pyevtana/benchmarks/run_point.sh "$JOBS" "$FILES" "$OUT"
done
echo "SCALING COMPLETE -> $OUT"
