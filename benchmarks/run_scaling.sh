#!/bin/bash
# Thread-scaling study (weak scaling): the file count grows with the worker count, so each
# worker always gets the same amount of work and the wall time per point stays comparable.
# Six files (~49k events) per worker keeps every configuration above ~17 s of real work, so
# process start-up is a small fraction rather than the thing being measured.
set -u
OUT=${1:-/tmp/scaling.jsonl}
FILES_PER_WORKER=${2:-6}
: > "$OUT"
for JOBS in 1 5 10 15 20; do
  FILES=$((JOBS * FILES_PER_WORKER))
  bash /exp/mu2e/app/users/mmackenz/main/pyevtana/benchmarks/run_point.sh "$JOBS" "$FILES" "$OUT"
done
echo "SCALING COMPLETE -> $OUT"
