#!/bin/bash
# Thread-scaling study. Files scale with thread count (3 files per worker) so that each
# worker gets enough events to amortize process start-up, and so the wall time per point
# stays roughly constant for the slowest configuration.
set -u
OUT=${1:-/tmp/scaling.jsonl}
: > "$OUT"
for JOBS in 1 5 10 15 20; do
  FILES=$((JOBS * 3))
  bash /exp/mu2e/app/users/mmackenz/main/pyevtana/benchmarks/run_point.sh "$JOBS" "$FILES" "$OUT"
done
echo "SCALING COMPLETE -> $OUT"
