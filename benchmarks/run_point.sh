#!/bin/bash
# One point of the scaling study: all three configurations at a given thread count.
#   ./run_point.sh <jobs> <files> <results.jsonl>
set -u
JOBS=$1; FILES=$2; OUT=$3
cd /exp/mu2e/app/users/mmackenz/main/pyevtana
echo "########## jobs=$JOBS files=$FILES ##########"
python3 benchmarks/bench_pyfitter.py  --files "$FILES" --jobs "$JOBS" --json "$OUT" 2>&1 | grep -E "^(cache|===|  )" 
python3 benchmarks/bench_pyevtana.py  --files "$FILES" --jobs "$JOBS" --mode read     --no-warm --json "$OUT" 2>&1 | grep -E "^(cache|===|  )"
python3 benchmarks/bench_pyevtana.py  --files "$FILES" --jobs "$JOBS" --mode objects --no-warm --json "$OUT" 2>&1 | grep -E "^(cache|===|  )"
