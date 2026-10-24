#!/usr/bin/env bash
# Reproduces the whole experiment in one command: stage 1 (road networks, C and Python heaps),
# stage 2 (synthetic ladder), stage 3 (fixed k, t), the USA graph if present, and the figures.
#   ./run_all.sh          full run: about 6 h and up to 12.5 GB of RAM; overwrites results/
#   ./run_all.sh --quick  reduced run (a few minutes) into results/quick/, to check that everything works
set -euo pipefail
cd "$(dirname "$0")"

if [ "${1:-}" = "--quick" ]; then
  OUT=results/quick; ROADS="datasets/USA-road-d.NY.gr.gz"; SOURCES=2; LADDER="15 16 17 18"; USA=""
else
  OUT=results; SOURCES=10; LADDER="15 16 17 18 19 20 21 22 23 24"; USA=datasets/USA-road-d.USA.gr.gz
  ROADS="datasets/USA-road-d.NY.gr.gz datasets/USA-road-d.COL.gr.gz datasets/USA-road-d.CAL.gr.gz datasets/USA-road-d.E.gr.gz"
fi
mkdir -p "$OUT/csv" "$OUT/logs"

# matplotlib is only needed for the figures: use .venv if it exists, otherwise create it
if [ ! -x .venv/bin/python ]; then python3 -m venv .venv && .venv/bin/pip install -q matplotlib; fi

# keep the machine awake during the long runs (a suspend freezes the benchmark)
run() {
  if command -v systemd-inhibit >/dev/null 2>&1; then
    systemd-inhibit --what=sleep:idle --who=run_all.sh --why="benchmark running" "$@"
  else
    "$@"
  fi
}

run python3 src/benchmark.py $ROADS --sources "$SOURCES" --heap both --output "$OUT/csv/results.csv" | tee "$OUT/logs/benchmark_roads.log"
run python3 src/benchmark.py --synthetic $LADDER --sources 3 --heap both --output "$OUT/csv/results_synthetic.csv" | tee "$OUT/logs/benchmark_synthetic.log"
run python3 src/benchmark.py $ROADS --sources 3 -k 2 -t 6 --no-count --output "$OUT/csv/results_kt_fixed.csv" | tee "$OUT/logs/benchmark_kt_fixed.log"
if [ -n "$USA" ] && [ -f "$USA" ]; then
  run python3 src/benchmark.py "$USA" --sources 3 --heap both --output "$OUT/csv/results_usa.csv" | tee "$OUT/logs/benchmark_usa.log"
elif [ -n "$USA" ]; then
  echo "USA graph not found ($USA): skipped, see datasets/README.md"
fi
.venv/bin/python src/plots.py --csv "$OUT/csv" --output "$OUT/figures"
echo "done: CSVs in $OUT/csv, logs in $OUT/logs, figures in $OUT/figures"
