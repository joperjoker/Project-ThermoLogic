#!/usr/bin/env bash
#
# Regenerate every experimental result and figure in the paper from scratch.
# CPU-only, fixed seeds. Takes ~15-20 min total.
#
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"

echo ">> [1/6] core experiments  -> figures/fig_{depth_wave,generalization,repair_curve,baselines,robustness,tnorm,energy_landscape}.png + results/metrics.json"
"$PY" experiments.py

echo ">> [2/6] projection head-to-head -> figures/fig_projection_baseline.png + results/baselines.json"
"$PY" experiments_baselines.py

echo ">> [3/6] train-through-repair (differentiability capability)"
"$PY" experiments_through_repair.py

echo ">> [4/6] pre- vs post-repair supervision (null result) -> figures/fig_supervision.png + results/supervision.json"
"$PY" experiments_supervision.py

echo ">> [5/6] semi-supervised hard rule (the win) -> figures/fig_semisup.png + results/semisup.json"
"$PY" experiments_semisup.py

echo ">> [6/6] plain-language slide figures -> assets/slide_{depthwave,baselines}.png"
"$PY" assets/make_slide_figures.py

echo ">> unit tests"
"$PY" -m unittest discover -s tests

echo ">> done. All figures in figures/ and assets/, metrics in results/."
