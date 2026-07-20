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

echo ">> [5/11] semi-supervised hard rule (the win) -> figures/fig_semisup.png + results/semisup.json"
"$PY" experiments_semisup.py

echo ">> [6/11] novelty check: fuzzy energy vs. Semantic Loss + weight sweep -> figures/fig_semanticloss{,_weight}.png (§5.11)"
"$PY" experiments_semanticloss.py

echo ">> [7/11] harder first-order-grounded cloud benchmark (§6.1)"
"$PY" benchmarks/cloud_config.py

echo ">> [8/11] end-to-end guardrail integration -> figures/fig_integration.png (§6.2)"
"$PY" integration_demo.py

echo ">> [9/11] external Latin-square solver head-to-head -> figures/fig_latin.png + results/latin.json (§6.3)"
"$PY" benchmarks/latin_square.py

echo ">> [10/11] measured scaling: Latin 3/4/5 + 9x9 Sudoku -> figures/fig_scaling.png + results/scaling.json (§6.4)"
"$PY" benchmarks/scaling.py

echo ">> [11/11] plain-language slide figures -> assets/slide_{depthwave,baselines}.png"
"$PY" assets/make_slide_figures.py

echo ">> model-agnostic LLM JSON guardrail demo (§6.5)"
"$PY" examples/llm_json_guardrail.py

echo ">> unit tests"
"$PY" -m unittest discover -s tests

echo ">> done. All figures in figures/ and assets/, metrics in results/."
