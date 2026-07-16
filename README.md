# Project ThermoLogic

A lightweight, self-contained PyTorch prototype fusing **Differentiable Theorem
Proving (DTP)** with an **Energy-Based Model (EBM)**. It demonstrates how a
neural network's probabilistic outputs can be mathematically constrained by
deterministic logical rules through an energy-minimization loss: *logically
valid states have (near-)zero energy; contradictions cost exponentially much.*

> Reasoning as thermodynamics — the network explores the latent space
> probabilistically, but the internal "physics" (the EBM energy) drives every
> belief state toward logical validity.

## Quick start

```bash
./run.sh                 # installs deps (torch, numpy) and trains end-to-end
./run.sh --epochs 60     # any train.py flag is forwarded
```

No dataset, no configuration, no manual steps — the synthetic logical dataset is
generated natively from the rule base at runtime.

## What you'll see

Across training the mean **energy** falls to ~0, per-rule **satisfaction** rises
to ~1, and accuracy on the **derived** atoms (never directly supervised —
inferred purely through the energy) climbs from chance to ~96%. A final probe
contrasts a valid world (`energy ≈ 0`) with a hand-built contradiction
(`energy ≈ 43`), a ~10¹⁰× gap.

## The paper

[`PAPER.md`](PAPER.md) is the full technical report — abstract, method,
labelled diagrams, charts, and results tables. Regenerate every figure and the
`results/metrics.json` it cites with:

```bash
python experiments.py    # writes figures/*.png and results/metrics.json
```

Headline results: derived-atom accuracy **0.55 → 0.96** (no-logic ablation:
**0.34**), mean energy → `4×10⁻⁴`, rule satisfaction `0.9999`, and a
**>10¹⁰×** energy gap between valid and contradictory belief states.

## Files

| File                    | Role |
|-------------------------|------|
| `architecture_plan.md`  | Technical design: discrete-logic → differentiable-tensor mapping, component map. |
| `PAPER.md`              | Full technical report with diagrams, charts, and results tables. |
| `logic_engine.py`       | DTP module: fuzzy t-norms (Product / Łukasiewicz / Gödel), residuated implication, `KnowledgeBase`, per-rule satisfaction. |
| `model.py`              | `NeuralProposer` (MLP), `EnergyBasedModel` (exponential energy), `ThermoLogicLoss`. |
| `train.py`              | Autonomous pipeline: synthetic data, mini-batch SGD, metrics, energy probe. |
| `experiments.py`        | Reproducible experiment battery → `figures/*.png` + `results/metrics.json`. |
| `run.sh`                | One-command pipeline driver. |
| `figures/`, `results/`  | Generated charts and machine-readable metrics for the paper. |

## Requirements

Python 3.9+, `torch`, `numpy` (installed automatically by `run.sh`).
