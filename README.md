# Project ThermoLogic

**A differentiable logic-energy layer for AI outputs.** Drop it after any model
to *score* how much an output violates your hard rules (a label-free
inconsistency signal) and *repair* it to the nearest valid state — at a compute
budget you control.

> Reasoning as thermodynamics: logically valid outputs sit at **low energy**;
> contradictions cost **exponentially** more. "Thinking" is letting an answer roll
> downhill until it obeys the rules.

By **Teo Qing Cong Eugene** · [linkedin.com/in/eugene-teo](https://www.linkedin.com/in/eugene-teo)

📄 [Technical report](PAPER.md) · 📑 [PDF](ThermoLogic_TechnicalReport.pdf) · 📖 [Plain-English explainer](EXPLAINER.md) · 🕹️ [Interactive playground](playground.html) · 🧭 [Walkthrough](WALKTHROUGH.md)

---

## Quick start

```bash
pip install -e .                          # installs the thermologic package
```

```python
import torch
from thermologic import LogicEnergy, implies

guard = LogicEnergy(
    rules=[
        implies(["sso"], "plan_enterprise", name="sso⇒enterprise"),
        implies(["plan_free"], "seats_gt_5", negate_consequent=True, name="free⇒≤5 seats"),
    ],
    atom_names=["plan_free", "plan_enterprise", "sso", "seats_gt_5"],
)

out = torch.tensor([[1.0, 0.0, 1.0, 1.0]])   # free plan + SSO + >5 seats (invalid)
guard.score(out)        # -> tensor([...])  label-free inconsistency signal (>0)
guard.violations(out)   # -> [['sso⇒enterprise', 'free⇒≤5 seats']]
guard.repair(out, budget=60)   # -> nearest rule-satisfying configuration
```

## What's inside

| Result (see the paper) | Product feature |
|---|---|
| Energy = logical inconsistency (label-free) | `score()` — a hallucination/trust signal, no ground truth needed |
| Test-time repair pulls outputs to validity | `repair()` — fix an output to the nearest valid state |
| Reasoning depth = test-time compute | `budget=` — a "thinking" dial; deeper rules need more |

**Headline numbers** (11-atom benchmark, unseen-world test, 5 seeds): a plain
supervised net matches on accuracy (`0.955`) but its outputs are logically
*inconsistent* (energy `0.95`); ThermoLogic + repair reaches `0.971` at energy
`0.001` — the value is the **consistency guarantee**, not accuracy.

## Run everything

```bash
./run.sh                                 # autonomous training pipeline
python experiments.py                    # regenerate all figures + results/metrics.json
python examples/config_validator.py      # worked use case: SaaS config validator
python -m unittest discover -s tests     # 18 unit tests
```

## Repository layout

| Path | Role |
|---|---|
| `thermologic/` | The library: `api.py` (`LogicEnergy`), `logic_engine.py` (DTP / t-norms), `model.py` (EBM, repair), `data.py` |
| `examples/config_validator.py` | Worked use case — config validation via score/repair |
| `experiments.py` | Reproducible experiment battery → `figures/`, `results/metrics.json` |
| `train.py` · `run.sh` | Autonomous training pipeline |
| `tests/` | Unit tests (t-norms, energy, repair, API) |
| `PAPER.md` · `EXPLAINER.md` · `architecture_plan.md` | Technical report, plain-English version, design doc |
| `figures/` · `results/` | Generated charts and metrics |

## Scope

v1 targets **structured/tabular outputs with propositional rules** (config/form
validation, tabular decisions, data-integrity repair). General LLM-text
validation is roadmap, not a claim. This is an honest proof-of-concept and a small
usable tool — see the [technical report](PAPER.md) for the full, caveated story.

## License

MIT.
