# ThermoLogic — Walkthrough

A practical guide to running, using, and sharing Project ThermoLogic. Five
minutes end to end.

---

## 1. Install

```bash
git clone <your-repo-url> && cd Project-ThermoLogic
pip install -e .            # installs the `thermologic` package (needs torch)
```

Optional extras for regenerating figures / building the PDF:

```bash
pip install -e ".[experiments]"   # numpy + matplotlib
pip install -e ".[dev]"           # pytest
```

## 2. Use the library — score & repair in 6 lines

`LogicEnergy` is the whole product. Give it named atoms and rules; it scores and
repairs any model's output.

```python
import torch
from thermologic import LogicEnergy, implies

guard = LogicEnergy(
    atom_names=["plan_free", "plan_enterprise", "sso", "seats_gt_5"],
    rules=[
        implies(["sso"], "plan_enterprise", name="sso⇒enterprise"),
        implies(["plan_free"], "seats_gt_5", negate_consequent=True, name="free⇒≤5 seats"),
    ],
)

out = torch.tensor([[1.0, 0.0, 1.0, 1.0]])   # free + SSO + >5 seats  (invalid)

guard.score(out)        # tensor([...])  — energy > 0  ⇒ inconsistent
guard.is_consistent(out)# tensor([False])
guard.violations(out)   # [['sso⇒enterprise', 'free⇒≤5 seats']]
guard.repair(out, budget=60)   # → nearest valid configuration (tensor)
```

**Reading the output**

| Call | Meaning | Use it for |
|---|---|---|
| `score()` | logical energy, `0` = fully consistent | a trust / hallucination signal, abstention, routing |
| `violations()` | names of the broken rules | explanations, debugging |
| `repair(budget=N)` | nearest rule-satisfying output; `N` = compute budget | auto-fixing outputs; bigger `N` for deeper rule chains |
| `repair(fixed=[...])` | repair while holding some atoms constant | trust the inputs, fix only the rest |

## 3. Run the worked use case

```bash
python examples/config_validator.py
```

A SaaS plan configurator: it flags an invalid request (e.g. a *Pro* customer
asking for enterprise-only *SSO*), names the exact rule, and repairs to the
nearest valid configuration — removing only what it must.

## 4. Reproduce the research

```bash
./run.sh                                 # autonomous training pipeline
python experiments.py                    # all 7 experiments → figures/ + results/metrics.json
python -m unittest discover -s tests     # 18 unit tests
```

`experiments.py` regenerates every figure in the paper (≈5 min on CPU, fixed
seeds). Key outputs:

- `figures/fig_depth_wave.png` — reasoning depth = test-time compute (the finding)
- `figures/fig_generalization.png` — repair fixes unseen-world inconsistency
- `figures/fig_baselines.png` — accuracy comparable, consistency is the differentiator
- `results/metrics.json` — every number, machine-readable

## 5. The interactive playground

Open `index.html` in any browser (no server needed) — or visit the GitHub Pages
site once enabled (Settings → Pages → deploy from branch, root folder):
`https://<your-username>.github.io/Project-ThermoLogic/`. Two live instruments:

- **Constraint Console** — toggle a plan + features, watch the energy gauge, hit
  **Repair** and watch the config cool to a valid state.
- **Reasoning Wave** — drag the *compute* slider and watch a proof propagate one
  hop at a time; switch t-norms to change the propagation speed.

## 6. Read / share

- **Technical report:** [`PAPER.md`](PAPER.md) (or the print-ready
  `ThermoLogic_TechnicalReport.pdf`).
- **Plain-English version:** [`EXPLAINER.md`](EXPLAINER.md).
- **Rebuild the PDF** (after editing `paper_print.html`): open it in a browser
  and *Print → Save as PDF*, or use a headless Chromium `page.pdf()`.

### Posting to LinkedIn — a suggested framing

> Built a small neuro-symbolic proof-of-concept: a *differentiable logic-energy
> layer* that scores how much an AI's output breaks your rules — and repairs it
> to the nearest valid answer, no labels needed. Interesting finding: reasoning
> depth behaves like test-time compute (truth propagates one hop at a time).
> Honest takeaway: it's a *reliability* tool, not an accuracy boost. Code, paper,
> and an interactive demo below.

Attach the PDF, link the playground, and link the repo. Keep the claims matched
to the evidence — it's a proof-of-concept that builds on prior neuro-symbolic
work, not a new paradigm.
