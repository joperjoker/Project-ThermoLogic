# LinkedIn post — ready-to-paste draft

*(Edit voice to taste. Claims below are matched to the measured results — safe
to post as-is. Attach: `ThermoLogic_TechnicalReport.pdf`. Link: the GitHub Pages
demo + repo.)*

---

I spent some time building **ThermoLogic** — a small neuro-symbolic AI
proof-of-concept that treats *reasoning like thermodynamics*.

The idea: today's neural networks have no internal mechanism that forbids
logically impossible outputs. So I fused two classic ideas — differentiable
fuzzy logic and energy-based models — into one PyTorch system where:

🔥 breaking a logical rule costs *exponentially* more "energy"
🧊 valid answers sit at zero energy
📉 and "reasoning" is literally gradient descent rolling an answer downhill
until it obeys the rules

Three findings I think are worth sharing:

1️⃣ **A plain neural net matched mine on accuracy — but couldn't tell when it
was wrong.** The energy score is a label-free lie detector: it flags a
rule-breaking output *without needing the right answer*, then repairs it to the
nearest valid one (energy 0.90 → 0.001 on inputs the model never saw).

2️⃣ **Reasoning depth behaves like test-time compute.** Watching the energy
descent, truth propagates through a chain of rules *one hop at a time* — twice
the reasoning depth costs about twice the compute. You can literally watch the
proof travel.

3️⃣ **Honesty matters.** This builds on prior work (Logic Tensor Networks,
DeepProbLog, Semantic Loss) — it's a reliability mechanism, not an accuracy
boost, and I've kept every claim tied to a reproducible number (fixed seeds,
open code, 18 unit tests).

🕹️ Try the interactive demo in your browser (no install):
https://joperjoker.github.io/Project-ThermoLogic/

📄 Technical report (PDF) attached · code:
https://github.com/joperjoker/Project-ThermoLogic

Where I'd love input: the practical version of this is a "logic guardrail"
layer for structured AI outputs — configs, forms, tabular decisions that must
obey hard business rules. If your team fights that problem, I'd enjoy
comparing notes.

#AI #MachineLearning #NeuroSymbolic #PyTorch #EnergyBasedModels #AIReliability

---

## Posting checklist

- [ ] Repo made public (Settings → General) — required for GitHub Pages on free plans
- [ ] Pages enabled (Settings → Pages → Deploy from branch → `/ (root)`)
- [ ] Demo URL loads: https://joperjoker.github.io/Project-ThermoLogic/
- [ ] Attach `ThermoLogic_TechnicalReport.pdf` as the post document
- [ ] Post at a weekday morning time for your network's timezone
