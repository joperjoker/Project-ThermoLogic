# Logical Energy as Test-Time Compute: Scoring and Repairing Neural Outputs with a Differentiable Theorem-Proving Energy-Based Model

**Teo Qing Cong Eugene** · [linkedin.com/in/eugene-teo](https://www.linkedin.com/in/eugene-teo)

*Technical report · Project ThermoLogic · reproducible PyTorch prototype*

---

> **Scope & honesty note.** This is an engineering *technical report / proof-of-
> concept*, not a peer-reviewed claim of a new paradigm. It builds directly on
> established neuro-symbolic work (Logic Tensor Networks, DeepProbLog, Semantic
> Loss, Neural Theorem Provers, and analyses of differentiable fuzzy operators —
> see §9). Every number below is produced by the accompanying code with fixed
> seeds. The contribution is (i) a clean, from-scratch, fully-reproducible
> implementation, and (ii) two small original observations about **test-time
> energy repair** and **how reasoning depth maps to test-time compute**.

---

## Abstract

> **In plain terms.** AI models often produce answers that break basic rules — an
> invalid configuration, a self-contradiction — with no built-in way to notice.
> ThermoLogic gives any answer a *score* for how much it breaks your rules (zero
> means it obeys all of them), then "rolls it downhill" to the nearest valid
> answer, using only the rules and no training data. This report shows the idea
> works on a small benchmark, that the score doubles as a label-free error
> detector, and that harder (deeper) reasoning takes proportionally more compute.

Neural models trained by maximum likelihood have no internal mechanism that
forbids logically inconsistent outputs. We present **Project ThermoLogic**, a
compact PyTorch system that fuses **Differentiable Theorem Proving** (relaxing
discrete rules into fuzzy-logic *t-norms*) with an **Energy-Based Model**: a
belief state's energy is `~0` when it satisfies every rule and grows
*exponentially* as rules are violated. A small MLP (the *Neural Proposer*)
proposes probabilistic truth values; the energy then acts as a differentiable
*prover*. On an 11-atom benchmark with a genuine world-level train/test split we
report five findings. **(1)** A plain supervised network matches ThermoLogic on
raw accuracy (`0.955` vs `0.970`) but its outputs are logically *inconsistent*
(mean energy `0.95`); the energy layer's value is a **label-free consistency
guarantee** (energy `0.001`), not accuracy. **(2)** **Test-time energy repair** —
gradient descent on the energy — turns an inconsistent output on *unseen* inputs
into a valid one (`0.935 → 0.971` accuracy, energy `0.90 → 0.0006`), a form of
inference-time reasoning. **(3)** In a proof chain, truth propagates through the
energy **one hop at a time**: inference cost grows *linearly with reasoning
depth*, and the choice of t-norm sets the propagation speed (`~3.4` descent steps
per hop for Łukasiewicz vs `~1.0` for Gödel/Product). We then ask *"why not just
project onto the nearest valid state with a solver?"* and answer it honestly:
**(4)** an exact projection **ties** energy repair on accuracy (`0.969` vs
`0.970`) — the energy's edge is instead **linear scaling** where enumeration
explodes (`2ᵈ`) and **differentiability**, which lets the repair be *trained
through* (a projection gives zero gradient). Finally we map the boundary of that
benefit: **(5)** with full labels and an easily-learned target it is a **null**
(no generalization gain), but as a semi-supervised loss on a *hard* target
(parity) it lifts accuracy by up to **+26 points** when labels are scarce — the
regime theory predicts. We package the mechanism as a small library
(`thermologic`) that **scores** and **repairs** any model's structured output
against hard rules at a controllable compute budget.

**Keywords:** neuro-symbolic AI, energy-based models, differentiable logic,
t-norms, test-time compute, constraint satisfaction, guardrails, PyTorch.

---

## 1. Introduction

A maximum-likelihood objective rewards fitting the data distribution; it never
assigns *infinite* cost to a self-contradictory output. Symbolic systems are the
opposite — rigorously consistent but non-differentiable, so they do not compose
with gradient learning. Project ThermoLogic asks a narrow, testable question:

> *Can a neural network's continuous, probabilistic beliefs be constrained — and
> repaired — by discrete logical rules, using only backpropagation?*

The organizing metaphor is thermodynamic:

```
logical validity ⇔ low energy (ground state)    contradiction ⇔ high energy
```

The network explores the belief space; the energy makes invalid configurations
expensive. Crucially, the energy is differentiable, so it can be minimized **at
training time** (as a loss) *and* **at test time** (as an inference procedure
that repairs a given output).

*(The thermodynamic framing is an analogy for intuition, not literal physics:
there is no entropy, and the "inverse temperature" `β` is simply a hyperparameter
that sharpens the penalty.)*

### 1.1 Contributions

1. A clean, fully-reproducible **DTP + EBM** implementation with three t-norm
   families and residuated implications (`thermologic` package, 18 unit tests).
2. A **label-free consistency signal + repair** operator, and the empirical
   finding that a plain supervised net matches on accuracy but *not* consistency
   — so the energy's value is reliability, not accuracy (§5.3).
3. **Test-time energy repair** framed as inference-time compute, with a measured
   compute–consistency curve (§5.2).
4. The **truth-propagation wave**: reasoning depth ↦ test-time compute is linear,
   and the t-norm sets the speed (§5.1).
5. An **honest boundary analysis**: energy repair ties an exact solver on accuracy
   but wins on scaling and differentiability (§5.7–5.8); training-through-repair is
   a null with easy targets/full labels (§5.9) yet gives **+26 points** as a
   semi-supervised loss on a hard target (§5.10).
6. A small **product** — `LogicEnergy.score()/repair()` — and a worked config-
   validation use case.

---

## 2. Background and related work

**Fuzzy logic & t-norms.** Truth values relax to `[0,1]`; conjunction becomes a
*t-norm* `⊗`, implication its *residuum*. van Krieken et al. (2022) analyze which
operators yield usable gradients — motivating our Łukasiewicz default.

**Neuro-symbolic learning.** Logic Tensor Networks (Badreddine et al., 2022),
DeepProbLog (Manhaeve et al., 2018), Semantic Loss (Xu et al., 2018), and Neural
Theorem Provers (Rocktäschel & Riedel, 2017) all inject logic into learning,
typically as a **training-time** loss or probabilistic program. **Energy-based
models** (LeCun et al., 2006) shape a scalar energy landscape. This report's
angle — using a differentiable-logic energy as a **test-time** repair/inference
procedure, and characterizing its compute-vs-depth behaviour — is the less-
explored combination.

---

## 3. Method

### 3.1 Discrete logic → differentiable tensors

Throughout, an **atom** is a single atomic proposition — one true/false statement
such as `Rain` or `sso_enabled` (the standard logic term for the smallest unit of
a rule; unrelated to physics atoms despite the thermodynamic framing).

Let `p_i ∈ [0,1]` be the belief that atom `i` is true. A literal's truth is `p_i`
(or `1−p_i` if negated). A Horn rule `L₁ ∧ … ∧ Lₖ → C` scores as
`sat = residuum( ⊗ᵢ truth(Lᵢ), truth(C) ) ∈ [0,1]`.

**Table 1 — Operator mapping per t-norm.**

| Discrete | Product | Łukasiewicz (default) | Gödel |
|---|---|---|---|
| `¬a` | `1−a` | `1−a` | `1−a` |
| `a ∧ b` | `a·b` | `max(0, a+b−1)` | `min(a,b)` |
| `a ⇒ c` (residuum) | `1 if a≤c else c/a` | `min(1, 1−a+c)` | `1 if a≤c else c` |

**Modus ponens is emergent:** with the antecedent believed (`≈1`),
`residuum(1, p_C)=p_C`, so `sat→1` *requires* `p_C→1`. The gradient of the energy
w.r.t. `p_C` is what derives `C`. Łukasiewicz is the default: its residuum
`min(1,1−a+c)` is piecewise-linear with bounded, division-free gradients.

### 3.2 The energy (EBM)

Per-rule energy `E_r = exp(β(1−sat_r)) − 1`; total `E = mean_r E_r`. Valid ⇒
`E=0`; contradictions grow exponentially with inverse-temperature `β`.

```mermaid
flowchart TB
    A["contradiction<br/>sat→0 · E≈exp(β)−1 (HIGH)"]:::hot --> B["partial violation<br/>E rising"]:::warm --> C["valid state<br/>sat→1 · E≈0 (GROUND STATE)"]:::cold
    classDef hot fill:#D55E00,color:#fff
    classDef warm fill:#E69F00,color:#000
    classDef cold fill:#009E73,color:#fff
```

### 3.3 Training objective and test-time repair

Training loss (amortized inference): `L = w_sup·BCE(p_causes, y) + w_rule·E +
w_par·mean(p_derived)`; supervision anchors observed atoms, energy constrains the
rest, parsimony selects the minimal model.

**Test-time repair (the key operator).** Given *any* belief state `p` (e.g. a
model's output on a novel input), hold trusted atoms fixed and run gradient
descent on `E` over the rest, in logit space:

```
p* = argmin_{free atoms}  E(p)      # iterative, label-free logical inference
```

Because minimizing `E` enforces every rule, this *repairs* an inconsistent output
to the nearest valid state. The number of steps is a **compute budget**.

```mermaid
flowchart LR
    X["input x"] --> NP["Neural Proposer (MLP)"] -->|"beliefs p"| DTP["Differentiable Logic Engine<br/>(t-norm satisfaction)"]
    DTP -->|"sat"| E["Energy  E=mean(exp(β(1−sat))−1)"]
    E -->|train: ∇ loss| NP
    E -->|"test-time: ∇ energy (repair)"| R["repaired p*"]
```

---

## 4. Benchmark

**Table 2 — Expanded knowledge base.** 11 atoms, 9 rules, **18** distinct
logically-consistent worlds; `294 / 2048` full Boolean worlds are consistent.

| Group | Atoms | Rules (sample) |
|---|---|---|
| Causes (5, observed) | Rain, Sprinkler, Cold, HeaterOn, Windy | `Rain→¬Sprinkler`, `HeaterOn→¬Cold` (constraints) |
| Derived (6, inferred) | Cloudy, WetGround, Slippery, Ice, Warm, Chilly | `Rain→Cloudy`, `Rain/Sprinkler→WetGround`, `WetGround→Slippery`, `Cold∧WetGround→Ice`, `HeaterOn→Warm`, `Windy∧Cold→Chilly` |

Ground truth is the forward-chained minimal model. We **hold out entire worlds**:
train on 12 worlds, test on 6 the model has *never seen*. Inputs are noisy
encodings of the cause atoms (σ=0.12) plus 3 distractor dims.

---

## 5. Experiments

All numbers regenerate from the scripts named per subsection (seeds fixed;
CPU-deterministic). Rather than argue the method is uniformly good, we map
*exactly* where the logic helps and where it does not.

**Table 3 — Findings scorecard (does the differentiable logic help?).**

| Question | Answer | Where |
|---|---|---|
| Is repair a form of inference-time reasoning? | **Yes** — energy → 0, unseen acc `0.935→0.971` | §5.2 |
| Does reasoning depth cost compute? | **Yes** — linear, one hop at a time | §5.1 |
| More accurate than a plain net? | **No** — comparable; the win is *consistency* | §5.3 |
| More accurate than an exact solver/projection? | **No** — a tie (`0.970` vs `0.969`) | §5.7 |
| Then why not just use a solver? | Energy **scales linearly** (solver is `2ᵈ`) and is **differentiable** | §5.7–5.8 |
| Can you train *through* it? | **Yes** — a projection gives zero gradient | §5.8 |
| Does training-through-repair improve generalization? | **No** (full labels, easy target — a null) | §5.9 |
| Does the logic ever improve learning? | **Yes** — semi-supervised, hard target: **+26 pts** | §5.10 |

### 5.1 Logical energy as test-time compute (the propagation wave)

Starting from a state where a cause is true but a chain `A₀→A₁→…→A_k` of derived
atoms is false, test-time energy descent (plain SGD) turns atoms on **one hop at
a time** — atom depth `d` activates only after `d−1`. Activation step is a
strictly monotone, near-linear function of depth; the t-norm sets the slope.

**Figure 1 — Truth propagates as a wave; inference cost is linear in reasoning depth.**

![Depth wave](figures/fig_depth_wave.png)

**Table 4 — Propagation speed (k=20 chain).**

| t-norm | steps per reasoning hop | wave monotonic? |
|---|---|---|
| Łukasiewicz | `3.38` | yes |
| Product | `0.98` | (near) |
| Gödel | `1.00` | yes |

*Caveats (honest):* the clean monotone wave requires momentum-free SGD (Adam's
momentum accelerates and distorts the tail), and because energy is a *mean* over
rules, longer chains propagate slightly slower per step. The robust claim is the
within-chain monotone wave and linear depth-cost at fixed length. We also stress
that the linear scaling is **intuitive, not surprising**: any iterative fixed-
point or message-passing procedure needs steps proportional to how far
information must travel. The contribution here is a clean, visual *demonstration*
of that behaviour in a differentiable-logic energy, not the discovery of a new
scaling law.

### 5.2 Generalization and test-time repair

On worlds never seen in training, a single feed-forward pass generalizes only
partially and leaves outputs logically *inconsistent* (high energy). Repair fixes
them.

**Figure 2 — Amortized inference is inconsistent on unseen worlds; repair fixes it.**

![Generalization](figures/fig_generalization.png)

**Table 5 — Generalization (mean ± std, 5 seeds).**

| Split | Derived accuracy | Mean energy |
|---|---|---|
| Seen worlds | `0.995 ± 0.010` | `~0` |
| Unseen (feed-forward) | `0.935 ± 0.022` | `0.898 ± 0.178` |
| **Unseen (+ repair)** | **`0.971 ± 0.007`** | **`0.001 ± 0.000`** |
| Cause recovery (unseen) | `0.981 ± 0.004` | — |

Repair is a **compute knob**: more descent steps → lower energy and higher
accuracy, plateauing around 50 steps.

**Figure 3 — Test-time compute vs. energy and accuracy.**

![Repair curve](figures/fig_repair_curve.png)

| Budget (steps) | 0 | 10 | 30 | 50 | 100 | 200 |
|---|---|---|---|---|---|---|
| Accuracy | 0.935 | 0.950 | 0.973 | 0.974 | 0.975 | 0.975 |
| Energy | 0.980 | 0.591 | 0.033 | 0.0014 | 0.0008 | 0.0004 |

### 5.3 Baselines — the value is consistency, not accuracy

A **plain supervised network** (BCE on all atoms, no logic) reaches `0.955`
accuracy on unseen worlds — comparable to ThermoLogic+repair (`0.970`). But its
outputs are logically *inconsistent* (energy `0.946`), with no signal that
anything is wrong. Only the energy provides a **label-free consistency guarantee**
(`0.001`).

**Figure 4 — Comparable accuracy; only the energy guarantees consistency.**

![Baselines](figures/fig_baselines.png)

**Table 6 — Baselines on unseen worlds (mean ± std, 3 seeds).**

| Method | Derived accuracy | Mean energy (consistency) |
|---|---|---|
| Supervised NN (no logic) | `0.955 ± 0.010` | `0.946 ± 0.053`  ✗ inconsistent |
| ThermoLogic (feed-forward) | `0.927 ± 0.026` | `0.90` |
| **ThermoLogic (+ repair)** | `0.970 ± 0.008` | **`0.001 ± 0.000`  ✓** |

> This is the honest headline: ThermoLogic is a **reliability** mechanism, not a
> way to beat a classifier on accuracy. The energy is a label-free detector *and*
> repairer of rule violations — which a plain net cannot provide.

### 5.4 Robustness

**Figure 5 — Robustness to input noise and holdout size (mean ± std, 3 seeds).**

![Robustness](figures/fig_robustness.png)

Repair helps most when causes are recoverable (low–moderate noise); at high noise
(σ≥0.3) mis-recovered causes cap accuracy and repair cannot help (garbage in). As
more worlds are held out (less coverage), generalization degrades gracefully.

### 5.5 t-norm deep-dive

**Figure 6 — t-norm comparison on the 11-atom KB.**

![t-norm](figures/fig_tnorm.png)

| t-norm | Unseen acc (+repair) | Feed-forward energy |
|---|---|---|
| **Łukasiewicz** | **`0.970 ± 0.008`** | `0.965` |
| Product | `0.876 ± 0.035` | `0.537` |
| Gödel | `0.878 ± 0.002` | `4.084` |

Łukasiewicz's smoother gradients give the best accuracy and repairability — the
default.

### 5.6 Energy landscape

**Figure 7 — Energy partitions all `2¹¹` worlds into consistent (E=0) vs. contradictions.**

![Energy landscape](figures/fig_energy_landscape.png)

Of `2048` Boolean worlds, exactly `294` satisfy all rules (`E=0`); the rest carry
energy in `[5.96, 41.69]`. The energy *is* the indicator of logical validity.

### 5.7 Do you even need learning? Energy repair vs. projection

The obvious objection to a differentiable repair is: *why not just project the
output onto the nearest valid state with a solver?* We compare energy repair
against four alternatives that all hold the recovered causes fixed
(`experiments_baselines.py`, 3 seeds): the raw **feed-forward** output;
**forward-chaining** (recompute derived atoms by the rules — a Horn-KB oracle);
an **exact projection** that enumerates the valid completions and picks the one
closest to the model's *soft* output (confidence-weighted); and a **greedy**
violation-reducing flip search.

**Figure 8 — Energy repair vs. projection: accuracy is a tie; scaling and differentiability are not.**

![Projection baseline](figures/fig_projection_baseline.png)

**Table 7 — Repair methods on unseen worlds (mean over 3 seeds).**

| Method | Derived acc | Valid | Δ from proposal (Hamming) | Agrees w/ exact | Time/sample |
|---|---|---|---|---|---|
| Feed-forward (no repair) | `0.927` | `0.78` | `0.00` | `0.78` | — |
| Forward-chaining (Horn oracle) | `0.973` | `1.00` | `0.41` | `0.92` | `0.02 ms` |
| Exact projection (conf-weighted) | `0.969` | `1.00` | **`0.26`** | `1.00` | `0.30 ms` |
| Greedy repair | `0.959` | `0.97` | `0.21` | `0.97` | `1.70 ms` |
| **Energy repair (ours)** | `0.970` | `1.00` | `0.31` | `0.96` | `0.21 ms` |

**We report this honestly: on accuracy, energy repair does not win.** Exact
projection and the forward-chaining oracle match it (`0.969`–`0.973` vs `0.970`,
within noise), and by construction the exact projection changes the fewest bits.
Energy repair lands on the *same* valid state as the exact projection **96%** of
the time — it is a good, cheap approximation of it, not a better answer.

The energy's advantages are elsewhere, and they are real:

- **Scaling (Figure 8b).** Exact projection enumerates `2ᵈ` completions; its cost
  grows from `0.4 ms` at `d=4` to `968 ms` at `d=18` (doubling each step), crossing
  energy repair's roughly-flat cost near `d≈17`. Beyond that, enumeration is
  intractable and even a MaxSAT/ILP projection is NP-hard and non-differentiable;
  energy repair stays `O(steps × rules)`.
- **Differentiability (§5.8).** A projection is a non-differentiable `argmin`;
  energy repair can be *trained through*.

### 5.8 The capability projection cannot provide: training through repair

Because repair is gradient descent, it can be **unrolled into a differentiable
module** and placed inside a training loop (`experiments_through_repair.py`). The
gradient of a post-repair loss w.r.t. the network is **`0.19` through
differentiable repair and exactly `0.00` through a projection** (a discrete
`argmin` has no gradient).

This is not academic. We supervise a proposer with **only a scalar readout** of
the repaired state (the mean of the derived atoms) — *no per-atom labels* — and
train it end-to-end *through* the repair. Per-atom logical correctness **emerges**:
derived-atom accuracy climbs from `0.47` (init) to **`0.993`** as the readout MSE
falls `0.25 → 0.08`. A projection-in-the-loop receives zero gradient here and
cannot learn this at all. This is the honest answer to "why not just project?":
when the repair must live *inside* a learned system, differentiability is not a
nicety — it is the whole point. (But see §5.9 for where this benefit does *not*
appear.)

### 5.9 A negative result: logic-in-the-loop does not always help

It is tempting to claim that training through the repair is *generally* a better
inductive bias. We tested this fairly and it is **not true on this benchmark**
(`experiments_supervision.py`). With identical architecture, initialization,
optimizer, and *full* per-atom labels, we compared **pre-repair** supervision
(`BCE(proposer(x), y)`) against **post-repair** supervision (`BCE(repair(proposer(x)), y)`),
sweeping the number of distinct training worlds (2 → 12 of 18, six held out), 3
seeds, both evaluated identically with test-time repair.

**Figure 9 — Null result: with full labels, training through repair ties standard supervision.**

![Supervision](figures/fig_supervision.png)

The two curves coincide at every point (e.g. `0.807` vs `0.805` at 6 worlds;
`0.948` vs `0.948` at 12) — no generalization benefit, and no consistency
difference. The reason is instructive: the derived atoms here are *simple*
functions of the causes, so a plain network learns them from data and test-time
repair becomes a no-op on the trained outputs. Logic-as-inductive-bias can only
help when the logical relationship is **hard to learn from data**, which this toy
is not.

**The honest, scoped conclusion.** Differentiable repair provides a *real
capability* — training under supervision that a projection cannot backpropagate
(§5.8) — but it is **not** a free generalization win when full labels are already
available (§5.9). It *does* help under the conditions theory predicts — see §5.10.

### 5.10 Where the logic genuinely helps: semi-supervised learning of a hard rule

The null result of §5.9 has two escapes, both suggested by the semantic-loss
literature (Xu et al., 2018): the logic should help when labels are **scarce**
*and* the target is **hard to learn from data**. We test both at once
(`experiments_semisup.py`) with **parity** — the canonical function an MLP
generalizes poorly — as the derived atom `Z = parity(A₀…A₃)` (16 minterm rules
fully define it). We compare **labels-only** training against the same plus the
**energy (logical inconsistency) as a loss on unlabelled data**, evaluated
feed-forward on the parity bit, sweeping the number of labels (3 seeds).

**Figure 10 — The logic genuinely helps when labels are scarce and the rule is hard.**

![Semi-supervised parity](figures/fig_semisup.png)

**Table 8 — Parity-bit accuracy (feed-forward, 3 seeds).**

| # labels | labels-only | + logic energy | gain |
|---|---|---|---|
| 8 | `0.500` | `0.496` | — (too few to bootstrap) |
| 16 | `0.508` | `0.569` | +0.06 |
| 32 | `0.578` | `0.758` | **+0.18** |
| 64 | `0.651` | `0.909` | **+0.26** |
| 128 | `0.902` | `0.985` | +0.08 |
| 256 | `0.996` | `1.000` | — (both saturate) |

Adding the differentiable-logic energy on unlabelled data lifts parity accuracy
by up to **+26 points** (`0.65 → 0.91` at 64 labels). The gain is largest in the
mid-label regime and vanishes at both ends — below `~8` labels there is too
little signal to bootstrap, and above `~256` the labels alone suffice. This is
exactly the semi-supervised, hard-target regime where a logic loss should help,
and it does.

**Putting §5.9 and §5.10 together** gives the precise boundary: the logic is
inert when the target is easily learned and labels are plentiful (§5.9), and
valuable when labels are scarce and the target is hard (§5.10) — an honest,
predictable characterization rather than a blanket "it helps" claim.

---

## 6. The product

The mechanism ships as `thermologic` — a differentiable logic-energy layer that
drops after any model:

```python
from thermologic import LogicEnergy, implies
guard = LogicEnergy(rules=[implies(["sso"], "plan_enterprise")], atom_names=[...])
guard.score(output)        # label-free inconsistency signal
guard.violations(output)   # which rules broke
guard.repair(output, budget=60)   # nearest valid output (compute-budgeted)
```

The three research results map onto three features: **energy = detector**,
**repair = fixer**, **depth = compute budget**. A worked SaaS config-validation
example (`examples/config_validator.py`) flags an invalid plan (e.g. a Pro request
for enterprise-only SSO), names the broken rule, and repairs to the nearest valid
configuration. *v1 scope:* structured/tabular outputs with propositional rules;
general LLM-text validation is roadmap, not a claim.

### 6.1 A harder, first-order-grounded benchmark

To move past the toy scale we build a realistic **cloud security-configuration**
benchmark (`benchmarks/cloud_config.py`) — the kind of policy checking cloud-
posture tools perform. Rules are written as **first-order templates** with a
variable over resources (e.g. `∀r: has_pii(r) → encrypted(r)`) and **grounded**
over `n` resources into a propositional KB (the standard neuro-symbolic route to
first-order logic). With 3 resources this yields **20 atoms, 19 rules, and 432
distinct valid worlds** — a real held-out generalization test. Trained with a
world-level split (288 train / 144 unseen), the full model reaches **100% policy
consistency** and `0.83` control accuracy on unseen configurations, confirming the
pipeline scales beyond the 11-atom toy.

### 6.2 End-to-end integration: a guardrail on a plain model

We then wire `LogicEnergy` onto a **plain multi-label network trained with no
logic** (`integration_demo.py`) — the realistic "drop it after someone else's
model" case. On unseen cloud configs the plain model emits **13.6% policy-
violating** outputs; the guardrail flags every one, with the exact broken rule,
**no labels** — and `repair` restores **100% consistency**.

**Figure 11 — LogicEnergy as a guardrail: flag every non-compliant config, then repair.**

![Integration](figures/fig_integration.png)

Two honest calibrations of the claim: **(i)** the flag has **~100% precision** — a
flagged output is not a valid world, so it cannot equal the (valid) ground truth —
but it detects *policy inconsistency*, not general correctness, so it misses
consistent-but-wrong outputs (a *high-precision, not high-recall* detector).
**(ii)** repair *guarantees consistency*, not correctness: it enforces the rules
against the model's own recovered facts, so when inputs are noisy, control
accuracy is bounded by input quality (here `0.85`, essentially unchanged) rather
than magically improved. The unambiguous product value is a **label-free,
deterministic policy checker + repairer** that bolts onto any model.

---

## 7. Discussion & limitations

- **Reliability, not accuracy.** The honest value proposition is a label-free
  consistency guarantee and repair, not beating supervised learning on accuracy.
- **Not better than a solver on accuracy.** An exact projection ties or beats
  energy repair (§5.7); the differentiators are scaling and differentiability.
- **Differentiability is a capability, not always a benefit.** Training through the
  repair enables supervision a projection cannot (§5.8), but gives no measurable
  generalization gain when full labels are available (§5.9, a null result).
- **The logic helps under known conditions.** As a semi-supervised loss on
  unlabelled data it lifts a *hard* target (parity) by up to +26 points when
  labels are scarce (§5.10) — inert when the target is easy, valuable when it is
  hard and labels are few.
- **Amortized vs. iterative inference.** The proposer is one forward pass; the
  energy adds an iterative, budgetable inference step — a small instance of
  "test-time compute" for logical consistency.
- **Scale & scope.** 11 atoms, propositional Horn-style rules, synthetic data. No
  first-order variables, no defeasible/non-monotonic rules, no real LLM. Soft
  energy makes contradictions *expensive*, not *impossible* (a crisp projection
  step would give a hard guarantee).
- **Optimizer sensitivity.** The clean propagation wave needs momentum-free SGD;
  mean-energy normalization couples wave speed to chain length.

---

## 8. Conclusion

A differentiable fuzzy-logic energy lets a neural network's beliefs be scored and
*repaired* against hard rules by backpropagation alone. Empirically the value is
reliability: comparable accuracy to a plain net but with a label-free consistency
guarantee, an inference-time repair that scales with a compute budget, and a
clean picture of reasoning depth as test-time compute. The result is a small,
honest, reproducible substrate — and a usable tool — for constraint-aware ML.

---

## 9. Reproducibility & references

```bash
./run.sh                                  # train pipeline
python experiments.py                     # regenerate all figures + metrics.json
python -m unittest discover -s tests      # 18 tests
python examples/config_validator.py       # worked use case
```

**Environment.** Python 3.11, PyTorch 2.13 (CPU), NumPy, Matplotlib. Seeds fixed.

**References (context this builds on).**
Badreddine, Garcez, Serafini, Spranger — *Logic Tensor Networks*, Artif. Intell.
2022 · Manhaeve et al. — *DeepProbLog*, NeurIPS 2018 · Xu, Zhang, Friedman, Liang,
Van den Broeck — *A Semantic Loss Function…*, ICML 2018 · Rocktäschel & Riedel —
*End-to-End Differentiable Proving*, NeurIPS 2017 · van Krieken, Acar, van Harmelen
— *Analyzing Differentiable Fuzzy Logic Operators*, Artif. Intell. 2022 · LeCun,
Chopra, Hadsell, Ranzato, Huang — *A Tutorial on Energy-Based Learning*, 2006 ·
Evans & Grefenstette — *Learning Explanatory Rules from Noisy Data (∂ILP)*, JAIR
2018 · Lu et al. — *NeuroLogic Decoding*, NAACL 2021.

*References are indicative of the intellectual context; this is a self-contained
prototype, not a comparative benchmark against these systems.*
