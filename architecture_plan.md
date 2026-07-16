# Project ThermoLogic — Architecture Plan

A local PyTorch prototype that fuses **Differentiable Theorem Proving (DTP)**
with an **Energy-Based Model (EBM)**. The thesis: a neural network's
probabilistic beliefs can be *mathematically constrained* by deterministic
symbolic logic, by treating logical validity as a low-energy state and logical
contradiction as a high-energy state, and letting gradient descent flow through
the whole graph.

> **Slogan.** Reasoning as thermodynamics: the network explores the latent
> space probabilistically, but the internal "physics" (the EBM energy) makes
> false statements cost (near-)infinite energy, so gradients push every state
> toward logical validity.

---

## 1. The core idea in one paragraph

Discrete logic is not differentiable — `AND`, `IMPLIES`, `NOT` operate on
`{True, False}` and have zero gradient almost everywhere. **Fuzzy logic** fixes
this: we relax truth values to the continuous interval `[0, 1]` and replace the
Boolean connectives with **t-norms** (differentiable algebraic operators whose
corner cases agree with Boolean logic). A neural network (the *Neural Proposer*)
emits probabilistic truth values for atomic propositions; the *DTP layer*
evaluates each symbolic rule as a fuzzy-logic expression, producing a per-rule
**satisfaction** in `[0, 1]`; the *EBM* converts satisfaction into an **energy**
that is `~0` when a rule holds and grows **exponentially** as a rule is
violated. Minimizing energy = proving the rule base holds for the network's
beliefs.

---

## 2. From discrete logic to continuous tensors

Let `p_i ∈ [0, 1]` be the network's belief that atom `i` is true.

| Discrete logic        | Fuzzy relaxation (per t-norm family)                         |
|-----------------------|--------------------------------------------------------------|
| `NOT a`               | `1 − a` (standard/strong negation)                           |
| `a AND b` (t-norm ⊗)  | Product: `a·b` — Łukasiewicz: `max(0, a+b−1)` — Gödel: `min(a,b)` |
| `a OR b` (t-conorm ⊕) | Product: `a+b−a·b` — Łukasiewicz: `min(1, a+b)` — Gödel: `max(a,b)` |
| `a IMPLIES c` (residuum `a ⇒ c`) | Product (Goguen): `1 if a≤c else c/a` — Łukasiewicz: `min(1, 1−a+c)` — Gödel: `1 if a≤c else c` |

A rule `L₁ ∧ … ∧ Lₖ → C` (a Horn-style implication, the workhorse of theorem
proving) is scored as:

```
antecedent = ⊗(literal_truth(L₁), …, literal_truth(Lₖ))   # fuzzy conjunction
sat        = residuum(antecedent, literal_truth(C))        # fuzzy implication
```

where `literal_truth((i, negated)) = p_i` if the literal is positive, else
`1 − p_i`. **Modus ponens** `A ∧ (A→B) → B` is emergent: if `A` and the rule
are believed (`antecedent ≈ 1`) then `residuum(1, p_B) = p_B`, so the only way
to reach `sat ≈ 1` is `p_B → 1`. The gradient of the energy w.r.t. `p_B` is what
"derives" `B`.

**Default t-norm: Łukasiewicz.** Its residuum `min(1, 1−a+c)` is
piecewise-linear with bounded, well-behaved gradients (no division blow-ups near
`a→0`, unlike the Product residuum). The Product and Gödel families are
implemented and switchable for experimentation.

---

## 3. The energy landscape (EBM)

Given per-rule satisfaction `sat_r ∈ [0, 1]`, define per-rule energy:

```
E_r = exp(β · (1 − sat_r)) − 1
```

- **Valid state** (`sat_r → 1`)  ⇒ `E_r → 0`   (ground state, low energy).
- **Contradiction** (`sat_r → 0`) ⇒ `E_r → exp(β) − 1` (exponentially high energy).

The `−1` offset anchors the ground state at exactly `0`; the `exp(β··)` makes
violations *increasingly expensive*, the discrete-logic intuition that "a false
theorem should require infinite energy to exist" rendered as a smooth penalty
with `β` as an inverse-temperature knob (higher `β` ⇒ sharper, colder, more
rule-like).

Total logical energy of a belief state: `E_logic = mean_r E_r`.

### Parsimony (minimal-model) energy

Implications alone do not pin *unforced* atoms (an implication `A→B` permits `B`
true even when `A` is false). To make the **minimal model** (least fixed point of
forward chaining) the unique energy minimum, we add a small Occam pressure that
defaults atoms to *false* unless a rule forces them true:

```
E_parsimony = mean(p_i)   over derived atoms
```

The rule energy pushes forced atoms up; parsimony pushes everything down; their
equilibrium is exactly the logically-derived truth assignment.

---

## 4. PyTorch component map

```
                 input x  (noisy encoding of observed "cause" atoms)
                    │
        ┌───────────▼─────────────┐
        │   NeuralProposer (MLP)  │   model.py
        │   Linear→ReLU→…→Linear  │
        │   → Sigmoid             │   emits p ∈ [0,1]^A  (probabilistic beliefs)
        └───────────┬─────────────┘
                    │ p  (belief over A atoms)
        ┌───────────▼─────────────┐
        │ DifferentiableLogic     │   logic_engine.py  (DTP)
        │ Engine.satisfaction(p)  │   fuzzy t-norms + residuated implication
        └───────────┬─────────────┘
                    │ sat ∈ [0,1]^R  (per-rule satisfaction)
        ┌───────────▼─────────────┐
        │  EnergyBasedModel       │   model.py  (EBM)
        │  E = exp(β(1−sat))−1    │   + parsimony
        └───────────┬─────────────┘
                    │ scalar energy per sample
        ┌───────────▼─────────────┐
        │   ThermoLogicLoss       │   model.py
        │  w_sup·BCE(causes)      │   supervision anchors beliefs to the input
        │  + w_rule·E_logic       │   logic constrains the derived atoms
        │  + w_par·E_parsimony    │
        └───────────┬─────────────┘
                    ▼   .backward()  → Adam  → beliefs become logically valid
```

| File                   | Component                     | Responsibility |
|------------------------|-------------------------------|----------------|
| `logic_engine.py`      | `TNorm`, residua, `Literal`, `ImplicationRule`, `KnowledgeBase`, `DifferentiableLogicEngine` | Discrete→continuous logic; per-rule satisfaction tensor. |
| `model.py`             | `NeuralProposer`, `EnergyBasedModel`, `ThermoLogicLoss` | Belief network; energy from satisfaction; composite loss. |
| `train.py`             | `build_default_kb`, `generate_dataset`, `train` | Native synthetic data; autonomous mini-batch SGD; metrics. |
| `run.sh`               | pipeline driver               | Zero-intervention execution (deps + train). |

---

## 5. The demonstration (synthetic world)

A tiny weather knowledge base with 5 atoms:

```
atoms   : Rain(0), Cloudy(1), WetGround(2), Sprinkler(3), Slippery(4)
rules   : Rain      → Cloudy
          Rain      → WetGround
          Sprinkler → WetGround
          WetGround → Slippery
          Rain      → ¬Sprinkler          (mutual exclusion; blocks trivial collapse)
causes  : {Rain, Sprinkler}   (observed, supervised from the noisy input)
derived : {Cloudy, WetGround, Slippery}   (must be inferred via the energy)
```

Ground-truth worlds are generated by **forward chaining** (minimal model) over
sampled causes, with the constraint `¬(Rain ∧ Sprinkler)`:
`Cloudy = Rain`, `WetGround = Rain ∨ Sprinkler`, `Slippery = WetGround`.

**What success looks like** (printed by `train.py`):
- total loss and mean energy **decrease** monotonically-ish across epochs;
- mean per-rule **satisfaction → 1**;
- accuracy on the **derived** atoms (never directly supervised — inferred purely
  through the energy) climbs toward **~100%**.

This proves the backpropagation flow: gradients from a *logical* energy term,
routed through differentiable t-norms, teach the network to produce beliefs that
are simultaneously data-consistent (supervision) and logically valid (energy).

---

## 6. Engineering guarantees

- **Autonomous**: no external data or downloads at train time; the dataset is
  synthesized natively from the rule base. `run.sh` runs the whole pipeline.
- **Deterministic**: global seed set for reproducibility.
- **Robust**: numerical clamping keeps truth values in `[0, 1]`, residua avoid
  division blow-ups, and inputs are validated (atom indices, t-norm names,
  probability ranges) with explicit exceptions.
- **Typed & documented**: strict type hints and docstrings throughout.
