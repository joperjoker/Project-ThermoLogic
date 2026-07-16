"""Autonomous training pipeline for Project ThermoLogic.

Runs end-to-end with **zero** user intervention and **no** external data: it
synthesizes a logical dataset natively from a knowledge base, trains the hybrid
DTP + EBM model, and demonstrates the full story:

1. the model learns to derive unobserved atoms (high train accuracy);
2. on **unseen** worlds (a disjoint world-level holdout) the single amortized
   forward pass generalizes only partially and leaves beliefs logically
   *inconsistent* (elevated energy);
3. **test-time energy repair** — gradient descent on the EBM energy — pulls those
   unseen states back to logical validity (energy → 0, accuracy → ~1).

Usage
-----
    python train.py                      # expanded KB + generalization + repair
    python train.py --kb default         # the small 5-atom demo
    python train.py --epochs 60 --seed 7
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from thermologic.data import Dataset, enumerate_worlds, sample_from_worlds, split_worlds
from thermologic.logic_engine import (
    DifferentiableLogicEngine,
    ImplicationRule,
    KnowledgeBase,
    Literal,
    TNorm,
)
from thermologic.model import (
    EnergyBasedModel,
    NeuralProposer,
    ThermoLogicLoss,
    repair_beliefs,
)

# ----- small "default" knowledge base (5 atoms) ----------------------------- #
RAIN, CLOUDY, WET_GROUND, SPRINKLER, SLIPPERY = range(5)


def build_default_kb() -> KnowledgeBase:
    """The compact 5-atom weather KB (used for the quick demo and unit tests)."""
    rules: Tuple[ImplicationRule, ...] = (
        ImplicationRule((Literal(RAIN),), Literal(CLOUDY), "rain→cloudy"),
        ImplicationRule((Literal(RAIN),), Literal(WET_GROUND), "rain→wet"),
        ImplicationRule((Literal(SPRINKLER),), Literal(WET_GROUND), "sprinkler→wet"),
        ImplicationRule((Literal(WET_GROUND),), Literal(SLIPPERY), "wet→slippery"),
        ImplicationRule(
            (Literal(RAIN),), Literal(SPRINKLER, negated=True), "rain→¬sprinkler"
        ),
    )
    return KnowledgeBase(
        atom_names=("Rain", "Cloudy", "WetGround", "Sprinkler", "Slippery"),
        rules=rules,
        cause_atoms=(RAIN, SPRINKLER),
        derived_atoms=(CLOUDY, WET_GROUND, SLIPPERY),
    )


# ----- expanded knowledge base (11 atoms, 18 distinct worlds) --------------- #
(E_RAIN, E_SPRK, E_COLD, E_HEAT, E_WIND,
 E_CLOUD, E_WET, E_SLIP, E_ICE, E_WARM, E_CHILL) = range(11)


def build_expanded_kb() -> KnowledgeBase:
    """A larger KB with 5 independent causes, 6 derived atoms, and 9 rules.

    Two-literal antecedents (``Cold ∧ WetGround → Ice``, ``Windy ∧ Cold →
    Chilly``) exercise fuzzy conjunction; a two-hop chain (``Rain → WetGround →
    Slippery``) exercises transitive derivation; two negative constraints
    (``Rain → ¬Sprinkler``, ``HeaterOn → ¬Cold``) prune the world space. There
    are 18 distinct logically-consistent worlds, enabling a genuine world-level
    train/test split.
    """
    rules: Tuple[ImplicationRule, ...] = (
        ImplicationRule((Literal(E_RAIN),), Literal(E_CLOUD), "rain→cloudy"),
        ImplicationRule((Literal(E_RAIN),), Literal(E_WET), "rain→wet"),
        ImplicationRule((Literal(E_SPRK),), Literal(E_WET), "sprinkler→wet"),
        ImplicationRule((Literal(E_WET),), Literal(E_SLIP), "wet→slippery"),
        ImplicationRule((Literal(E_COLD), Literal(E_WET)), Literal(E_ICE), "cold∧wet→ice"),
        ImplicationRule((Literal(E_HEAT),), Literal(E_WARM), "heater→warm"),
        ImplicationRule((Literal(E_WIND), Literal(E_COLD)), Literal(E_CHILL), "wind∧cold→chilly"),
        ImplicationRule((Literal(E_RAIN),), Literal(E_SPRK, negated=True), "rain→¬sprinkler"),
        ImplicationRule((Literal(E_HEAT),), Literal(E_COLD, negated=True), "heater→¬cold"),
    )
    return KnowledgeBase(
        atom_names=(
            "Rain", "Sprinkler", "Cold", "HeaterOn", "Windy",
            "Cloudy", "WetGround", "Slippery", "Ice", "Warm", "Chilly",
        ),
        rules=rules,
        cause_atoms=(E_RAIN, E_SPRK, E_COLD, E_HEAT, E_WIND),
        derived_atoms=(E_CLOUD, E_WET, E_SLIP, E_ICE, E_WARM, E_CHILL),
    )


def build_kb(name: str) -> KnowledgeBase:
    """Return a knowledge base by name (``"expanded"`` or ``"default"``)."""
    if name == "expanded":
        return build_expanded_kb()
    if name == "default":
        return build_default_kb()
    raise ValueError(f"unknown kb {name!r}; choose 'expanded' or 'default'")


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
@dataclass
class EvalMetrics:
    """Metrics on a dataset split.

    Attributes
    ----------
    energy:
        Mean logical energy (lower ⇒ more logically consistent).
    satisfaction:
        Mean per-rule fuzzy satisfaction in ``[0, 1]``.
    consistency:
        Fraction of samples whose *thresholded* belief world satisfies every rule
        crisply (i.e. would have exactly zero energy). Replaces the old, trivially
        saturated "valid-fraction" metric.
    derived_accuracy:
        Accuracy of thresholded beliefs on derived atoms vs. ground truth.
    per_atom:
        Per-derived-atom accuracy, keyed by atom name.
    """

    energy: float
    satisfaction: float
    consistency: float
    derived_accuracy: float
    per_atom: Dict[str, float] = field(default_factory=dict)


@torch.no_grad()
def evaluate(model: EnergyBasedModel, data: Dataset, kb: KnowledgeBase) -> EvalMetrics:
    """Compute energy, satisfaction, crisp consistency, and derived accuracy."""
    model.eval()
    out = model(data.inputs)

    energy = float(out.energy.mean())
    satisfaction = float(out.satisfaction.mean())

    # Crisp consistency: round beliefs, re-evaluate the rules, require all hold.
    crisp = (out.beliefs > 0.5).float()
    crisp_sat = model.engine.satisfaction(crisp)
    consistency = float((crisp_sat > 0.5).all(dim=1).float().mean())

    derived_idx = torch.tensor(kb.derived_atoms, dtype=torch.long)
    pred = (out.beliefs.index_select(1, derived_idx) > 0.5).float()
    truth = data.targets.index_select(1, derived_idx)
    derived_accuracy = float((pred == truth).float().mean())
    per_atom = {
        kb.atom_names[a]: float((pred[:, j] == truth[:, j]).float().mean())
        for j, a in enumerate(kb.derived_atoms)
    }

    return EvalMetrics(
        energy=energy,
        satisfaction=satisfaction,
        consistency=consistency,
        derived_accuracy=derived_accuracy,
        per_atom=per_atom,
    )


@torch.no_grad()
def cause_accuracy(model: EnergyBasedModel, data: Dataset, kb: KnowledgeBase) -> float:
    """Accuracy of thresholded beliefs on the observed cause atoms."""
    model.eval()
    beliefs = model(data.inputs).beliefs
    idx = torch.tensor(kb.cause_atoms, dtype=torch.long)
    pred = (beliefs.index_select(1, idx) > 0.5).float()
    truth = data.targets.index_select(1, idx)
    return float((pred == truth).float().mean())


def repair_derived(model: EnergyBasedModel, data: Dataset, kb: KnowledgeBase,
                   steps: int = 80, lr: float = 0.2) -> Tuple[Tensor, List[Tuple[int, float]]]:
    """Test-time energy repair holding recovered causes fixed; returns beliefs+trace."""
    with torch.no_grad():
        beliefs = model(data.inputs).beliefs
    fixed_mask = torch.zeros(kb.num_atoms)
    fixed_mask[list(kb.cause_atoms)] = 1.0
    result = repair_beliefs(model, beliefs, fixed_mask, steps=steps, lr=lr)
    return result.beliefs, result.energy_trace


def _derived_accuracy_from_beliefs(beliefs: Tensor, data: Dataset, kb: KnowledgeBase) -> float:
    derived_idx = torch.tensor(kb.derived_atoms, dtype=torch.long)
    pred = (beliefs.index_select(1, derived_idx) > 0.5).float()
    truth = data.targets.index_select(1, derived_idx)
    return float((pred == truth).float().mean())


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments (all optional; defaults run autonomously)."""
    p = argparse.ArgumentParser(description="Train Project ThermoLogic.")
    p.add_argument("--kb", type=str, default="expanded", choices=["expanded", "default"])
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--train-samples", type=int, default=4096)
    p.add_argument("--eval-samples", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--beta", type=float, default=4.0, help="energy sharpness")
    p.add_argument("--noise-std", type=float, default=0.12)
    p.add_argument("--noise-dims", type=int, default=3)
    p.add_argument("--hidden", type=int, nargs="+", default=[96, 96])
    p.add_argument("--holdout", type=int, default=6, help="worlds reserved for testing")
    p.add_argument("--w-supervised", type=float, default=1.0)
    p.add_argument("--w-rule", type=float, default=1.0)
    p.add_argument("--w-parsimony", type=float, default=0.1)
    p.add_argument("--tnorm", type=str, default=TNorm.LUKASIEWICZ.value,
                   choices=[t.value for t in TNorm])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--repair-steps", type=int, default=80)
    return p.parse_args()


def train(args: argparse.Namespace) -> EnergyBasedModel:
    """Run the full autonomous pipeline and return the trained model."""
    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)

    kb = build_kb(args.kb)
    engine = DifferentiableLogicEngine(kb, tnorm=TNorm(args.tnorm))

    worlds = enumerate_worlds(kb, engine)
    holdout = min(args.holdout, max(1, len(worlds) // 3))
    holdout = min(holdout, len(worlds) - 1)
    split = split_worlds(worlds, holdout=holdout, seed=0)

    train_data = sample_from_worlds(
        kb, split.train_worlds, args.train_samples, args.noise_std, args.noise_dims, generator
    )
    # Evaluate on seen (train) worlds and unseen (held-out) worlds separately.
    seen_eval = sample_from_worlds(
        kb, split.train_worlds, args.eval_samples, args.noise_std, args.noise_dims, generator
    )
    unseen_eval = sample_from_worlds(
        kb, split.test_worlds, args.eval_samples, args.noise_std, args.noise_dims, generator
    )

    proposer = NeuralProposer(train_data.input_dim, kb.num_atoms, tuple(args.hidden))
    model = EnergyBasedModel(proposer, engine, beta=args.beta)
    criterion = ThermoLogicLoss(kb, args.w_supervised, args.w_rule, args.w_parsimony)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    loader = DataLoader(
        TensorDataset(train_data.inputs, train_data.targets),
        batch_size=args.batch_size, shuffle=True, generator=generator,
    )

    print("=" * 78)
    print("Project ThermoLogic — hybrid Differentiable Theorem Proving + EBM")
    print("=" * 78)
    print(f"knowledge base : {kb.num_atoms} atoms, {kb.num_rules} rules, "
          f"t-norm={engine.tnorm.value}, β={args.beta}")
    for rule in kb.rules:
        print(f"    {rule}")
    print(f"worlds         : {len(worlds)} distinct  →  "
          f"{len(split.train_worlds)} train / {len(split.test_worlds)} UNSEEN (held out)")
    print(f"data           : {args.train_samples} train samples, "
          f"input_dim={train_data.input_dim}, noise σ={args.noise_std}")
    print("-" * 78)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            if not torch.isfinite(loss.total):
                raise RuntimeError("non-finite loss encountered; aborting")
            loss.total.backward()
            optimizer.step()
            running += float(loss.total.detach()) * xb.shape[0]
            seen += xb.shape[0]

        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            m_seen = evaluate(model, seen_eval, kb)
            m_uns = evaluate(model, unseen_eval, kb)
            print(
                f"epoch {epoch:3d} | loss {running/seen:6.4f} | "
                f"train-acc {m_seen.derived_accuracy:5.3f} | "
                f"UNSEEN-acc {m_uns.derived_accuracy:5.3f} "
                f"(E={m_uns.energy:6.4f}, consist={m_uns.consistency:4.2f})"
            )

    # ----- final generalization + test-time repair ------------------------- #
    m_seen = evaluate(model, seen_eval, kb)
    m_uns = evaluate(model, unseen_eval, kb)
    repaired, trace = repair_derived(model, unseen_eval, kb, steps=args.repair_steps)
    unseen_repaired_acc = _derived_accuracy_from_beliefs(repaired, unseen_eval, kb)
    repaired_energy = trace[-1][1]

    print("-" * 78)
    print("GENERALIZATION (derived-atom accuracy on worlds never seen in training):")
    print(f"    seen worlds       : acc {m_seen.derived_accuracy:5.3f}  "
          f"energy {m_seen.energy:7.4f}")
    print(f"    unseen (feed-fwd) : acc {m_uns.derived_accuracy:5.3f}  "
          f"energy {m_uns.energy:7.4f}   ← amortized pass leaves states inconsistent")
    print(f"    unseen (repaired) : acc {unseen_repaired_acc:5.3f}  "
          f"energy {repaired_energy:7.4f}   ← test-time energy descent repairs them")
    print(f"    cause-acc (unseen): {cause_accuracy(model, unseen_eval, kb):5.3f} "
          f"(caps derived accuracy)")
    print("    per-derived-atom (unseen, repaired):")
    rep_metrics = evaluate_beliefs(repaired, unseen_eval, kb, model)
    for name, acc in rep_metrics.per_atom.items():
        print(f"        {name:10s}: {acc:5.3f}")
    print("-" * 78)
    _demonstrate_energy_gap(model, kb)
    return model


@torch.no_grad()
def evaluate_beliefs(beliefs: Tensor, data: Dataset, kb: KnowledgeBase,
                     model: EnergyBasedModel) -> EvalMetrics:
    """Like :func:`evaluate` but for an externally-supplied belief tensor."""
    energy = float(model.energy_from_satisfaction(model.engine.satisfaction(beliefs)).mean())
    satisfaction = float(model.engine.satisfaction(beliefs).mean())
    crisp = (beliefs > 0.5).float()
    consistency = float((model.engine.satisfaction(crisp) > 0.5).all(dim=1).float().mean())
    derived_idx = torch.tensor(kb.derived_atoms, dtype=torch.long)
    pred = (beliefs.index_select(1, derived_idx) > 0.5).float()
    truth = data.targets.index_select(1, derived_idx)
    per_atom = {
        kb.atom_names[a]: float((pred[:, j] == truth[:, j]).float().mean())
        for j, a in enumerate(kb.derived_atoms)
    }
    return EvalMetrics(energy, satisfaction, consistency,
                       float((pred == truth).float().mean()), per_atom)


@torch.no_grad()
def _demonstrate_energy_gap(model: EnergyBasedModel, kb: KnowledgeBase) -> None:
    """Contrast the energy of a valid belief state vs. a contradictory one."""
    engine = model.engine
    valid = torch.zeros(1, kb.num_atoms)
    contradiction = torch.zeros(1, kb.num_atoms)
    # Build one valid world (all-false is always the minimal model of no causes)
    # and one deliberate contradiction (an implication with true body, false head).
    rule = kb.rules[0]
    for lit in rule.antecedents:
        contradiction[0, lit.atom] = 0.0 if lit.negated else 1.0
    contradiction[0, rule.consequent.atom] = 1.0 if rule.consequent.negated else 0.0

    e_valid = float(model.energy_from_satisfaction(engine.satisfaction(valid)))
    e_bad = float(model.energy_from_satisfaction(engine.satisfaction(contradiction)))
    print("energy probe (fixed belief states):")
    print(f"    valid (all-false)  -> energy {e_valid:8.4f}")
    print(f"    contradiction      -> energy {e_bad:8.4f}   "
          f"(violates rule: {rule.name})")
    print("=" * 78)


def main() -> None:
    """Entry point: parse arguments and run training."""
    train(parse_args())


if __name__ == "__main__":
    main()
