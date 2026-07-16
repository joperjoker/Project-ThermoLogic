"""Autonomous training pipeline for Project ThermoLogic.

Running this script end-to-end requires **zero** user intervention and **no**
external data: it synthesizes a logical dataset natively from a small weather
knowledge base, trains the hybrid DTP + EBM model with mini-batch gradient
descent, and prints the decreasing energy/loss together with logical-consistency
and derived-atom-accuracy metrics.

Usage
-----
    python train.py                 # sensible defaults
    python train.py --epochs 60 --tnorm product --seed 7
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from logic_engine import (
    DifferentiableLogicEngine,
    ImplicationRule,
    KnowledgeBase,
    Literal,
    TNorm,
    forward_chaining,
)
from model import EnergyBasedModel, NeuralProposer, ThermoLogicLoss

# Atom indices for the weather knowledge base (see architecture_plan.md).
RAIN, CLOUDY, WET_GROUND, SPRINKLER, SLIPPERY = range(5)


def build_default_kb() -> KnowledgeBase:
    """Construct the 5-atom weather knowledge base used in the demonstration.

    Rules
    -----
    * ``Rain      -> Cloudy``
    * ``Rain      -> WetGround``
    * ``Sprinkler -> WetGround``
    * ``WetGround -> Slippery``
    * ``Rain      -> ¬Sprinkler`` (mutual exclusion; prevents trivial collapse)

    Returns
    -------
    KnowledgeBase
        A validated knowledge base with ``Rain``/``Sprinkler`` as causes and
        ``Cloudy``/``WetGround``/``Slippery`` as derived atoms.
    """
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


@dataclass
class Dataset:
    """A synthetic split of noisy inputs and ground-truth belief targets.

    Attributes
    ----------
    inputs:
        ``(n, input_dim)`` noisy encodings of the cause atoms.
    targets:
        ``(n, num_atoms)`` crisp minimal-model truth assignments.
    input_dim:
        Dimensionality of ``inputs``.
    """

    inputs: Tensor
    targets: Tensor
    input_dim: int


def generate_dataset(
    kb: KnowledgeBase,
    num_samples: int,
    noise_std: float,
    noise_dims: int,
    generator: torch.Generator,
) -> Dataset:
    """Synthesize logically-consistent samples natively from the rule base.

    For each sample a cause assignment is drawn subject to the constraint
    ``¬(Rain ∧ Sprinkler)``; the remaining (derived) atoms are filled in by
    :func:`forward_chaining` to obtain the crisp minimal model. The network input
    is a noisy real-valued encoding of the cause bits padded with pure-noise
    distractor features, so the causes must be *recovered* and the derived atoms
    *inferred* through the energy — nothing is handed the derived truth directly.

    Parameters
    ----------
    kb:
        The knowledge base defining atoms, rules, and cause atoms.
    num_samples:
        Number of samples to generate.
    noise_std:
        Standard deviation of the Gaussian noise added to cause bits.
    noise_dims:
        Number of extra pure-noise distractor input features.
    generator:
        Seeded RNG for reproducibility.

    Returns
    -------
    Dataset
        Inputs and targets ready for a :class:`~torch.utils.data.TensorDataset`.
    """
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if noise_std < 0.0:
        raise ValueError("noise_std must be non-negative")
    if noise_dims < 0:
        raise ValueError("noise_dims must be non-negative")

    num_causes = len(kb.cause_atoms)
    input_dim = num_causes + noise_dims

    inputs = torch.empty((num_samples, input_dim))
    targets = torch.empty((num_samples, kb.num_atoms))

    for i in range(num_samples):
        # Draw causes uniformly, then enforce the mutual-exclusion constraint by
        # dropping the sprinkler whenever it rains (keeps ground truth valid).
        cause_bits = (torch.rand(num_causes, generator=generator) > 0.5).tolist()
        if kb.cause_atoms == (RAIN, SPRINKLER) and cause_bits[0] and cause_bits[1]:
            cause_bits[1] = False

        world: List[bool] = forward_chaining(kb, cause_bits)
        targets[i] = torch.tensor([1.0 if b else 0.0 for b in world])

        cause_vec = torch.tensor([1.0 if b else 0.0 for b in cause_bits])
        noisy_causes = cause_vec + noise_std * torch.randn(
            num_causes, generator=generator
        )
        distractors = torch.randn(noise_dims, generator=generator) if noise_dims else torch.empty(0)
        inputs[i] = torch.cat([noisy_causes, distractors])

    return Dataset(inputs=inputs, targets=targets, input_dim=input_dim)


@dataclass
class EvalMetrics:
    """Evaluation metrics on a dataset split.

    Attributes
    ----------
    energy:
        Mean logical energy (lower ⇒ more logically consistent).
    satisfaction:
        Mean per-rule satisfaction in ``[0, 1]`` (higher is better).
    valid_fraction:
        Fraction of samples where *every* rule is satisfied (``sat > 0.5``).
    derived_accuracy:
        Accuracy of thresholded beliefs on the derived atoms vs. ground truth.
    """

    energy: float
    satisfaction: float
    valid_fraction: float
    derived_accuracy: float


@torch.no_grad()
def evaluate(model: EnergyBasedModel, data: Dataset, kb: KnowledgeBase) -> EvalMetrics:
    """Compute logical-consistency and derived-accuracy metrics for ``model``."""
    model.eval()
    out = model(data.inputs)

    energy = float(out.energy.mean())
    satisfaction = float(out.satisfaction.mean())
    all_rules_ok = (out.satisfaction > 0.5).all(dim=1).float()
    valid_fraction = float(all_rules_ok.mean())

    derived_idx = torch.tensor(kb.derived_atoms, dtype=torch.long)
    pred = (out.beliefs.index_select(1, derived_idx) > 0.5).float()
    truth = data.targets.index_select(1, derived_idx)
    derived_accuracy = float((pred == truth).float().mean())

    return EvalMetrics(
        energy=energy,
        satisfaction=satisfaction,
        valid_fraction=valid_fraction,
        derived_accuracy=derived_accuracy,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments (all optional; defaults run autonomously)."""
    parser = argparse.ArgumentParser(description="Train Project ThermoLogic.")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-samples", type=int, default=2048)
    parser.add_argument("--eval-samples", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--beta", type=float, default=4.0, help="energy sharpness")
    parser.add_argument("--noise-std", type=float, default=0.25)
    parser.add_argument("--noise-dims", type=int, default=3)
    parser.add_argument("--w-supervised", type=float, default=1.0)
    parser.add_argument("--w-rule", type=float, default=1.0)
    parser.add_argument("--w-parsimony", type=float, default=0.1)
    parser.add_argument(
        "--tnorm",
        type=str,
        default=TNorm.LUKASIEWICZ.value,
        choices=[t.value for t in TNorm],
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def train(args: argparse.Namespace) -> EnergyBasedModel:
    """Run the full autonomous training pipeline and return the trained model."""
    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)

    kb = build_default_kb()
    engine = DifferentiableLogicEngine(kb, tnorm=TNorm(args.tnorm))

    train_data = generate_dataset(
        kb, args.train_samples, args.noise_std, args.noise_dims, generator
    )
    eval_data = generate_dataset(
        kb, args.eval_samples, args.noise_std, args.noise_dims, generator
    )

    proposer = NeuralProposer(input_dim=train_data.input_dim, num_atoms=kb.num_atoms)
    model = EnergyBasedModel(proposer, engine, beta=args.beta)
    criterion = ThermoLogicLoss(
        kb,
        w_supervised=args.w_supervised,
        w_rule=args.w_rule,
        w_parsimony=args.w_parsimony,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    loader = DataLoader(
        TensorDataset(train_data.inputs, train_data.targets),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )

    print("=" * 74)
    print("Project ThermoLogic — hybrid Differentiable Theorem Proving + EBM")
    print("=" * 74)
    print(f"knowledge base : {kb.num_atoms} atoms, {kb.num_rules} rules, "
          f"t-norm={engine.tnorm.value}")
    for rule in kb.rules:
        print(f"    {rule}")
    print(f"train/eval     : {args.train_samples}/{args.eval_samples} samples, "
          f"input_dim={train_data.input_dim}, beta={args.beta}")
    print("-" * 74)

    start = evaluate(model, eval_data, kb)
    print(
        f"epoch  00 (init) | loss   ----   | energy {start.energy:7.4f} "
        f"| sat {start.satisfaction:5.3f} | valid {start.valid_fraction:5.3f} "
        f"| derived-acc {start.derived_accuracy:5.3f}"
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = {"total": 0.0, "sup": 0.0, "energy": 0.0, "par": 0.0}
        seen = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            if not torch.isfinite(loss.total):
                raise RuntimeError("non-finite loss encountered; aborting training")
            loss.total.backward()
            optimizer.step()

            bs = xb.shape[0]
            seen += bs
            running["total"] += float(loss.total.detach()) * bs
            running["sup"] += float(loss.supervised) * bs
            running["energy"] += float(loss.rule_energy) * bs
            running["par"] += float(loss.parsimony) * bs

        metrics = evaluate(model, eval_data, kb)
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(
                f"epoch {epoch:3d}       | loss {running['total']/seen:7.4f} "
                f"| energy {metrics.energy:7.4f} | sat {metrics.satisfaction:5.3f} "
                f"| valid {metrics.valid_fraction:5.3f} "
                f"| derived-acc {metrics.derived_accuracy:5.3f}"
            )

    final = evaluate(model, eval_data, kb)
    print("-" * 74)
    print("RESULT — energy minimized, logical validity enforced by construction:")
    print(f"    mean energy         : {start.energy:7.4f}  ->  {final.energy:7.4f}")
    print(f"    rule satisfaction   : {start.satisfaction:7.4f}  ->  {final.satisfaction:7.4f}")
    print(f"    valid-world fraction: {start.valid_fraction:7.4f}  ->  {final.valid_fraction:7.4f}")
    print(f"    derived-atom acc    : {start.derived_accuracy:7.4f}  ->  {final.derived_accuracy:7.4f}")
    print("=" * 74)

    _demonstrate_energy_gap(model, kb)
    return model


@torch.no_grad()
def _demonstrate_energy_gap(model: EnergyBasedModel, kb: KnowledgeBase) -> None:
    """Contrast the energy of a valid belief state vs. a contradictory one.

    This makes the EBM thesis concrete: a logically valid assignment sits at
    ``E ≈ 0`` while a hand-crafted contradiction pays an exponential penalty.
    """
    engine = model.engine
    valid = torch.tensor([[1.0, 1.0, 1.0, 0.0, 1.0]])  # Rain world, minimal model
    # Contradiction: Rain true but Cloudy/WetGround false and Sprinkler on.
    contradiction = torch.tensor([[1.0, 0.0, 0.0, 1.0, 0.0]])

    e_valid = float(model.energy_from_satisfaction(engine.satisfaction(valid)))
    e_bad = float(model.energy_from_satisfaction(engine.satisfaction(contradiction)))
    print("energy landscape probe (fixed belief states):")
    print(f"    valid world  {valid.tolist()[0]} -> energy {e_valid:8.4f}")
    print(f"    contradiction{contradiction.tolist()[0]} -> energy {e_bad:8.4f}")
    print(f"    energy gap (contradiction / valid): {e_bad / max(e_valid, 1e-9):8.1f}x")
    print("=" * 74)


def main() -> None:
    """Entry point: parse arguments and run training."""
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
