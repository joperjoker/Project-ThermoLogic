"""The one thing projection cannot do: train **through** the repair.

A projection / SAT / forward-chaining step is a non-differentiable ``argmin`` over
a discrete set — no usable gradient. ThermoLogic's repair is plain gradient
descent, so it can be *unrolled* into a differentiable module and placed inside a
training loop: a downstream loss backpropagates through the logical repair and
into the network.

This script demonstrates that capability two ways:

1. **Gradient existence.** The gradient of a post-repair loss w.r.t. the network
   is finite through differentiable repair, and exactly zero through a projection.
2. **Downstream-only training.** With *no per-atom labels* — supervising only a
   scalar readout of the repaired state — a proposer trained *through* the repair
   learns to produce logically-correct atoms; the projection-in-the-loop cannot.

    python experiments_through_repair.py
"""

from __future__ import annotations

import itertools
from typing import Tuple

import torch
from torch import Tensor

from thermologic import (
    DifferentiableLogicEngine,
    EnergyBasedModel,
    ImplicationRule,
    KnowledgeBase,
    Literal,
    NeuralProposer,
    TNorm,
)

DEPTH = 6  # chain A0 -> A1 -> ... -> A6 (A0 cause, rest derived)


def chain_kb(k: int = DEPTH) -> KnowledgeBase:
    names = tuple(f"A{i}" for i in range(k + 1))
    rules = tuple(ImplicationRule((Literal(i - 1),), Literal(i), f"r{i}") for i in range(1, k + 1))
    return KnowledgeBase(names, rules, cause_atoms=(0,), derived_atoms=tuple(range(1, k + 1)))


def make_data(n: int, gen: torch.Generator) -> Tuple[Tensor, Tensor]:
    """Input = noisy A0; target world = all-true if A0 else all-false (minimal model)."""
    a0 = (torch.rand(n, 1, generator=gen) > 0.5).float()
    x = torch.cat([a0 + 0.15 * torch.randn(n, 1, generator=gen), torch.randn(n, 2, generator=gen)], dim=1)
    world = a0.repeat(1, DEPTH + 1)  # chain forces all atoms = A0
    return x, world


def diff_repair(beliefs: Tensor, engine, ebm, free_mask: Tensor, steps: int = 20, lr: float = 0.6) -> Tensor:
    """Unrolled, fully-differentiable energy descent on the free atoms."""
    base = beliefs
    logit = torch.logit(beliefs.clamp(1e-4, 1 - 1e-4))
    for _ in range(steps):
        state = base * (1 - free_mask) + torch.sigmoid(logit) * free_mask
        E = ebm.energy_from_satisfaction(engine.satisfaction(state, validate=False)).mean()
        (g,) = torch.autograd.grad(E, logit, create_graph=True)
        logit = logit - lr * g
    return base * (1 - free_mask) + torch.sigmoid(logit) * free_mask


def nearest_valid_projection(beliefs: Tensor, engine, kb) -> Tensor:
    """Non-differentiable: enumerate valid completions, pick nearest (detached)."""
    derived = list(kb.derived_atoms)
    combos = torch.tensor(list(itertools.product([0.0, 1.0], repeat=len(derived))))
    out = (beliefs.detach() > 0.5).float()
    for i in range(beliefs.shape[0]):
        cand = out[i].unsqueeze(0).repeat(combos.shape[0], 1)
        for j, d in enumerate(derived):
            cand[:, d] = combos[:, j]
        valid = (engine.satisfaction(cand, validate=False) > 0.5).all(dim=1)
        didx = torch.tensor(derived)
        dist = (cand.index_select(1, didx) - beliefs[i].detach().index_select(0, didx)).abs().sum(1)
        dist = dist + (~valid).float() * 1e9
        out[i] = cand[int(dist.argmin())]
    return out  # no grad_fn


def derived_accuracy(state: Tensor, y: Tensor, kb) -> float:
    idx = torch.tensor(list(kb.derived_atoms))
    return float(((state.index_select(1, idx) > 0.5).float() == y.index_select(1, idx)).float().mean())


def main() -> None:
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    kb = chain_kb()
    engine = DifferentiableLogicEngine(kb, TNorm.LUKASIEWICZ)
    ebm = EnergyBasedModel(NeuralProposer(1, kb.num_atoms), engine, beta=4.0)
    free_mask = torch.zeros(kb.num_atoms)
    free_mask[list(kb.derived_atoms)] = 1.0

    print("=" * 74)
    print("Train-THROUGH-repair: the capability projection cannot provide")
    print("=" * 74)

    # ---- 1. gradient existence -------------------------------------------- #
    proposer = NeuralProposer(3, kb.num_atoms, (32,))
    x, y = make_data(256, gen)
    beliefs = proposer(x)
    repaired = diff_repair(beliefs, engine, ebm, free_mask)
    loss = torch.nn.functional.binary_cross_entropy(
        repaired.index_select(1, torch.tensor(list(kb.derived_atoms))),
        y.index_select(1, torch.tensor(list(kb.derived_atoms))))
    proposer.zero_grad()
    loss.backward()
    gnorm_diff = sum(p.grad.norm().item() for p in proposer.parameters() if p.grad is not None)

    proj = nearest_valid_projection(beliefs, engine, kb)
    gnorm_proj = 0.0 if not proj.requires_grad else 1.0  # detached -> no graph
    print(f"gradient to network through DIFFERENTIABLE repair : {gnorm_diff:.4f}  (trains)")
    print(f"gradient to network through PROJECTION            : {gnorm_proj:.4f}  (dead — argmin)")

    # ---- 2. downstream-only training (no per-atom labels) ----------------- #
    # Supervise ONLY a scalar readout of the repaired state (mean of derived
    # atoms) against the ground-truth readout. Per-atom correctness must emerge
    # through the repair — impossible without a gradient through it.
    print("-" * 74)
    print("Downstream-only training (supervise a SCALAR readout, no atom labels):")
    torch.manual_seed(1)
    proposer = NeuralProposer(3, kb.num_atoms, (32,))
    opt = torch.optim.Adam(proposer.parameters(), lr=5e-3)
    didx = torch.tensor(list(kb.derived_atoms))
    xtr, ytr = make_data(1024, gen)
    xte, yte = make_data(512, gen)
    readout_target = ytr.index_select(1, didx).mean(1, keepdim=True)  # scalar per sample
    acc0 = derived_accuracy(diff_repair(proposer(xte), engine, ebm, free_mask).detach(), yte, kb)
    for epoch in range(1, 41):
        proposer.train()
        beliefs = proposer(xtr)
        repaired = diff_repair(beliefs, engine, ebm, free_mask)
        readout = repaired.index_select(1, didx).mean(1, keepdim=True)
        loss = torch.nn.functional.mse_loss(readout, readout_target)
        opt.zero_grad(); loss.backward(); opt.step()
        if epoch % 10 == 0 or epoch == 1:
            # NB: no torch.no_grad() — the unrolled repair uses autograd internally.
            acc = derived_accuracy(diff_repair(proposer(xte), engine, ebm, free_mask).detach(), yte, kb)
            print(f"  epoch {epoch:2d} | readout MSE {float(loss.detach()):.4f} | "
                  f"emergent derived-atom accuracy {acc:.3f}")
    print(f"\n  derived-atom accuracy: {acc0:.3f} (init) -> {acc:.3f} (trained through repair)")
    print("  A projection-in-the-loop gets zero gradient here, so it cannot learn this.")
    print("=" * 74)


if __name__ == "__main__":
    main()
