"""Novelty check: our fuzzy-logic *energy* vs. Xu et al.'s **Semantic Loss**.

The +26-point semi-supervised win (`experiments_semisup.py`) raises the obvious
question: is a differentiable t-norm *energy* actually doing anything that the
established **Semantic Loss** (Xu, Zhang, Friedman, Liang, Van den Broeck, ICML
2018) does not? Semantic Loss is the *principled* object here — the negative log
of the probability that an independent-Bernoulli sample of the network's outputs
satisfies the constraint (an exact weighted model count, WMC). If our energy just
approximates it, honesty demands we say so; if it is competitive, that is a
modest but real result.

We run **three arms on the identical task, net, and unlabelled data** — only the
loss on the unlabelled batch differs:

* ``labels-only``   — BCE on a few labelled examples
* ``+ fuzzy energy`` — ours: Łukasiewicz t-norm energy on unlabelled data
* ``+ semantic loss`` — Xu et al.: −log WMC(constraint) on unlabelled data

Task = parity(A0..A3) (16 models, so the WMC is exact and cheap). Whichever wins,
we report it.

    python experiments_semanticloss.py
"""

from __future__ import annotations

import itertools
import json
import os
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from thermologic import (
    DifferentiableLogicEngine,
    EnergyBasedModel,
    NeuralProposer,
    TNorm,
)
from experiments_semisup import K, parity_kb, sample, parity_acc

FIG_DIR, RES_DIR = "figures", "results"
# Precompute the 16 parity minterms once: (assignment tuple, parity bit).
_MINTERMS = [(combo, sum(combo) % 2) for combo in itertools.product([0, 1], repeat=K)]


def semantic_loss(probs: torch.Tensor) -> torch.Tensor:
    """Exact Semantic Loss for ``Z = parity(A0..A3)`` (Xu et al., 2018).

    ``-log`` of the probability that an independent-Bernoulli draw from the
    network's per-atom outputs satisfies the constraint = ``-log WMC``.
    """
    p = probs.clamp(1e-6, 1.0 - 1e-6)
    pa, pz = p[:, :K], p[:, K]
    wmc = torch.zeros(probs.shape[0])
    for combo, par in _MINTERMS:
        term = torch.ones(probs.shape[0])
        for i, a in enumerate(combo):
            term = term * (pa[:, i] if a == 1 else (1.0 - pa[:, i]))
        term = term * (pz if par == 1 else (1.0 - pz))
        wmc = wmc + term
    return -(wmc.clamp_min(1e-12).log()).mean()


def train(kb, engine, ebm, n_lab, mode, seed, epochs=400, w=1.0, n_unlab=2000):
    """mode in {'plain', 'energy', 'semantic'} — identical except the unlabelled loss."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    xl, yl = sample(n_lab, gen)
    xu, _ = sample(n_unlab, gen)
    prop = NeuralProposer(K + 2, kb.num_atoms, (64, 64))
    opt = torch.optim.Adam(prop.parameters(), lr=5e-3)
    bce = torch.nn.BCELoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = bce(prop(xl), yl)
        if mode == "energy":
            loss = loss + w * ebm.energy_from_satisfaction(
                engine.satisfaction(prop(xu), validate=False)).mean()
        elif mode == "semantic":
            loss = loss + w * semantic_loss(prop(xu))
        loss.backward()
        opt.step()
    return prop


def weight_sweep(kb, engine, ebm, n_lab=64, seeds=(0, 1, 2), weights=(0.5, 1.0, 2.0, 4.0)):
    """Fairness check: does tuning the loss weight let Semantic Loss catch up?

    Both losses have different natural scales, so a single matched weight is not a
    fair verdict. We sweep the weight for both arms and report the curve. (Finding:
    the fuzzy energy is weight-robust; Semantic Loss peaks at a small weight and
    degrades as it grows — so the ranking is not a weighting artifact.)
    """
    res = {"n_lab": n_lab, "weights": list(weights), "energy": [], "semantic": []}
    print("\nWEIGHT SENSITIVITY (n_lab=%d, mean acc over %d seeds):" % (n_lab, len(seeds)))
    print(f"  {'w':>6}{'energy (ours)':>15}{'semantic (Xu)':>15}")
    for w in weights:
        e = statistics.mean(parity_acc(train(kb, engine, ebm, n_lab, "energy", s, w=w),
                                       torch.Generator().manual_seed(9000 + s)) for s in seeds)
        sl = statistics.mean(parity_acc(train(kb, engine, ebm, n_lab, "semantic", s, w=w),
                                        torch.Generator().manual_seed(9000 + s)) for s in seeds)
        res["energy"].append(e); res["semantic"].append(sl)
        print(f"  {w:>6}{e:>15.3f}{sl:>15.3f}")
    with open(os.path.join(RES_DIR, "semisup_sl_weight.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(res["weights"], res["energy"], "s-", color="#CC79A7", lw=2, label="+ fuzzy energy (ours)")
    ax.plot(res["weights"], res["semantic"], "^-", color="#0072B2", lw=2, label="+ semantic loss (Xu et al.)")
    ax.set_xlabel("loss weight w on the unlabelled term")
    ax.set_ylabel(f"parity-bit accuracy (n={n_lab} labels)")
    ax.set_title("Robustness to the loss weight (parity, 3 seeds)")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_semanticloss_weight.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote figures/fig_semanticloss_weight.png")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(RES_DIR, exist_ok=True)
    kb = parity_kb()
    engine = DifferentiableLogicEngine(kb, TNorm.LUKASIEWICZ)
    ebm = EnergyBasedModel(NeuralProposer(1, kb.num_atoms), engine, 4.0)
    label_counts = [8, 16, 32, 64, 128]
    seeds = (0, 1, 2)
    arms = [("plain", "labels-only"), ("energy", "+ fuzzy energy (ours)"),
            ("semantic", "+ semantic loss (Xu et al.)")]
    res = {"label_counts": label_counts, "arms": {k: {"mean": [], "std": []} for k, _ in arms}}

    print("=" * 74)
    print("Semi-supervised parity — fuzzy energy vs. Semantic Loss (3 seeds)")
    print("=" * 74)
    print(f"  {'# labels':<10}{'labels-only':>14}{'+ fuzzy (ours)':>16}{'+ semantic':>14}")
    for n_lab in label_counts:
        cells = {}
        for mode, _ in arms:
            accs = [parity_acc(train(kb, engine, ebm, n_lab, mode, s),
                               torch.Generator().manual_seed(9000 + s)) for s in seeds]
            res["arms"][mode]["mean"].append(statistics.mean(accs))
            res["arms"][mode]["std"].append(statistics.pstdev(accs))
            cells[mode] = statistics.mean(accs)
        print(f"  {n_lab:<10}{cells['plain']:>14.3f}{cells['energy']:>16.3f}{cells['semantic']:>14.3f}")

    with open(os.path.join(RES_DIR, "semisup_sl.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)

    fig, ax = plt.subplots(figsize=(7.8, 4.7))
    styles = {"plain": ("o-", "#999999"), "energy": ("s-", "#CC79A7"),
              "semantic": ("^-", "#0072B2")}
    for mode, label in arms:
        m, st = res["arms"][mode]["mean"], res["arms"][mode]["std"]
        fmt, col = styles[mode]
        ax.errorbar(label_counts, m, yerr=st, fmt=fmt, color=col, capsize=4, lw=2, label=label)
    ax.axhline(0.5, color="#D55E00", ls="--", lw=1, label="chance")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("# labelled examples (log scale)")
    ax.set_ylabel("parity-bit accuracy (feed-forward, held-out)")
    ax.set_ylim(0.4, 1.02)
    ax.set_title("Is the fuzzy energy just Semantic Loss? Head-to-head on the same task")
    ax.legend(fontsize=9); ax.grid(alpha=0.25)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig_semanticloss.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)
    weight_sweep(kb, engine, ebm)
    print("=" * 74)


if __name__ == "__main__":
    main()
