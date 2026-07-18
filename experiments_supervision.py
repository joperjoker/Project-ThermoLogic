"""Does post-repair supervision beat pre-repair supervision on held-out data?

The strong claim behind training-through-repair: inserting the differentiable
logic repair into the training loop acts as an **inductive bias**, so the network
should generalise to *unseen worlds* better than one that must learn the derived
logic from data — most when the training data covers few worlds, and less as
coverage grows.

Fair comparison (identical architecture, init, optimiser, epochs, labels; only the
training loss differs), swept over the number of distinct training worlds:

* pre-repair  : loss = BCE(proposer(x),               y)   — learn every atom from data
* post-repair : loss = BCE(repair(proposer(x)),       y)   — logic enforced in the loop

Both are evaluated identically on 6 held-out worlds, each with test-time repair
(so the contrast is the *training* signal, not who gets repair at inference).

    python experiments_supervision.py
"""

from __future__ import annotations

import json
import os
import statistics
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from thermologic import (
    DifferentiableLogicEngine,
    EnergyBasedModel,
    NeuralProposer,
    TNorm,
    enumerate_worlds,
    sample_from_worlds,
    split_worlds,
)
from train import build_expanded_kb
from experiments_through_repair import diff_repair

FIG_DIR, RES_DIR = "figures", "results"
NOISE, NOISE_DIMS = 0.12, 3


def _acc(state, y, idx):
    return float(((state.index_select(1, idx) > 0.5).float() == y.index_select(1, idx)).float().mean())


def run_one(kb, engine, ebm, free_mask, train_worlds, seed, mode,
            epochs=70, n_train=1536, lr=5e-3, repair_steps=15):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    tr = sample_from_worlds(kb, train_worlds, n_train, NOISE, NOISE_DIMS, gen)
    proposer = NeuralProposer(tr.input_dim, kb.num_atoms, (96, 96))
    opt = torch.optim.Adam(proposer.parameters(), lr=lr)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(tr.inputs, tr.targets),
        batch_size=128, shuffle=True, generator=gen)
    for _ in range(epochs):
        proposer.train()
        for xb, yb in loader:
            opt.zero_grad()
            beliefs = proposer(xb)
            if mode == "post":
                beliefs = diff_repair(beliefs, engine, ebm, free_mask, steps=repair_steps, lr=0.6)
            loss = torch.nn.functional.binary_cross_entropy(beliefs, yb)
            loss.backward()
            opt.step()
    return proposer


def evaluate(proposer, kb, engine, ebm, free_mask, test_worlds, seed):
    gen = torch.Generator().manual_seed(1000 + seed)
    ev = sample_from_worlds(kb, test_worlds, 1024, NOISE, NOISE_DIMS, gen)
    didx = torch.tensor(list(kb.derived_atoms))
    cidx = torch.tensor(list(kb.cause_atoms))
    beliefs = proposer(ev.inputs)
    repaired = diff_repair(beliefs, engine, ebm, free_mask, steps=40, lr=0.6).detach()
    crisp = (repaired > 0.5).float()
    consistency = float((engine.satisfaction(crisp, validate=False) > 0.5).all(1).float().mean())
    return {
        "derived_ff": _acc(beliefs.detach(), ev.targets, didx),      # raw network
        "derived_repaired": _acc(repaired, ev.targets, didx),        # with test-time repair
        "cause": _acc(beliefs.detach(), ev.targets, cidx),
        "consistency_repaired": consistency,
    }


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(RES_DIR, exist_ok=True)
    kb = build_expanded_kb()
    engine = DifferentiableLogicEngine(kb, TNorm.LUKASIEWICZ)
    ebm = EnergyBasedModel(NeuralProposer(1, kb.num_atoms), engine, 4.0)
    free_mask = torch.zeros(kb.num_atoms)
    free_mask[list(kb.derived_atoms)] = 1.0

    all_worlds = enumerate_worlds(kb, engine)
    split = split_worlds(all_worlds, holdout=6, seed=0)
    train_pool, test_worlds = split.train_worlds, split.test_worlds  # 12 / 6
    world_counts = [2, 4, 6, 8, 12]
    seeds = (0, 1, 2)

    results = {"world_counts": world_counts, "pre": {}, "post": {}}
    metrics = ["derived_repaired", "derived_ff", "cause", "consistency_repaired"]
    for mode in ("pre", "post"):
        for m in metrics:
            results[mode][m] = {"mean": [], "std": []}

    for W in world_counts:
        tw = train_pool[:W]
        for mode in ("pre", "post"):
            per = {m: [] for m in metrics}
            for s in seeds:
                prop = run_one(kb, engine, ebm, free_mask, tw, s, mode)
                ev = evaluate(prop, kb, engine, ebm, free_mask, test_worlds, s)
                for m in metrics:
                    per[m].append(ev[m])
            for m in metrics:
                results[mode][m]["mean"].append(statistics.mean(per[m]))
                results[mode][m]["std"].append(statistics.pstdev(per[m]) if len(per[m]) > 1 else 0.0)
        print(f"  W={W:2d} worlds | "
              f"pre derived(+repair) {results['pre']['derived_repaired']['mean'][-1]:.3f} | "
              f"post {results['post']['derived_repaired']['mean'][-1]:.3f} | "
              f"pre consist {results['pre']['consistency_repaired']['mean'][-1]:.2f} "
              f"post {results['post']['consistency_repaired']['mean'][-1]:.2f}")

    with open(os.path.join(RES_DIR, "supervision.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    # ---- figure ---- #
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    pre = results["pre"]["derived_repaired"]
    post = results["post"]["derived_repaired"]
    ax.errorbar(world_counts, pre["mean"], yerr=pre["std"], fmt="o-", color="#999999",
                capsize=4, lw=2, label="pre-repair supervision")
    ax.errorbar(world_counts, post["mean"], yerr=post["std"], fmt="s-", color="#CC79A7",
                capsize=4, lw=2, label="post-repair supervision (through repair)")
    ax.set_xlabel("# distinct training worlds seen  (of 12; 6 held out)")
    ax.set_ylabel("held-out derived accuracy  (both + test-time repair)")
    ax.set_title("Logic-in-the-loop generalises from fewer worlds")
    ax.legend(fontsize=9); ax.grid(alpha=0.25)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig_supervision.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
