"""Where the logic genuinely helps: semi-supervised learning of a HARD rule.

The null result of ``experiments_supervision.py`` was on an *easy* derived logic
with *plentiful* labels. Theory (Xu et al., Semantic Loss, 2018) says a logic
loss should help most when (i) labels are scarce and (ii) the target is hard to
learn from data. We test both at once with **parity** — the canonical function an
MLP generalizes poorly — as a differentiable-logic energy on **unlabeled** data.

Task: ``Z = parity(A0..A3)`` (5 atoms; 16 minterm rules fully define Z). Inputs
are noisy cause bits + 2 distractors.

* labels-only : BCE on a few labelled examples
* + logic     : same, plus the energy (logical inconsistency) on unlabelled data

Both evaluated feed-forward (no test-time repair) on the parity bit. Swept over
the number of labelled examples.

    python experiments_semisup.py
"""

from __future__ import annotations

import itertools
import json
import os
import statistics
from typing import Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from thermologic import (
    DifferentiableLogicEngine,
    EnergyBasedModel,
    ImplicationRule,
    KnowledgeBase,
    Literal,
    NeuralProposer,
    TNorm,
)

FIG_DIR, RES_DIR = "figures", "results"
K = 4  # number of cause bits; parity over them


def parity_kb() -> KnowledgeBase:
    names = tuple([f"A{i}" for i in range(K)] + ["Z"])
    rules = []
    for combo in itertools.product([0, 1], repeat=K):
        parity = sum(combo) % 2
        ante = tuple(Literal(i, negated=(combo[i] == 0)) for i in range(K))
        cons = Literal(K, negated=(parity == 0))  # odd → Z true, even → ¬Z
        rules.append(ImplicationRule(ante, cons, "m" + "".join(map(str, combo))))
    return KnowledgeBase(names, tuple(rules), cause_atoms=tuple(range(K)), derived_atoms=(K,))


def sample(n: int, gen: torch.Generator) -> Tuple[torch.Tensor, torch.Tensor]:
    combos = (torch.rand(n, K, generator=gen) > 0.5).float()
    parity = (combos.sum(1) % 2).unsqueeze(1)
    y = torch.cat([combos, parity], dim=1)
    x = torch.cat([combos + 0.10 * torch.randn(n, K, generator=gen),
                   torch.randn(n, 2, generator=gen)], dim=1)
    return x, y


def train(kb, engine, ebm, n_lab, mode, seed, epochs=400, w=1.0, n_unlab=2000):
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
        if mode == "logic":
            e = ebm.energy_from_satisfaction(engine.satisfaction(prop(xu), validate=False)).mean()
            loss = loss + w * e
        loss.backward()
        opt.step()
    return prop


def parity_acc(prop, gen):
    xt, yt = sample(3000, gen)
    with torch.no_grad():
        pred = (prop(xt)[:, K] > 0.5).float()
    return float((pred == yt[:, K]).float().mean())


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(RES_DIR, exist_ok=True)
    kb = parity_kb()
    engine = DifferentiableLogicEngine(kb, TNorm.LUKASIEWICZ)
    ebm = EnergyBasedModel(NeuralProposer(1, kb.num_atoms), engine, 4.0)
    label_counts = [8, 16, 32, 64, 128, 256]
    seeds = (0, 1, 2)
    res = {"label_counts": label_counts,
           "labels_only": {"mean": [], "std": []},
           "plus_logic": {"mean": [], "std": []}}

    for n_lab in label_counts:
        row = {"labels-only": [], "logic": []}
        for s in seeds:
            for mode, key in (("plain", "labels-only"), ("logic", "logic")):
                prop = train(kb, engine, ebm, n_lab, mode, s)
                row[key].append(parity_acc(prop, torch.Generator().manual_seed(9000 + s)))
        res["labels_only"]["mean"].append(statistics.mean(row["labels-only"]))
        res["labels_only"]["std"].append(statistics.pstdev(row["labels-only"]))
        res["plus_logic"]["mean"].append(statistics.mean(row["logic"]))
        res["plus_logic"]["std"].append(statistics.pstdev(row["logic"]))
        print(f"  labels={n_lab:3d} | labels-only parity acc {res['labels_only']['mean'][-1]:.3f} "
              f"| + logic {res['plus_logic']['mean'][-1]:.3f}")

    with open(os.path.join(RES_DIR, "semisup.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    lo, pl = res["labels_only"], res["plus_logic"]
    ax.errorbar(label_counts, lo["mean"], yerr=lo["std"], fmt="o-", color="#999999",
                capsize=4, lw=2, label="labels-only")
    ax.errorbar(label_counts, pl["mean"], yerr=pl["std"], fmt="s-", color="#CC79A7",
                capsize=4, lw=2, label="+ logic energy on unlabelled data")
    ax.axhline(0.5, color="#D55E00", ls="--", lw=1, label="chance")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("# labelled examples (log scale)")
    ax.set_ylabel("parity-bit accuracy (feed-forward, held-out)")
    ax.set_ylim(0.4, 1.02)
    ax.set_title("Logic genuinely helps: semi-supervised learning of a HARD rule (parity)")
    ax.legend(fontsize=9); ax.grid(alpha=0.25)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig_semisup.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
