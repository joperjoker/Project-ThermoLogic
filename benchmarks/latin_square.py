"""Head-to-head on a canonical, NON-circular constraint problem: Latin squares.

Addresses the sharpest critiques of the cloud benchmark — *self-generated ground
truth* and *toy scale/structure* — by using a task where:

* ground truth = real order-4 Latin squares (576 of them), generated independently
  of the logic rules;
* the constraints (rows/cols all-different) are a *property* of the answer, not
  the process that makes the labels;
* recovering the grid from a **noisy observation** genuinely needs *both* the
  learned prior (perception) *and* the constraints (many grids satisfy the rules;
  the observation is ambiguous) — the neuro-symbolic sweet spot.

We compare four approaches, and report honestly whichever wins:

1. **ML-only**              — a per-cell-softmax net; argmax (may violate rules)
2. **ML + exact solver**    — snap the net output to the nearest of the 576 valid
                              grids (the optimal symbolic min-change repair)
3. **ML + energy repair**   — ours, at runtime
4. **ML + logic loss**      — ours, at TRAINING time (semi-supervised, scarce labels)

    python benchmarks/latin_square.py
"""

from __future__ import annotations

import json
import os
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import Tensor, nn

from thermologic import (
    DifferentiableLogicEngine,
    EnergyBasedModel,
    ImplicationRule,
    KnowledgeBase,
    Literal,
    NeuralProposer,
    TNorm,
    repair_beliefs,
)

N = 4
FIG_DIR, RES_DIR = "figures", "results"


def aidx(r, c, v):
    return (r * N + c) * N + v


def build_latin_kb() -> KnowledgeBase:
    names = tuple(f"x{r}{c}{v}" for r in range(N) for c in range(N) for v in range(N))
    rules = []
    for r in range(N):
        for c in range(N):
            for v in range(N):
                for w in range(v + 1, N):
                    rules.append(ImplicationRule((Literal(aidx(r, c, v)),), Literal(aidx(r, c, w), True), "cell"))
    for r in range(N):
        for v in range(N):
            for c in range(N):
                for d in range(c + 1, N):
                    rules.append(ImplicationRule((Literal(aidx(r, c, v)),), Literal(aidx(r, d, v), True), "row"))
    for c in range(N):
        for v in range(N):
            for r in range(N):
                for s in range(r + 1, N):
                    rules.append(ImplicationRule((Literal(aidx(r, c, v)),), Literal(aidx(s, c, v), True), "col"))
    return KnowledgeBase(names, tuple(rules), derived_atoms=tuple(range(N * N * N)))


def all_latin_squares() -> List[List[List[int]]]:
    grids, grid = [], [[-1] * N for _ in range(N)]

    def ok(r, c, v):
        return all(grid[r][cc] != v for cc in range(N)) and all(grid[rr][c] != v for rr in range(N))

    def bt(pos):
        if pos == N * N:
            grids.append([row[:] for row in grid]); return
        r, c = divmod(pos, N)
        for v in range(N):
            if ok(r, c, v):
                grid[r][c] = v; bt(pos + 1); grid[r][c] = -1
    bt(0)
    return grids


def grid_to_onehot(g) -> Tensor:
    t = torch.zeros(N * N * N)
    for r in range(N):
        for c in range(N):
            t[aidx(r, c, g[r][c])] = 1.0
    return t


def is_latin(oh: Tensor) -> bool:
    """Does a (64,) argmax-per-cell grid form a valid Latin square?"""
    g = oh.view(N * N, N).argmax(1).view(N, N)
    for i in range(N):
        if len(set(g[i].tolist())) != N or len(set(g[:, i].tolist())) != N:
            return False
    return True


class LatinNet(nn.Module):
    """MLP with per-cell softmax so 'exactly one value per cell' is built in."""

    def __init__(self, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(N * N * N, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, N * N * N))

    def forward(self, x):
        logits = self.net(x).view(-1, N * N, N)
        return torch.softmax(logits, dim=2).reshape(-1, N * N * N)


def make_data(n, noise, gen, squares) -> Tuple[Tensor, Tensor]:
    idx = torch.randint(len(squares), (n,), generator=gen)
    y = torch.stack([grid_to_onehot(squares[int(i)]) for i in idx])
    x = y + noise * torch.randn(n, N * N * N, generator=gen)
    return x, y


def cell_acc(pred_oh, y):
    p = pred_oh.view(-1, N * N, N).argmax(2)
    t = y.view(-1, N * N, N).argmax(2)
    return float((p == t).float().mean())


def grid_exact(pred_oh, y):
    p = pred_oh.view(-1, N * N, N).argmax(2)
    t = y.view(-1, N * N, N).argmax(2)
    return float((p == t).all(1).float().mean())


def validity(pred_oh):
    return float(sum(is_latin(pred_oh[i]) for i in range(pred_oh.shape[0])) / pred_oh.shape[0])


def solver_project(soft, valid_mat):
    """Exact symbolic repair: nearest of the 576 valid grids (max overlap)."""
    scores = soft @ valid_mat.T          # [batch, 576]
    best = scores.argmax(1)
    return valid_mat[best]


def train_net(kb, engine, ebm, squares, n_lab, noise, seed, epochs, logic_w=0.0, n_unlab=2000):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    xl, yl = make_data(n_lab, noise, gen, squares)
    xu, _ = make_data(n_unlab, noise, gen, squares) if logic_w > 0 else (None, None)
    net = LatinNet()
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    for _ in range(epochs):
        opt.zero_grad()
        loss = -(yl * torch.log(net(xl).clamp_min(1e-6))).sum(1).mean()   # cross-entropy
        if logic_w > 0:
            e = ebm.energy_from_satisfaction(engine.satisfaction(net(xu), validate=False)).mean()
            loss = loss + logic_w * e
        loss.backward()
        opt.step()
    net.eval()
    return net


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(RES_DIR, exist_ok=True)
    kb = build_latin_kb()
    engine = DifferentiableLogicEngine(kb, TNorm.LUKASIEWICZ)
    ebm = EnergyBasedModel(NeuralProposer(1, kb.num_atoms), engine, 4.0)
    squares = all_latin_squares()
    valid_mat = torch.stack([grid_to_onehot(s) for s in squares])
    NOISE = 0.9
    gen = torch.Generator().manual_seed(999)

    print("=" * 78)
    print(f"Latin-square head-to-head — {kb.num_atoms} atoms, {kb.num_rules} rules, "
          f"{len(squares)} valid grids, obs noise σ={NOISE}")
    print("=" * 78)

    # ---- runtime comparison: train one decent net, apply 3 inference methods ---- #
    net = train_net(kb, engine, ebm, squares, n_lab=3000, noise=NOISE, seed=0, epochs=400)
    xte, yte = make_data(2000, NOISE, gen, squares)
    with torch.no_grad():
        soft = net(xte)
    ml = soft
    solv = solver_project(soft, valid_mat)
    fixed = torch.zeros(kb.num_atoms)
    rep = repair_beliefs(ebm, soft, fixed, steps=120, lr=0.3, parsimony_weight=0.0).beliefs

    print("\nRUNTIME (same net, three ways to enforce constraints):")
    print(f"  {'method':<26}{'cell acc':>10}{'grid exact':>12}{'validity':>10}")
    for name, p in [("1. ML-only", ml), ("2. ML + exact solver", solv), ("3. ML + energy repair", rep)]:
        print(f"  {name:<26}{cell_acc(p, yte):>10.3f}{grid_exact(p, yte):>12.3f}{validity(p):>10.3f}")

    # ---- training-time comparison: scarce labels, plain vs + logic loss ---- #
    # grid-exact is ~0 for everything in this noise regime (needs all 16 cells
    # right at once), so we report per-cell accuracy — the discriminative metric.
    print("\nTRAINING-TIME (scarce labels; feed-forward per-cell accuracy):")
    print(f"  {'# labels':<10}{'plain':>10}{'+ logic loss':>14}")
    label_counts = [20, 50, 100, 300]
    plain_acc, logic_acc = [], []
    plain_grid, logic_grid = [], []
    for nl in label_counts:
        pnets = [train_net(kb, engine, ebm, squares, nl, NOISE, s, 500, logic_w=0.0) for s in (0, 1, 2)]
        lnets = [train_net(kb, engine, ebm, squares, nl, NOISE, s, 500, logic_w=1.0) for s in (0, 1, 2)]
        pa = [cell_acc(m(xte).detach(), yte) for m in pnets]
        la = [cell_acc(m(xte).detach(), yte) for m in lnets]
        pg = [grid_exact(m(xte).detach(), yte) for m in pnets]
        lg = [grid_exact(m(xte).detach(), yte) for m in lnets]
        plain_acc.append(sum(pa) / 3); logic_acc.append(sum(la) / 3)
        plain_grid.append(sum(pg) / 3); logic_grid.append(sum(lg) / 3)
        print(f"  {nl:<10}{plain_acc[-1]:>10.3f}{logic_acc[-1]:>14.3f}")

    # ---- figure ---- #
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(12, 4.6))
    methods = ["ML-only", "ML + exact\nsolver", "ML + energy\nrepair"]
    accs = [grid_exact(ml, yte), grid_exact(solv, yte), grid_exact(rep, yte)]
    vals = [validity(ml), validity(solv), validity(rep)]
    x = range(len(methods)); w = 0.38
    axa.bar([i - w / 2 for i in x], accs, w, color="#0072B2", label="grid-exact accuracy")
    axa.bar([i + w / 2 for i in x], vals, w, color="#2bb673", label="valid Latin square")
    axa.set_xticks(list(x)); axa.set_xticklabels(methods, fontsize=9)
    axa.set_ylim(0, 1.08); axa.legend(fontsize=8); axa.set_title("(a) Runtime: enforce constraints on a fixed net")
    for i, (a, v) in enumerate(zip(accs, vals)):
        axa.text(i - w / 2, a + 0.01, f"{a:.2f}", ha="center", fontsize=8)
        axa.text(i + w / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    axb.plot(label_counts, plain_acc, "o-", color="#999999", lw=2, label="plain (labels only)")
    axb.plot(label_counts, logic_acc, "s-", color="#CC79A7", lw=2, label="+ logic loss (unlabelled)")
    axb.set_xscale("log"); axb.set_xlabel("# labelled grids"); axb.set_ylabel("per-cell accuracy")
    axb.set_title("(b) Training-time: logic loss with scarce labels"); axb.legend(fontsize=8); axb.grid(alpha=0.25)
    for s in ("top", "right"):
        axa.spines[s].set_visible(False); axb.spines[s].set_visible(False)
    # Title reflects the actual finding: the solver dominates at runtime; whether
    # the logic loss helps at training time is read off the numbers, not assumed.
    lift = sum(logic_acc[i] - plain_acc[i] for i in range(len(label_counts))) / len(label_counts)
    tail = ("the logic loss adds a small training-time lift"
            if lift > 0.005 else "the logic loss gives no training-time lift here")
    fig.suptitle(f"Latin squares: an exact solver dominates at runtime; {tail}", y=1.02, fontsize=12)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig_latin.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out}")

    with open(os.path.join(RES_DIR, "latin.json"), "w", encoding="utf-8") as fh:
        json.dump({"noise": NOISE, "runtime": {m: {"cell": cell_acc(p, yte), "grid": grid_exact(p, yte),
                                                    "validity": validity(p)}
                                               for m, p in [("ml", ml), ("solver", solv), ("energy", rep)]},
                   "label_counts": label_counts,
                   "plain_cell": plain_acc, "logic_cell": logic_acc,
                   "plain_grid": plain_grid, "logic_grid": logic_grid}, fh, indent=2)
    print("=" * 78)


if __name__ == "__main__":
    main()
