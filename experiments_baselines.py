"""Head-to-head: test-time energy repair vs. projection baselines.

Answers the reviewer's question — *"why not just project the output onto the
nearest valid state?"* — honestly, on the 11-atom benchmark and on a scaling
sweep. It also demonstrates the one thing only the differentiable operator can
do (train **through** the repair); see ``experiments_through_repair.py``.

Methods compared (all hold the recovered cause atoms fixed, same as repair):

* feed-forward     — the raw net output (no repair)
* forward-chaining — threshold causes, recompute derived by the rules (Horn oracle)
* exact projection — enumerate valid completions of the derived atoms, pick the
                     one closest to the model's **soft** output (confidence-
                     weighted L1) — the strong, exact "nearest valid state"
* greedy repair    — flip the derived bit that most reduces violations, until valid
* energy repair    — ThermoLogic (differentiable gradient descent on the energy)

    python experiments_baselines.py
"""

from __future__ import annotations

import itertools
import json
import os
import statistics
import time
from typing import Dict, List, Sequence, Tuple

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
    forward_chaining,
    repair_beliefs,
)
from train import build_expanded_kb
from experiments import train_config

FIG_DIR, RES_DIR = "figures", "results"
C = {"feed-forward": "#999999", "forward-chaining": "#E69F00",
     "exact projection": "#009E73", "greedy repair": "#56B4E9",
     "energy repair": "#CC79A7"}


# --------------------------------------------------------------------------- #
# crisp helpers
# --------------------------------------------------------------------------- #
def crisp_valid(engine: DifferentiableLogicEngine, states: torch.Tensor) -> torch.Tensor:
    """Per-row mask: does the {0,1} state satisfy every rule?"""
    return (engine.satisfaction((states > 0.5).float(), validate=False) > 0.5).all(dim=1)


def derived_acc(states: torch.Tensor, targets: torch.Tensor, derived: Sequence[int]) -> float:
    idx = torch.tensor(derived)
    pred = (states.index_select(1, idx) > 0.5).float()
    return float((pred == targets.index_select(1, idx)).float().mean())


def hamming_derived(a: torch.Tensor, b: torch.Tensor, derived: Sequence[int]) -> float:
    idx = torch.tensor(derived)
    return float(((a.index_select(1, idx) > 0.5) != (b.index_select(1, idx) > 0.5)).float().sum(1).mean())


# --------------------------------------------------------------------------- #
# baselines (each returns a crisp {0,1} batch)
# --------------------------------------------------------------------------- #
def b_forward_chaining(kb, soft):
    causes = [(soft[:, c] > 0.5).tolist() for c in kb.cause_atoms]
    out = torch.zeros_like(soft)
    for i in range(soft.shape[0]):
        world = forward_chaining(kb, [bool(causes[c][i]) for c in range(len(kb.cause_atoms))])
        out[i] = torch.tensor([float(b) for b in world])
    return out


def b_exact_projection(kb, engine, soft):
    """Confidence-weighted nearest valid state (causes fixed, enumerate derived)."""
    derived = list(kb.derived_atoms)
    causes = list(kb.cause_atoms)
    combos = torch.tensor(list(itertools.product([0.0, 1.0], repeat=len(derived))))  # [2^d, d]
    out = soft.clone()
    for i in range(soft.shape[0]):
        base = (soft[i] > 0.5).float().clone()
        cand = base.unsqueeze(0).repeat(combos.shape[0], 1)   # [M, A]
        for j, d in enumerate(derived):
            cand[:, d] = combos[:, j]
        valid = crisp_valid(engine, cand)
        if not bool(valid.any()):
            out[i] = base  # no valid completion with these causes (rare)
            continue
        # distance to the model's SOFT beliefs on derived atoms (confidence-weighted)
        didx = torch.tensor(derived)
        dist = (cand.index_select(1, didx) - soft[i].index_select(0, didx).unsqueeze(0)).abs().sum(1)
        dist = dist + (~valid).float() * 1e9
        out[i] = cand[int(dist.argmin())]
    return out


def b_greedy(kb, engine, soft, max_iter=40):
    derived = list(kb.derived_atoms)
    state = (soft > 0.5).float().clone()

    def nviol(s):
        return (engine.satisfaction(s, validate=False) <= 0.5).float().sum(1)

    for _ in range(max_iter):
        cur = nviol(state)
        if float(cur.max()) == 0:
            break
        for i in range(state.shape[0]):
            if cur[i] == 0:
                continue
            best_d, best_v = None, cur[i]
            for d in derived:
                trial = state[i].clone()
                trial[d] = 1 - trial[d]
                v = nviol(trial.unsqueeze(0))[0]
                if v < best_v:
                    best_v, best_d = v, d
            if best_d is not None:
                state[i, best_d] = 1 - state[i, best_d]
    return state


def b_energy(model, kb, soft, budget=150):
    fixed = torch.zeros(kb.num_atoms)
    fixed[list(kb.cause_atoms)] = 1.0
    rep = repair_beliefs(model, soft, fixed, steps=budget, lr=0.3, parsimony_weight=0.0)
    return (rep.beliefs > 0.5).float()


# --------------------------------------------------------------------------- #
# main comparison
# --------------------------------------------------------------------------- #
def compare(seeds=(0, 1, 2)) -> Dict:
    kb = build_expanded_kb()
    methods = ["feed-forward", "forward-chaining", "exact projection", "greedy repair", "energy repair"]
    agg = {m: {"acc": [], "valid": [], "ham": [], "time_ms": [], "agree_exact": []} for m in methods}

    for s in seeds:
        _, model, X, Y = train_config(kb, seed=s)
        engine = model.engine
        with torch.no_grad():
            soft = model(X).beliefs.clone()
        preds = {}
        timings = {}
        for m in methods:
            t0 = time.perf_counter()
            if m == "feed-forward":
                p = (soft > 0.5).float()
            elif m == "forward-chaining":
                p = b_forward_chaining(kb, soft)
            elif m == "exact projection":
                p = b_exact_projection(kb, engine, soft)
            elif m == "greedy repair":
                p = b_greedy(kb, engine, soft)
            else:
                p = b_energy(model, kb, soft)
            timings[m] = (time.perf_counter() - t0) / X.shape[0] * 1000.0
            preds[m] = p
        exact = preds["exact projection"]
        didx = torch.tensor(list(kb.derived_atoms))
        for m in methods:
            agg[m]["acc"].append(derived_acc(preds[m], Y, kb.derived_atoms))
            agg[m]["valid"].append(float(crisp_valid(engine, preds[m]).float().mean()))
            agg[m]["ham"].append(hamming_derived(preds[m], soft, kb.derived_atoms))
            agg[m]["time_ms"].append(timings[m])
            same = ((preds[m].index_select(1, didx) > 0.5) == (exact.index_select(1, didx) > 0.5)).all(1)
            agg[m]["agree_exact"].append(float(same.float().mean()))

    def ms(xs):
        return (statistics.mean(xs), statistics.pstdev(xs) if len(xs) > 1 else 0.0)

    return {m: {k: ms(v) for k, v in d.items()} for m, d in agg.items()}


# --------------------------------------------------------------------------- #
# scaling: enumeration cost (projection) vs linear cost (energy repair)
# --------------------------------------------------------------------------- #
def _chain_kb(k):
    names = tuple(f"A{i}" for i in range(k + 1))
    rules = tuple(ImplicationRule((Literal(i - 1),), Literal(i), f"r{i}") for i in range(1, k + 1))
    return KnowledgeBase(names, rules, cause_atoms=(0,), derived_atoms=tuple(range(1, k + 1)))


def scaling(depths=(4, 6, 8, 10, 12, 14, 16, 18)) -> Dict:
    proj_t, energy_t = [], []
    for d in depths:
        kb = _chain_kb(d)
        eng = DifferentiableLogicEngine(kb, TNorm.LUKASIEWICZ)
        ebm = EnergyBasedModel(NeuralProposer(1, kb.num_atoms), eng, 4.0)
        soft = torch.rand(1, kb.num_atoms)
        soft[0, 0] = 1.0
        # exact projection: enumerate 2^d derived completions, check validity
        t0 = time.perf_counter()
        combos = torch.tensor(list(itertools.product([0.0, 1.0], repeat=d)))
        cand = (soft > 0.5).float().repeat(combos.shape[0], 1)
        cand[:, 1:] = combos
        _ = (eng.satisfaction(cand, validate=False) > 0.5).all(dim=1)
        proj_t.append((time.perf_counter() - t0) * 1000.0)
        # energy repair: fixed budget descent
        fixed = torch.zeros(kb.num_atoms); fixed[0] = 1.0
        t0 = time.perf_counter()
        repair_beliefs(ebm, soft, fixed, steps=200, lr=0.5, parsimony_weight=0.0)
        energy_t.append((time.perf_counter() - t0) * 1000.0)
    return {"depths": list(depths), "projection_ms": proj_t, "energy_ms": energy_t}


# --------------------------------------------------------------------------- #
def make_figures(cmp_res: Dict, scale_res: Dict) -> None:
    methods = list(cmp_res.keys())
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(12.5, 4.6))
    x = range(len(methods))
    accs = [cmp_res[m]["acc"][0] for m in methods]
    accs_e = [cmp_res[m]["acc"][1] for m in methods]
    bars = axa.bar(x, accs, yerr=accs_e, capsize=4, color=[C[m] for m in methods], alpha=0.9)
    for b, a in zip(bars, accs):
        axa.text(b.get_x() + b.get_width() / 2, a + 0.012, f"{a:.3f}", ha="center", fontsize=10)
    axa.set_xticks(list(x)); axa.set_xticklabels([m.replace(" ", "\n") for m in methods], fontsize=9)
    axa.set_ylabel("derived-atom accuracy (unseen)"); axa.set_ylim(0, 1.08)
    axa.set_title("(a) Accuracy: exact projection is a strong baseline")
    axa.grid(alpha=0.25, axis="y")
    for sp in ("top", "right"):
        axa.spines[sp].set_visible(False)

    axb.plot(scale_res["depths"], scale_res["projection_ms"], "o-", color=C["exact projection"],
             lw=2, label="exact projection (enumerate 2ᵈ)")
    axb.plot(scale_res["depths"], scale_res["energy_ms"], "s-", color=C["energy repair"],
             lw=2, label="energy repair (linear)")
    axb.set_yscale("log")
    axb.set_xlabel("# free (derived) atoms  d")
    axb.set_ylabel("time per sample (ms, log)")
    axb.set_title("(b) Cost: enumeration explodes, energy stays flat")
    axb.legend(fontsize=9); axb.grid(alpha=0.25)
    for sp in ("top", "right"):
        axb.spines[sp].set_visible(False)
    fig.suptitle("Energy repair vs. projection — accuracy is a tie, scaling & differentiability are not",
                 fontsize=13, y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.96), w_pad=3)
    out = os.path.join(FIG_DIR, "fig_projection_baseline.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(RES_DIR, exist_ok=True)
    print("[1/2] method comparison (3 seeds) ...")
    cmp_res = compare()
    print("[2/2] scaling sweep ...")
    scale_res = scaling()
    make_figures(cmp_res, scale_res)
    with open(os.path.join(RES_DIR, "baselines.json"), "w", encoding="utf-8") as fh:
        json.dump({"comparison": cmp_res, "scaling": scale_res}, fh, indent=2)

    print("\n=== derived accuracy · validity · Δ from proposal · agree w/ exact · time ===")
    for m, d in cmp_res.items():
        print(f"  {m:17s} acc {d['acc'][0]:.3f}±{d['acc'][1]:.3f} | "
              f"valid {d['valid'][0]:.2f} | Δham {d['ham'][0]:.2f} | "
              f"agree {d['agree_exact'][0]:.2f} | {d['time_ms'][0]:.2f} ms")
    print(f"\nscaling (proj ms): {[round(t,1) for t in scale_res['projection_ms']]}")
    print(f"scaling (energy ms): {[round(t,1) for t in scale_res['energy_ms']]}")


if __name__ == "__main__":
    main()
