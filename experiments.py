"""Reproducible experiment battery for the Project ThermoLogic paper.

Runs every experiment reported in ``PAPER.md`` on the 11-atom expanded knowledge
base with a genuine world-level train/test holdout, and writes:

* ``results/metrics.json`` — all machine-readable metrics (with per-seed spread);
* ``figures/*.png``        — every labelled chart in the paper.

Regenerate everything (fixed seeds, deterministic on CPU) with::

    python experiments.py

Experiments
-----------
1. Truth-propagation wave  — depth vs. test-time-compute (the centerpiece).
2. Generalization + repair — seen / unseen-feedforward / unseen-repaired.
3. Repair curve            — accuracy & energy vs. compute budget.
4. Baselines               — supervised NN (no logic) vs. ThermoLogic (+repair).
5. Robustness              — input-noise sweep and holdout-size sweep.
6. t-norm deep-dive        — Product / Lukasiewicz / Godel on the new KB.
7. Energy landscape        — energy distribution over consistent vs. inconsistent.
"""

from __future__ import annotations

import itertools
import json
import os
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from thermologic.data import enumerate_worlds, sample_from_worlds, split_worlds
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
from train import build_expanded_kb

FIG_DIR, RES_DIR = "figures", "results"

# Colour-blind-safe (Okabe-Ito) palette, used consistently across figures.
C_ENERGY = "#D55E00"
C_ACC = "#0072B2"
C_SAT = "#009E73"
C_REPAIR = "#CC79A7"
C_BASE = "#999999"
C_WARM = "#E69F00"
TNORM_COLOR = {"lukasiewicz": C_ACC, "product": C_ENERGY, "godel": C_SAT}


def _style(ax: "plt.Axes") -> None:
    ax.grid(True, alpha=0.25, linewidth=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# --------------------------------------------------------------------------- #
# Shared training routine
# --------------------------------------------------------------------------- #
@dataclass
class RunResult:
    """Metrics from one trained configuration."""

    seen_acc: float
    unseen_ff_acc: float
    unseen_ff_energy: float
    unseen_repaired_acc: float
    unseen_repaired_energy: float
    unseen_cause_acc: float
    per_atom_repaired: Dict[str, float] = field(default_factory=dict)


def _derived_acc(beliefs: torch.Tensor, targets: torch.Tensor, kb: KnowledgeBase) -> float:
    idx = torch.tensor(kb.derived_atoms)
    pred = (beliefs.index_select(1, idx) > 0.5).float()
    return float((pred == targets.index_select(1, idx)).float().mean())


def _cause_acc(beliefs: torch.Tensor, targets: torch.Tensor, kb: KnowledgeBase) -> float:
    idx = torch.tensor(kb.cause_atoms)
    pred = (beliefs.index_select(1, idx) > 0.5).float()
    return float((pred == targets.index_select(1, idx)).float().mean())


def train_config(
    kb: KnowledgeBase,
    tnorm: TNorm = TNorm.LUKASIEWICZ,
    beta: float = 4.0,
    epochs: int = 30,
    seed: int = 0,
    noise_std: float = 0.12,
    noise_dims: int = 3,
    holdout: int = 6,
    hidden: Sequence[int] = (96, 96),
    supervise_all: bool = False,
    use_energy: bool = True,
    repair_budget: int = 120,
    lr: float = 1e-2,
    train_samples: int = 4096,
    eval_samples: int = 1024,
) -> Tuple[RunResult, EnergyBasedModel, "torch.Tensor", "torch.Tensor"]:
    """Train one configuration; return metrics, model, and the unseen eval split.

    ``supervise_all=True`` gives the fully-supervised (no-logic) baseline: BCE on
    every atom, no energy term. Otherwise supervision is on cause atoms only and
    the derived atoms are constrained by the energy (the ThermoLogic model).
    """
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    engine = DifferentiableLogicEngine(kb, tnorm)
    worlds = enumerate_worlds(kb, engine)
    split = split_worlds(worlds, holdout=min(holdout, len(worlds) - 1), seed=0)

    tr = sample_from_worlds(kb, split.train_worlds, train_samples, noise_std, noise_dims, gen)
    seen = sample_from_worlds(kb, split.train_worlds, eval_samples, noise_std, noise_dims, gen)
    unseen = sample_from_worlds(kb, split.test_worlds, eval_samples, noise_std, noise_dims, gen)

    model = EnergyBasedModel(NeuralProposer(tr.input_dim, kb.num_atoms, tuple(hidden)), engine, beta)

    if supervise_all:
        criterion = torch.nn.BCELoss()
    else:
        w_rule = 1.0 if use_energy else 0.0
        criterion = ThermoLogicLoss(kb, w_supervised=1.0, w_rule=w_rule, w_parsimony=0.1)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(tr.inputs, tr.targets),
        batch_size=64, shuffle=True, generator=gen,
    )
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            if supervise_all:
                loss = criterion(model(xb).beliefs, yb)
            else:
                loss = criterion(model(xb), yb).total
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        seen_b = model(seen.inputs).beliefs
        unseen_out = model(unseen.inputs)
        unseen_b = unseen_out.beliefs
    # test-time repair on unseen worlds (hold recovered causes fixed)
    fixed_mask = torch.zeros(kb.num_atoms)
    fixed_mask[list(kb.cause_atoms)] = 1.0
    rep = repair_beliefs(model, unseen_b, fixed_mask, steps=repair_budget, lr=0.3,
                         parsimony_weight=0.0, optimizer="adam")
    rep_b = rep.beliefs
    with torch.no_grad():
        rep_energy = float(model.energy_from_satisfaction(engine.satisfaction(rep_b)).mean())

    d_idx = kb.derived_atoms
    per_atom = {
        kb.atom_names[a]: float(((rep_b[:, a] > 0.5).float() == unseen.targets[:, a]).float().mean())
        for a in d_idx
    }
    res = RunResult(
        seen_acc=_derived_acc(seen_b, seen.targets, kb),
        unseen_ff_acc=_derived_acc(unseen_b, unseen.targets, kb),
        unseen_ff_energy=float(unseen_out.energy.mean()),
        unseen_repaired_acc=_derived_acc(rep_b, unseen.targets, kb),
        unseen_repaired_energy=rep_energy,
        unseen_cause_acc=_cause_acc(unseen_b, unseen.targets, kb),
        per_atom_repaired=per_atom,
    )
    return res, model, unseen.inputs, unseen.targets


def _mean_std(xs: Sequence[float]) -> Tuple[float, float]:
    return statistics.mean(xs), (statistics.pstdev(xs) if len(xs) > 1 else 0.0)


# --------------------------------------------------------------------------- #
# Experiment 1 — truth-propagation wave (depth vs. test-time compute)
# --------------------------------------------------------------------------- #
def _chain_kb(k: int) -> KnowledgeBase:
    names = tuple(f"A{i}" for i in range(k + 1))
    rules = tuple(ImplicationRule((Literal(i - 1),), Literal(i), f"A{i-1}->A{i}") for i in range(1, k + 1))
    return KnowledgeBase(names, rules, cause_atoms=(0,), derived_atoms=tuple(range(1, k + 1)))


def _descend(kb: KnowledgeBase, tnorm=TNorm.LUKASIEWICZ, beta=4.0, lr=10.0, steps=160, init=0.02):
    eng = DifferentiableLogicEngine(kb, tnorm)
    ebm = EnergyBasedModel(NeuralProposer(1, kb.num_atoms), eng, beta)
    A = kb.num_atoms
    b0 = torch.full((1, A), init)
    b0[0, 0] = 1.0
    fixed = torch.zeros(A)
    fixed[0] = 1.0
    free = 1 - fixed
    logit = torch.logit(b0.clamp(1e-4, 1 - 1e-4)).clone().requires_grad_(True)
    opt = torch.optim.SGD([logit], lr=lr)
    hist = []
    for _ in range(steps):
        opt.zero_grad()
        st = b0 * fixed + torch.sigmoid(logit) * free
        ebm.energy_from_satisfaction(eng.satisfaction(st)).mean().backward()
        opt.step()
        with torch.no_grad():
            hist.append((b0 * fixed + torch.sigmoid(logit) * free)[0, 1:].clone())
    return torch.stack(hist)


def _activations(B: torch.Tensor) -> List[int]:
    return [int(next((i for i in range(B.shape[0]) if B[i, d] > 0.5), B.shape[0])) for d in range(B.shape[1])]


def _linfit(a: Sequence[float]) -> Tuple[float, float]:
    n = len(a)
    xs = list(range(1, n + 1))
    sx, sy, sxx, sxy = sum(xs), sum(a), sum(x * x for x in xs), sum(x * v for x, v in zip(xs, a))
    m = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    return m, (sy - m * sx) / n


def exp_depth_wave(metrics: Dict) -> None:
    K = 20
    B = _descend(_chain_kb(K))
    act = _activations(B)
    slope, icpt = _linfit(act)
    tn_acts = {}
    for tn in (TNorm.LUKASIEWICZ, TNorm.PRODUCT, TNorm.GODEL):
        tn_acts[tn.value] = _activations(_descend(_chain_kb(K), tnorm=tn))
    metrics["depth_wave"] = {
        "chain_length": K,
        "lukasiewicz_activation_steps": act,
        "lukasiewicz_steps_per_hop": slope,
        "steps_per_hop_by_tnorm": {k: _linfit(v)[0] for k, v in tn_acts.items()},
        "monotonic": all(act[i] <= act[i + 1] for i in range(len(act) - 1)),
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.3))
    im = ax1.imshow(B.T.numpy(), aspect="auto", origin="lower", cmap="viridis",
                    extent=[0, B.shape[0], 1, K], vmin=0, vmax=1)
    ax1.plot(act, range(1, K + 1), color="red", lw=2, label="activation front (p=0.5)")
    ax1.set_xlim(0, act[-1] + 8)
    ax1.set_xlabel("energy-descent step  (test-time compute)")
    ax1.set_ylabel("atom depth in proof chain")
    ax1.set_title(f"(a) Truth propagates as a wave — one hop at a time (k={K})")
    ax1.legend(loc="lower right", fontsize=8)
    fig.colorbar(im, ax=ax1).set_label("belief  p(atom = true)")
    for name, a in tn_acts.items():
        m, _ = _linfit(a)
        ax2.plot(range(1, K + 1), a, "o-", color=TNORM_COLOR[name], ms=4, label=f"{name} ({m:.1f} steps/hop)")
    ax2.set_xlabel("atom depth in proof chain  (reasoning hops)")
    ax2.set_ylabel("activation step  (test-time compute)")
    ax2.set_title("(b) Inference cost grows linearly with reasoning depth")
    ax2.legend(fontsize=8, title="t-norm")
    _style(ax2)
    fig.suptitle("Figure 1 — Logical energy as test-time compute", y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_depth_wave.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Experiment 2 — generalization + repair (multi-seed)
# --------------------------------------------------------------------------- #
def exp_generalization(kb: KnowledgeBase, metrics: Dict, seeds=(0, 1, 2, 3, 4)) -> None:
    seen, uff, urep, uff_e, urep_e, cause = [], [], [], [], [], []
    for s in seeds:
        r, *_ = train_config(kb, seed=s)
        seen.append(r.seen_acc); uff.append(r.unseen_ff_acc); urep.append(r.unseen_repaired_acc)
        uff_e.append(r.unseen_ff_energy); urep_e.append(r.unseen_repaired_energy); cause.append(r.unseen_cause_acc)
    labels = ["seen worlds", "unseen\n(feed-forward)", "unseen\n(+ repair)"]
    means = [_mean_std(seen)[0], _mean_std(uff)[0], _mean_std(urep)[0]]
    stds = [_mean_std(seen)[1], _mean_std(uff)[1], _mean_std(urep)[1]]
    metrics["generalization"] = {
        "seeds": list(seeds),
        "seen_acc": _mean_std(seen), "unseen_ff_acc": _mean_std(uff), "unseen_repaired_acc": _mean_std(urep),
        "unseen_ff_energy": _mean_std(uff_e), "unseen_repaired_energy": _mean_std(urep_e),
        "unseen_cause_acc": _mean_std(cause),
    }

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(11, 4.3), gridspec_kw={"width_ratios": [1.1, 1]})
    colors = [C_SAT, C_ENERGY, C_REPAIR]
    bars = axa.bar(labels, means, yerr=stds, capsize=5, color=colors, alpha=0.9)
    for b, m, sd in zip(bars, means, stds):
        axa.text(b.get_x() + b.get_width() / 2, m + sd + 0.01, f"{m:.3f}", ha="center", fontsize=9)
    axa.axhline(0.5, color=C_BASE, ls="--", lw=1, label="chance")
    axa.set_ylabel("derived-atom accuracy"); axa.set_ylim(0, 1.05)
    axa.set_title("(a) Generalization to unseen worlds"); axa.legend(fontsize=8); _style(axa)
    e_labels = ["unseen\n(feed-forward)", "unseen\n(+ repair)"]
    e_means = [_mean_std(uff_e)[0], _mean_std(urep_e)[0]]
    e_std = [_mean_std(uff_e)[1], _mean_std(urep_e)[1]]
    ebars = axb.bar(e_labels, e_means, yerr=e_std, capsize=5, color=[C_ENERGY, C_REPAIR], alpha=0.9)
    for b, m in zip(ebars, e_means):
        axb.text(b.get_x() + b.get_width() / 2, m, f"{m:.3f}", ha="center", va="bottom", fontsize=9)
    axb.set_ylabel("mean logical energy"); axb.set_title("(b) Repair drives energy to zero"); _style(axb)
    fig.suptitle(f"Figure 2 — Amortized inference leaves unseen states inconsistent; repair fixes them "
                 f"(mean ± std over {len(seeds)} seeds)", y=1.02, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_generalization.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Experiment 3 — repair curve (accuracy & energy vs. compute budget)
# --------------------------------------------------------------------------- #
def exp_repair_curve(kb: KnowledgeBase, metrics: Dict, seed=0) -> None:
    _, model, X, Y = train_config(kb, seed=seed, repair_budget=1)
    engine = model.engine
    with torch.no_grad():
        b0 = model(X).beliefs.clone()
    fixed = torch.zeros(kb.num_atoms); fixed[list(kb.cause_atoms)] = 1.0
    budgets = [0, 5, 10, 20, 30, 50, 75, 100, 150, 200]
    accs, ens = [], []
    d_idx = torch.tensor(kb.derived_atoms)
    for bud in budgets:
        if bud == 0:
            b = b0
        else:
            b = repair_beliefs(model, b0, fixed, steps=bud, lr=0.3, parsimony_weight=0.0).beliefs
        with torch.no_grad():
            accs.append(float(((b.index_select(1, d_idx) > 0.5).float() == Y.index_select(1, d_idx)).float().mean()))
            ens.append(float(model.energy_from_satisfaction(engine.satisfaction(b)).mean()))
    metrics["repair_curve"] = {"budgets": budgets, "accuracy": accs, "energy": ens}

    fig, ax1 = plt.subplots(figsize=(7.2, 4.3))
    ax2 = ax1.twinx()
    l1 = ax1.plot(budgets, ens, "o-", color=C_ENERGY, lw=2, label="mean energy")
    ax1.set_yscale("symlog", linthresh=1e-3)
    ax1.set_xlabel("repair budget  (energy-descent steps = test-time compute)")
    ax1.set_ylabel("mean logical energy", color=C_ENERGY); ax1.tick_params(axis="y", labelcolor=C_ENERGY)
    l2 = ax2.plot(budgets, accs, "s-", color=C_ACC, lw=2, label="derived-atom accuracy")
    ax2.set_ylabel("derived-atom accuracy", color=C_ACC); ax2.tick_params(axis="y", labelcolor=C_ACC)
    ax2.set_ylim(min(accs) - 0.02, 1.005)
    ax1.legend(l1 + l2, [x.get_label() for x in l1 + l2], loc="center right", fontsize=9)
    _style(ax1)
    ax1.set_title("Figure 3 — Spend more test-time compute → lower energy, higher accuracy")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_repair_curve.png"), dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Experiment 4 — baselines
# --------------------------------------------------------------------------- #
def exp_baselines(kb: KnowledgeBase, metrics: Dict, seeds=(0, 1, 2)) -> None:
    sup, sup_e, ff, rep, rep_e = [], [], [], [], []
    for s in seeds:
        rb, mb, Xb, Yb = train_config(kb, seed=s, supervise_all=True)   # supervised NN, no logic
        sup.append(rb.unseen_ff_acc)
        with torch.no_grad():                                          # its logical energy
            sup_e.append(float(mb.energy_from_satisfaction(mb.engine.satisfaction(mb(Xb).beliefs)).mean()))
        rt, *_ = train_config(kb, seed=s)                              # ThermoLogic
        ff.append(rt.unseen_ff_acc); rep.append(rt.unseen_repaired_acc); rep_e.append(rt.unseen_repaired_energy)
    metrics["baselines"] = {
        "supervised_nn": _mean_std(sup), "supervised_nn_energy": _mean_std(sup_e),
        "thermologic_ff": _mean_std(ff),
        "thermologic_repair": _mean_std(rep), "thermologic_repair_energy": _mean_std(rep_e),
    }
    names = ["Supervised NN\n(no logic)", "ThermoLogic\n(feed-forward)", "ThermoLogic\n(+ repair)"]
    accs = [_mean_std(sup), _mean_std(ff), _mean_std(rep)]
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(11, 4.3))
    bars = axa.bar(names, [m for m, _ in accs], yerr=[s for _, s in accs], capsize=5,
                   color=[C_BASE, C_ACC, C_REPAIR], alpha=0.9)
    for b, (m, sd) in zip(bars, accs):
        axa.text(b.get_x() + b.get_width() / 2, m + sd + 0.008, f"{m:.3f}", ha="center", fontsize=9)
    axa.axhline(0.5, color=C_ENERGY, ls="--", lw=1, label="chance")
    axa.set_ylabel("derived-atom accuracy (unseen)"); axa.set_ylim(0, 1.05)
    axa.set_title("(a) Accuracy: comparable"); axa.legend(fontsize=8); _style(axa)
    # (b) the differentiator — logical consistency (energy). Lower = consistent.
    e_names = ["Supervised NN\n(no logic)", "ThermoLogic\n(+ repair)"]
    e_vals = [_mean_std(sup_e), _mean_std(rep_e)]
    ebars = axb.bar(e_names, [m for m, _ in e_vals], yerr=[s for _, s in e_vals], capsize=5,
                    color=[C_BASE, C_REPAIR], alpha=0.9)
    for b, (m, _) in zip(ebars, e_vals):
        axb.text(b.get_x() + b.get_width() / 2, m, f"{m:.3f}", ha="center", va="bottom", fontsize=9)
    axb.set_ylabel("mean logical energy (unseen)  — lower = consistent")
    axb.set_title("(b) Consistency: only ThermoLogic guarantees it"); _style(axb)
    fig.suptitle("Figure 4 — A plain net matches on accuracy, but only the energy gives a "
                 "consistency guarantee (mean ± std, 3 seeds)", y=1.02, fontsize=10.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_baselines.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Experiment 5 — robustness (noise sweep + holdout sweep)
# --------------------------------------------------------------------------- #
def exp_robustness(kb: KnowledgeBase, metrics: Dict, seeds=(0, 1, 2)) -> None:
    noises = [0.05, 0.12, 0.2, 0.3, 0.4]
    n_ff, n_rep = [], []
    for ns in noises:
        ff = [train_config(kb, seed=s, noise_std=ns)[0] for s in seeds]
        n_ff.append(_mean_std([r.unseen_ff_acc for r in ff]))
        n_rep.append(_mean_std([r.unseen_repaired_acc for r in ff]))
    holdouts = [3, 6, 9, 12]
    h_ff, h_rep = [], []
    for ho in holdouts:
        ff = [train_config(kb, seed=s, holdout=ho)[0] for s in seeds]
        h_ff.append(_mean_std([r.unseen_ff_acc for r in ff]))
        h_rep.append(_mean_std([r.unseen_repaired_acc for r in ff]))
    metrics["robustness"] = {
        "noise": {"levels": noises, "ff": n_ff, "repair": n_rep},
        "holdout": {"sizes": holdouts, "ff": h_ff, "repair": h_rep},
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))
    ax1.errorbar(noises, [m for m, _ in n_ff], yerr=[s for _, s in n_ff], fmt="o-", color=C_ACC,
                 capsize=4, label="feed-forward")
    ax1.errorbar(noises, [m for m, _ in n_rep], yerr=[s for _, s in n_rep], fmt="s-", color=C_REPAIR,
                 capsize=4, label="+ repair")
    ax1.set_xlabel("input noise  σ"); ax1.set_ylabel("unseen derived accuracy")
    ax1.set_title("(a) Robustness to input noise"); ax1.legend(fontsize=8); _style(ax1); ax1.set_ylim(0.5, 1.02)
    ax2.errorbar(holdouts, [m for m, _ in h_ff], yerr=[s for _, s in h_ff], fmt="o-", color=C_ACC,
                 capsize=4, label="feed-forward")
    ax2.errorbar(holdouts, [m for m, _ in h_rep], yerr=[s for _, s in h_rep], fmt="s-", color=C_REPAIR,
                 capsize=4, label="+ repair")
    ax2.set_xlabel("# worlds held out for testing (of 18)"); ax2.set_ylabel("unseen derived accuracy")
    ax2.set_title("(b) Robustness to holdout size"); ax2.legend(fontsize=8); _style(ax2); ax2.set_ylim(0.5, 1.02)
    fig.suptitle("Figure 5 — Robustness sweeps (mean ± std over 3 seeds)", y=1.02, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_robustness.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Experiment 6 — t-norm deep-dive
# --------------------------------------------------------------------------- #
def exp_tnorm(kb: KnowledgeBase, metrics: Dict, seeds=(0, 1, 2)) -> None:
    out = {}
    for tn in (TNorm.LUKASIEWICZ, TNorm.PRODUCT, TNorm.GODEL):
        ff = [train_config(kb, tnorm=tn, seed=s)[0] for s in seeds]
        out[tn.value] = {
            "unseen_ff_acc": _mean_std([r.unseen_ff_acc for r in ff]),
            "unseen_repaired_acc": _mean_std([r.unseen_repaired_acc for r in ff]),
            "unseen_ff_energy": _mean_std([r.unseen_ff_energy for r in ff]),
        }
    metrics["tnorm"] = out
    names = list(out.keys())
    x = range(len(names))
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    w = 0.35
    ax.bar([i - w / 2 for i in x], [out[n]["unseen_ff_acc"][0] for n in names], w,
           yerr=[out[n]["unseen_ff_acc"][1] for n in names], capsize=4, color=C_ACC, label="feed-forward")
    ax.bar([i + w / 2 for i in x], [out[n]["unseen_repaired_acc"][0] for n in names], w,
           yerr=[out[n]["unseen_repaired_acc"][1] for n in names], capsize=4, color=C_REPAIR, label="+ repair")
    ax.set_xticks(list(x)); ax.set_xticklabels([n.capitalize() for n in names])
    ax.set_ylabel("unseen derived accuracy"); ax.set_ylim(0, 1.05)
    ax.set_title("Figure 6 — t-norm comparison on the 11-atom KB (mean ± std, 3 seeds)")
    ax.legend(fontsize=8); _style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_tnorm.png"), dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Experiment 7 — energy landscape over the world space
# --------------------------------------------------------------------------- #
def exp_energy_landscape(kb: KnowledgeBase, metrics: Dict, beta=4.0) -> None:
    engine = DifferentiableLogicEngine(kb, TNorm.LUKASIEWICZ)
    ebm = EnergyBasedModel(NeuralProposer(1, kb.num_atoms), engine, beta)
    valid_e, invalid_e = [], []
    for bits in itertools.product([0, 1], repeat=kb.num_atoms):
        probs = torch.tensor([[float(b) for b in bits]])
        e = float(ebm.energy_from_satisfaction(engine.satisfaction(probs)))
        (valid_e if e < 1e-6 else invalid_e).append(e)
    metrics["energy_landscape"] = {
        "num_worlds": 2 ** kb.num_atoms, "num_consistent": len(valid_e),
        "num_inconsistent": len(invalid_e),
        "max_inconsistent_energy": max(invalid_e), "min_inconsistent_energy": min(invalid_e),
    }
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    ax.hist(invalid_e, bins=30, color=C_ENERGY, alpha=0.85, label=f"inconsistent ({len(invalid_e)})")
    ax.axvline(0, color=C_SAT, lw=3, label=f"consistent, E=0 ({len(valid_e)})")
    ax.set_xlabel("logical energy  E"); ax.set_ylabel("number of Boolean worlds")
    ax.set_title(f"Figure 7 — Energy partitions all $2^{{{kb.num_atoms}}}$={2**kb.num_atoms} worlds "
                 f"into consistent (E=0) vs. contradictions")
    ax.legend(fontsize=9); _style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_energy_landscape.png"), dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(RES_DIR, exist_ok=True)
    kb = build_expanded_kb()
    metrics: Dict = {"kb": {"num_atoms": kb.num_atoms, "num_rules": kb.num_rules}}

    print("[1/7] depth-propagation wave ...");      exp_depth_wave(metrics)
    print("[2/7] generalization + repair ...");     exp_generalization(kb, metrics)
    print("[3/7] repair curve ...");                exp_repair_curve(kb, metrics)
    print("[4/7] baselines ...");                   exp_baselines(kb, metrics)
    print("[5/7] robustness sweeps ...");           exp_robustness(kb, metrics)
    print("[6/7] t-norm deep-dive ...");            exp_tnorm(kb, metrics)
    print("[7/7] energy landscape ...");            exp_energy_landscape(kb, metrics)

    with open(os.path.join(RES_DIR, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    g = metrics["generalization"]
    print("\nDone. figures/*.png + results/metrics.json written.")
    print(f"  seen acc            : {g['seen_acc'][0]:.3f} ± {g['seen_acc'][1]:.3f}")
    print(f"  unseen feed-forward : {g['unseen_ff_acc'][0]:.3f} ± {g['unseen_ff_acc'][1]:.3f}  "
          f"(energy {g['unseen_ff_energy'][0]:.3f})")
    print(f"  unseen + repair     : {g['unseen_repaired_acc'][0]:.3f} ± {g['unseen_repaired_acc'][1]:.3f}  "
          f"(energy {g['unseen_repaired_energy'][0]:.4f})")
    print(f"  depth steps/hop     : {metrics['depth_wave']['steps_per_hop_by_tnorm']}")


if __name__ == "__main__":
    main()
