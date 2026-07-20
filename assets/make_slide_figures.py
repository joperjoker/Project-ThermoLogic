"""Plain-language versions of two paper figures, for the LinkedIn carousel.

The paper keeps its precise, technical axis labels; the slides need friendlier
ones. This script regenerates just the two deck charts with plain wording:

* assets/slide_depthwave.png  — "deeper reasoning takes more thinking"
* assets/slide_baselines.png  — "accurate, but can't tell when it's wrong"

Baseline numbers are read from results/metrics.json (no retraining); the
propagation wave is recomputed directly (fast, no training).

    python assets/make_slide_figures.py
"""

from __future__ import annotations

import json
import os

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

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(HERE)
C_DEFAULT, C_ALT1, C_ALT2 = "#0072B2", "#D55E00", "#009E73"
C_PLAIN, C_TL, C_REPAIR, C_HOT = "#999999", "#0072B2", "#CC79A7", "#D55E00"
plt.rcParams.update({"font.size": 13})


# --------------------------------------------------------------------------- #
# Figure A — the truth-propagation wave (recomputed)
# --------------------------------------------------------------------------- #
def _chain_kb(k: int) -> KnowledgeBase:
    names = tuple(f"A{i}" for i in range(k + 1))
    rules = tuple(ImplicationRule((Literal(i - 1),), Literal(i), f"r{i}") for i in range(1, k + 1))
    return KnowledgeBase(names, rules, cause_atoms=(0,), derived_atoms=tuple(range(1, k + 1)))


def _descend(kb, tnorm=TNorm.LUKASIEWICZ, beta=4.0, lr=10.0, steps=160, init=0.02):
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
        ebm.energy_from_satisfaction(eng.satisfaction(st, validate=False)).mean().backward()
        opt.step()
        with torch.no_grad():
            hist.append((b0 * fixed + torch.sigmoid(logit) * free)[0, 1:].clone())
    return torch.stack(hist)


def _activations(B):
    return [int(next((i for i in range(B.shape[0]) if B[i, d] > 0.5), B.shape[0])) for d in range(B.shape[1])]


def _fit(a):
    n = len(a); xs = list(range(1, n + 1))
    sx, sy, sxx, sxy = sum(xs), sum(a), sum(x * x for x in xs), sum(x * v for x, v in zip(xs, a))
    m = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    return m, (sy - m * sx) / n


def make_depth_wave():
    K = 20
    B = _descend(_chain_kb(K))
    act = _activations(B)
    tn = {
        "Łukasiewicz (default)": (_activations(_descend(_chain_kb(K), TNorm.LUKASIEWICZ)), C_DEFAULT),
        "Gödel": (_activations(_descend(_chain_kb(K), TNorm.GODEL)), C_ALT2),
        "Product": (_activations(_descend(_chain_kb(K), TNorm.PRODUCT)), C_ALT1),
    }
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    im = ax1.imshow(B.T.numpy(), aspect="auto", origin="lower", cmap="viridis",
                    extent=[0, B.shape[0], 1, K], vmin=0, vmax=1)
    ax1.plot(act, range(1, K + 1), color="red", lw=2.4, label="truth has reached here")
    ax1.set_xlim(0, act[-1] + 8)
    ax1.set_xlabel("thinking steps  (more compute →)")
    ax1.set_ylabel("reasoning depth  (link in the chain)")
    ax1.set_title("Truth spreads one link at a time", fontsize=14)
    ax1.legend(loc="lower right", fontsize=10)
    cb = fig.colorbar(im, ax=ax1); cb.set_label("how true  (0 → 1)")
    for name, (a, c) in tn.items():
        m, _ = _fit(a)
        ax2.plot(range(1, K + 1), a, "o-", color=c, ms=4, label=f"{name} — {m:.1f} steps/link")
    ax2.set_xlabel("reasoning depth  (links in the chain)")
    ax2.set_ylabel("thinking steps to prove it")
    ax2.set_title("Deeper reasoning ⇒ proportionally more thinking", fontsize=14)
    ax2.legend(fontsize=10, title="rule setting")
    ax2.grid(alpha=0.25)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)
    fig.tight_layout()
    out = os.path.join(HERE, "slide_depthwave.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


# --------------------------------------------------------------------------- #
# Figure B — baselines (from stored metrics)
# --------------------------------------------------------------------------- #
def make_baselines():
    with open(os.path.join(ROOT, "results", "metrics.json"), encoding="utf-8") as fh:
        b = json.load(fh)["baselines"]
    sup, sup_e = b["supervised_nn"], b["supervised_nn_energy"]
    tl_ff, tl_rep, tl_rep_e = b["thermologic_ff"], b["thermologic_repair"], b["thermologic_repair_energy"]

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(12.6, 4.6))
    names = ["Plain neural net\n(no rules)", "ThermoLogic", "ThermoLogic\n+ repair"]
    accs = [sup, tl_ff, tl_rep]
    bars = axa.bar(names, [m for m, _ in accs], yerr=[s for _, s in accs], capsize=5,
                   color=[C_PLAIN, C_TL, C_REPAIR], alpha=0.92)
    for bar, (m, sd) in zip(bars, accs):
        axa.text(bar.get_x() + bar.get_width() / 2, m + sd + 0.01, f"{m:.3f}", ha="center", fontsize=12)
    axa.axhline(0.5, color=C_HOT, ls="--", lw=1, label="random guessing")
    axa.set_ylabel("accuracy on new cases  (higher = better)")
    axa.set_ylim(0, 1.06)
    axa.set_title("Just as accurate", fontsize=14)
    axa.legend(fontsize=10)

    e_names = ["Plain neural net\n(no rules)", "ThermoLogic\n+ repair"]
    e_vals = [sup_e, tl_rep_e]
    ebars = axb.bar(e_names, [m for m, _ in e_vals], yerr=[s for _, s in e_vals], capsize=5,
                    color=[C_PLAIN, C_REPAIR], alpha=0.92)
    for bar, (m, _) in zip(ebars, e_vals):
        axb.text(bar.get_x() + bar.get_width() / 2, m, f"{m:.3f}", ha="center", va="bottom", fontsize=12)
    axb.set_ylabel("rule-break score  (lower = better)")
    axb.set_title("…but only ThermoLogic actually obeys the rules", fontsize=14)
    for ax in (axa, axb):
        ax.grid(alpha=0.25, axis="y")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle("A plain net can match on accuracy — but can't tell when it's wrong",
                 fontsize=15, y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.96), w_pad=4.0)
    out = os.path.join(HERE, "slide_baselines.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


# --------------------------------------------------------------------------- #
# Figure C — novelty check: our logic vs. Semantic Loss (from stored JSON)
# --------------------------------------------------------------------------- #
def make_novelty():
    with open(os.path.join(ROOT, "results", "semisup_sl.json"), encoding="utf-8") as fh:
        d = json.load(fh)
    with open(os.path.join(ROOT, "results", "semisup_sl_weight.json"), encoding="utf-8") as fh:
        w = json.load(fh)
    lc = d["label_counts"]
    plain = d["arms"]["plain"]["mean"]
    ours = d["arms"]["energy"]["mean"]
    sl = d["arms"]["semantic"]["mean"]

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(12.6, 4.6))
    axa.plot(lc, plain, "o-", color=C_PLAIN, lw=2.4, label="labels only")
    axa.plot(lc, ours, "s-", color=C_REPAIR, lw=2.4, label="+ our logic")
    axa.plot(lc, sl, "^-", color=C_TL, lw=2.4, label="+ Semantic Loss (prior art)")
    axa.axhline(0.5, color=C_HOT, ls="--", lw=1, label="random guessing")
    axa.set_xscale("log", base=2)
    axa.set_xlabel("# labelled examples  (fewer ←)")
    axa.set_ylabel("accuracy on a hard rule (parity)")
    axa.set_ylim(0.45, 1.02)
    axa.set_title("It genuinely helps when labels are scarce", fontsize=14)
    axa.legend(fontsize=10)

    axb.plot(w["weights"], w["energy"], "s-", color=C_REPAIR, lw=2.4, label="+ our logic")
    axb.plot(w["weights"], w["semantic"], "^-", color=C_TL, lw=2.4, label="+ Semantic Loss")
    axb.set_xlabel("how hard we lean on the rule  (loss weight)")
    axb.set_ylabel("accuracy (64 labels)")
    axb.set_ylim(0.5, 0.95)
    axb.set_title("…and needs no tuning — prior art is fragile", fontsize=14)
    axb.legend(fontsize=10)
    for ax in (axa, axb):
        ax.grid(alpha=0.25)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle("A distinct mechanism from Semantic Loss — as good here, and far more robust",
                 fontsize=15, y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.96), w_pad=4.0)
    out = os.path.join(HERE, "slide_novelty.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


# --------------------------------------------------------------------------- #
# Figure D — measured scaling (from stored JSON)
# --------------------------------------------------------------------------- #
def make_scaling():
    with open(os.path.join(ROOT, "results", "scaling.json"), encoding="utf-8") as fh:
        rows = json.load(fh)
    labels = [r["problem"] for r in rows]
    valids = [r["valid"] for r in rows]
    times = [r["repair_s"] for r in rows]
    v0, t0 = valids[0], times[0]
    vr = [v / v0 for v in valids]
    tr = [t / t0 for t in times]
    xs = list(range(len(rows)))

    fig, ax = plt.subplots(figsize=(11.5, 5.0))
    ax.set_yscale("log")
    ax.plot(xs, vr, "o-", color=C_HOT, lw=3,
            label="answers a solver must sift through")
    ax.plot(xs, tr, "s--", color=C_TL, lw=3, label="our method's time")
    ax.set_ylabel("cost, relative to the smallest puzzle  (log scale)")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(0.3, 1e22)
    for x, v, r in zip(xs, valids, vr):
        ax.annotate(f"{v:,}" if v < 1e6 else f"{v:.0e}", (x, r), textcoords="offset points",
                    xytext=(0, 11), ha="center", fontsize=11, color=C_HOT)
    for x, t, r in zip(xs, times, tr):
        ax.annotate(f"{t:.0f}s" if t >= 1 else f"{t:.1f}s", (x, r), textcoords="offset points",
                    xytext=(26, -4), ha="left", fontsize=11, color=C_TL)
    ax.set_title("When there are too many valid answers to list, our method still runs",
                 fontsize=15)
    ax.legend(fontsize=11, loc="upper left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    out = os.path.join(HERE, "slide_scaling.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    make_depth_wave()
    make_baselines()
    make_novelty()
    make_scaling()
