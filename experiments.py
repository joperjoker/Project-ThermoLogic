"""Reproducible experiment harness for the Project ThermoLogic paper.

Runs the full battery of experiments reported in ``PAPER.md`` and writes:

* ``results/metrics.json`` — machine-readable metrics for every experiment;
* ``figures/*.png``        — all labelled charts embedded in the paper.

Everything is derived from the same modules used by ``train.py`` (no external
data, fixed seeds), so the paper's numbers regenerate exactly with:

    python experiments.py
"""

from __future__ import annotations

import itertools
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # headless backend; write PNGs without a display
import matplotlib.pyplot as plt
import torch

from logic_engine import DifferentiableLogicEngine, TNorm
from model import EnergyBasedModel, NeuralProposer, ThermoLogicLoss
from train import Dataset, build_default_kb, evaluate, generate_dataset

FIG_DIR = "figures"
RES_DIR = "results"

# A colour-blind-safe qualitative palette (Okabe-Ito subset), used consistently.
C_ENERGY = "#D55E00"   # vermillion
C_ACC = "#0072B2"      # blue
C_SAT = "#009E73"      # green
C_SUP = "#CC79A7"      # purple
C_PAR = "#E69F00"      # orange
C_NEUTRAL = "#555555"


@dataclass
class History:
    """Per-epoch training history for one run."""

    epoch: List[int] = field(default_factory=list)
    loss: List[float] = field(default_factory=list)
    supervised: List[float] = field(default_factory=list)
    rule_energy: List[float] = field(default_factory=list)
    parsimony: List[float] = field(default_factory=list)
    eval_energy: List[float] = field(default_factory=list)
    eval_sat: List[float] = field(default_factory=list)
    eval_valid: List[float] = field(default_factory=list)
    eval_acc: List[float] = field(default_factory=list)


def run_training(
    tnorm: TNorm = TNorm.LUKASIEWICZ,
    beta: float = 4.0,
    w_rule: float = 1.0,
    w_parsimony: float = 0.1,
    epochs: int = 40,
    seed: int = 42,
    train_samples: int = 2048,
    eval_samples: int = 512,
    batch_size: int = 64,
    lr: float = 1e-2,
) -> Tuple[History, "torch.nn.Module", Dataset]:
    """Train one configuration and return its history, model, and eval split."""
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)

    kb = build_default_kb()
    engine = DifferentiableLogicEngine(kb, tnorm=tnorm)
    train_data = generate_dataset(kb, train_samples, 0.25, 3, generator)
    eval_data = generate_dataset(kb, eval_samples, 0.25, 3, generator)

    proposer = NeuralProposer(input_dim=train_data.input_dim, num_atoms=kb.num_atoms)
    model = EnergyBasedModel(proposer, engine, beta=beta)
    criterion = ThermoLogicLoss(kb, 1.0, w_rule, w_parsimony)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_data.inputs, train_data.targets),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )

    hist = History()
    for epoch in range(1, epochs + 1):
        model.train()
        agg = {"loss": 0.0, "sup": 0.0, "energy": 0.0, "par": 0.0}
        seen = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.total.backward()
            optimizer.step()
            bs = xb.shape[0]
            seen += bs
            agg["loss"] += float(loss.total.detach()) * bs
            agg["sup"] += float(loss.supervised) * bs
            agg["energy"] += float(loss.rule_energy) * bs
            agg["par"] += float(loss.parsimony) * bs

        m = evaluate(model, eval_data, kb)
        hist.epoch.append(epoch)
        hist.loss.append(agg["loss"] / seen)
        hist.supervised.append(agg["sup"] / seen)
        hist.rule_energy.append(agg["energy"] / seen)
        hist.parsimony.append(agg["par"] / seen)
        hist.eval_energy.append(m.energy)
        hist.eval_sat.append(m.satisfaction)
        hist.eval_valid.append(m.valid_fraction)
        hist.eval_acc.append(m.derived_accuracy)

    return hist, model, eval_data


# --------------------------------------------------------------------------- #
# Energy landscape over the full Boolean world space
# --------------------------------------------------------------------------- #
def energy_landscape(beta: float = 4.0) -> List[Dict[str, object]]:
    """Energy of every one of the 2^A Boolean worlds under the rule base.

    A world is *valid* iff it equals the minimal model implied by its own cause
    atoms (i.e. it is logically self-consistent). Returns a record per world with
    its bits, validity, mean satisfaction, and energy.
    """
    kb = build_default_kb()
    engine = DifferentiableLogicEngine(kb, tnorm=TNorm.LUKASIEWICZ)
    dummy = NeuralProposer(1, kb.num_atoms)
    ebm = EnergyBasedModel(dummy, engine, beta=beta)

    records: List[Dict[str, object]] = []
    for bits in itertools.product([0, 1], repeat=kb.num_atoms):
        probs = torch.tensor([[float(b) for b in bits]])
        sat = engine.satisfaction(probs)
        energy = float(ebm.energy_from_satisfaction(sat))
        # A crisp world is logically *valid* iff it satisfies every rule (all
        # per-rule satisfactions equal 1), which is exactly the E=0 condition.
        # This includes the negative constraint Rain -> not Sprinkler, so the
        # inconsistent world [1,1,1,1,1] is correctly flagged as a contradiction.
        valid = bool((sat > 0.999).all())
        records.append(
            {
                "bits": list(bits),
                "valid": bool(valid),
                "satisfaction": float(sat.mean()),
                "energy": energy,
            }
        )
    return records


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _style(ax: "plt.Axes") -> None:
    ax.grid(True, alpha=0.25, linewidth=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def fig_training_curves(hist: History) -> None:
    """Twin-axis: mean energy (down) and derived accuracy (up) vs epoch."""
    fig, ax1 = plt.subplots(figsize=(7.2, 4.2))
    ax2 = ax1.twinx()

    l1 = ax1.plot(hist.epoch, hist.eval_energy, color=C_ENERGY, lw=2.2,
                  marker="o", ms=3, label="Mean energy $E$")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Mean energy $E$  (log scale)", color=C_ENERGY)
    ax1.set_yscale("log")
    ax1.tick_params(axis="y", labelcolor=C_ENERGY)

    l2 = ax2.plot(hist.epoch, hist.eval_acc, color=C_ACC, lw=2.2,
                  marker="s", ms=3, label="Derived-atom accuracy")
    l3 = ax2.plot(hist.epoch, hist.eval_sat, color=C_SAT, lw=1.6, ls="--",
                  label="Rule satisfaction")
    ax2.set_ylabel("Accuracy / Satisfaction", color=C_ACC)
    ax2.set_ylim(0.4, 1.02)
    ax2.tick_params(axis="y", labelcolor=C_ACC)

    ax1.axhline(1e-3, color=C_NEUTRAL, lw=0.8, ls=":", alpha=0.6)
    lines = l1 + l2 + l3
    ax1.legend(lines, [ln.get_label() for ln in lines], loc="center right",
               frameon=True, fontsize=9)
    _style(ax1)
    ax1.set_title("Figure 2 — Energy minimization vs. logical accuracy (Łukasiewicz)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_training_curves.png"), dpi=140)
    plt.close(fig)


def fig_loss_components(hist: History) -> None:
    """Stacked view of the three loss terms across training."""
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(hist.epoch, hist.supervised, color=C_SUP, lw=2, marker="o", ms=3,
            label="Supervised BCE (causes)")
    ax.plot(hist.epoch, hist.rule_energy, color=C_ENERGY, lw=2, marker="s", ms=3,
            label="Rule energy  $\\overline{E}$")
    ax.plot(hist.epoch, hist.parsimony, color=C_PAR, lw=2, marker="^", ms=3,
            label="Parsimony (derived mass)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss component value")
    ax.set_yscale("log")
    ax.legend(frameon=True, fontsize=9)
    _style(ax)
    ax.set_title("Figure 3 — Decomposition of the composite objective")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_loss_components.png"), dpi=140)
    plt.close(fig)


def fig_tnorm_comparison(hists: Dict[str, History]) -> None:
    """Derived-accuracy learning curves for the three t-norm families."""
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(10.4, 4.2))
    colours = {"lukasiewicz": C_ACC, "product": C_ENERGY, "godel": C_SAT}
    markers = {"lukasiewicz": "o", "product": "s", "godel": "^"}
    for name, h in hists.items():
        axa.plot(h.epoch, h.eval_acc, color=colours[name], lw=2,
                 marker=markers[name], ms=3, label=name.capitalize())
        axb.plot(h.epoch, h.eval_energy, color=colours[name], lw=2,
                 marker=markers[name], ms=3, label=name.capitalize())
    axa.set_xlabel("Epoch"); axa.set_ylabel("Derived-atom accuracy")
    axa.set_ylim(0.4, 1.02); axa.legend(frameon=True, fontsize=9)
    axa.set_title("(a) Derived-atom accuracy")
    axb.set_xlabel("Epoch"); axb.set_ylabel("Mean energy $E$ (log)")
    axb.set_yscale("log"); axb.legend(frameon=True, fontsize=9)
    axb.set_title("(b) Mean energy")
    for ax in (axa, axb):
        _style(ax)
    fig.suptitle("Figure 4 — t-norm family comparison", y=1.00)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_tnorm_comparison.png"), dpi=140)
    plt.close(fig)


def fig_ablation_wrule(results: List[Tuple[float, float, float]]) -> None:
    """Bar chart: final derived accuracy as the logic weight w_rule varies."""
    ws = [f"{w:g}" for w, _, _ in results]
    accs = [a for _, a, _ in results]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bars = ax.bar(ws, accs, color=C_ACC, alpha=0.9, width=0.6)
    bars[0].set_color(C_NEUTRAL)  # highlight the w_rule=0 (no-logic) baseline
    ax.axhline(0.5, color=C_ENERGY, ls="--", lw=1.2, label="chance (0.5)")
    for b, a in zip(bars, accs):
        ax.text(b.get_x() + b.get_width() / 2, a + 0.01, f"{a:.3f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Logic energy weight  $w_{\\mathrm{rule}}$")
    ax.set_ylabel("Final derived-atom accuracy")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=True, fontsize=9)
    _style(ax)
    ax.set_title("Figure 5 — Ablation: the logical energy term drives derivation")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_ablation_wrule.png"), dpi=140)
    plt.close(fig)


def fig_energy_landscape(records: List[Dict[str, object]]) -> None:
    """Sorted energy of all 32 Boolean worlds, coloured by logical validity."""
    order = sorted(range(len(records)), key=lambda i: records[i]["energy"])
    energies = [records[i]["energy"] for i in order]
    valid = [records[i]["valid"] for i in order]
    colours = [C_SAT if v else C_ENERGY for v in valid]

    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    ax.bar(range(len(energies)), energies, color=colours, width=0.9)
    ax.set_yscale("symlog", linthresh=1e-3)
    # Zero-energy (valid) worlds are invisible on a log axis; mark them
    # explicitly at the baseline so the ground state is legible.
    valid_x = [i for i, v in enumerate(valid) if v]
    ax.scatter(valid_x, [5e-4] * len(valid_x), marker="^", s=55, color=C_SAT,
               zorder=5, edgecolor="white", linewidth=0.5)
    ax.annotate(f"{len(valid_x)} valid worlds at $E=0$ (ground state)",
                xy=(valid_x[-1], 5e-4), xytext=(len(energies) * 0.16, 0.15),
                fontsize=9, color=C_SAT,
                arrowprops=dict(arrowstyle="->", color=C_SAT, lw=1.2))
    ax.set_xlabel("Boolean world (sorted by energy)")
    ax.set_ylabel("Energy $E$  (symlog)")
    ax.set_ylim(0, max(energies) * 2)
    handles = [
        plt.Line2D([0], [0], marker="^", color="w", markerfacecolor=C_SAT, markersize=9),
        plt.Rectangle((0, 0), 1, 1, color=C_ENERGY),
    ]
    ax.legend(handles, ["logically valid ($E=0$)", "contradiction ($E>0$)"],
              frameon=True, fontsize=9, loc="upper left")
    _style(ax)
    n_valid = sum(valid)
    ax.set_title(f"Figure 6 — Energy over all $2^5$=32 worlds "
                 f"({n_valid} valid at $E\\approx0$, {32 - n_valid} penalized)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_energy_landscape.png"), dpi=140)
    plt.close(fig)


def fig_beta_sweep(rows: List[Tuple[float, float, float]]) -> None:
    """How the inverse-temperature beta shapes the valid/contradiction gap."""
    betas = [b for b, _, _ in rows]
    e_valid = [ev for _, ev, _ in rows]
    e_bad = [eb for _, _, eb in rows]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(betas, e_bad, color=C_ENERGY, lw=2, marker="s", ms=5,
            label="contradiction energy")
    ax.plot(betas, [max(v, 1e-9) for v in e_valid], color=C_SAT, lw=2,
            marker="o", ms=5, label="valid-world energy")
    ax.set_yscale("log")
    ax.set_xlabel(r"Inverse temperature  $\beta$")
    ax.set_ylabel("Energy $E$ (log)")
    ax.legend(frameon=True, fontsize=9)
    _style(ax)
    ax.set_title(r"Figure 7 — Inverse temperature $\beta$ sharpens the energy gap")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_beta_sweep.png"), dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _energy_probe(model: "torch.nn.Module") -> Tuple[float, float]:
    """Return (valid_world_energy, contradiction_energy) for a trained EBM."""
    engine = model.engine
    valid = torch.tensor([[1.0, 1.0, 1.0, 0.0, 1.0]])
    bad = torch.tensor([[1.0, 0.0, 0.0, 1.0, 0.0]])
    ev = float(model.energy_from_satisfaction(engine.satisfaction(valid)))
    eb = float(model.energy_from_satisfaction(engine.satisfaction(bad)))
    return ev, eb


def main() -> None:
    """Run every experiment, write metrics.json and all figures."""
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(RES_DIR, exist_ok=True)
    metrics: Dict[str, object] = {}

    print("[1/5] main run (Łukasiewicz) ...")
    main_hist, main_model, _ = run_training(tnorm=TNorm.LUKASIEWICZ)
    fig_training_curves(main_hist)
    fig_loss_components(main_hist)
    metrics["main"] = {
        "final_energy": main_hist.eval_energy[-1],
        "final_accuracy": main_hist.eval_acc[-1],
        "final_satisfaction": main_hist.eval_sat[-1],
        "init_accuracy": main_hist.eval_acc[0],
        "history": main_hist.__dict__,
    }

    print("[2/5] t-norm comparison ...")
    tnorm_hists: Dict[str, History] = {"lukasiewicz": main_hist}
    tnorm_final: Dict[str, Dict[str, float]] = {}
    for tn in (TNorm.PRODUCT, TNorm.GODEL):
        h, m, _ = run_training(tnorm=tn)
        tnorm_hists[tn.value] = h
    for name, h in tnorm_hists.items():
        tnorm_final[name] = {
            "final_accuracy": h.eval_acc[-1],
            "final_energy": h.eval_energy[-1],
            "final_satisfaction": h.eval_sat[-1],
        }
    fig_tnorm_comparison(tnorm_hists)
    metrics["tnorm"] = tnorm_final

    print("[3/5] ablation over w_rule ...")
    ablation: List[Tuple[float, float, float]] = []
    for w in (0.0, 0.25, 0.5, 1.0, 2.0):
        h, _, _ = run_training(w_rule=w)
        ablation.append((w, h.eval_acc[-1], h.eval_energy[-1]))
    fig_ablation_wrule(ablation)
    metrics["ablation_wrule"] = [
        {"w_rule": w, "final_accuracy": a, "final_energy": e} for w, a, e in ablation
    ]

    print("[4/5] beta sweep ...")
    beta_rows: List[Tuple[float, float, float]] = []
    for beta in (1.0, 2.0, 4.0, 8.0):
        recs = energy_landscape(beta=beta)
        e_valid = max(r["energy"] for r in recs if r["valid"])
        e_bad = max(r["energy"] for r in recs if not r["valid"])
        beta_rows.append((beta, e_valid, e_bad))
    fig_beta_sweep(beta_rows)
    metrics["beta_sweep"] = [
        {"beta": b, "max_valid_energy": ev, "max_contradiction_energy": eb}
        for b, ev, eb in beta_rows
    ]

    print("[5/5] energy landscape (β=4) ...")
    records = energy_landscape(beta=4.0)
    fig_energy_landscape(records)
    ev, eb = _energy_probe(main_model)
    metrics["landscape"] = {
        "num_worlds": len(records),
        "num_valid": sum(1 for r in records if r["valid"]),
        "trained_valid_energy": ev,
        "trained_contradiction_energy": eb,
        "records": records,
    }

    with open(os.path.join(RES_DIR, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    print("\nDone. Wrote figures/*.png and results/metrics.json")
    print(f"  main derived-acc : {main_hist.eval_acc[0]:.3f} -> {main_hist.eval_acc[-1]:.3f}")
    print(f"  main energy      : {main_hist.eval_energy[0]:.4f} -> {main_hist.eval_energy[-1]:.4f}")
    print(f"  ablation w=0 acc : {ablation[0][1]:.3f}  (no-logic baseline)")


if __name__ == "__main__":
    main()
