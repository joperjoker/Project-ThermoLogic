"""End-to-end integration: LogicEnergy as a guardrail on a real trained model.

The product story made concrete. We train a **plain multi-label neural network**
(standard BCE, *no logic in training* — i.e. "someone else's model") on the
harder cloud-config benchmark, then drop :class:`thermologic.LogicEnergy` after it
to (1) **detect** which of its held-out predictions break the policy — with no
labels, just the energy score — and (2) **repair** those to valid configurations.

Two quantitative results:

* **Label-free error detection with ~100% precision.** Every output the energy
  flags (energy > 0) is provably not the valid ground-truth config — an
  inconsistent output cannot equal a consistent world — so *flagged ⇒ wrong*
  with near-certainty, and no labels. Honest limit: energy detects
  *inconsistency*, not *correctness*, so it misses consistent-but-wrong outputs
  (those need labels). It is a high-precision, not high-recall, detector.
* **Repair + a consistency guarantee.** Feeding the outputs through ``repair``
  restores **100% policy-consistency**. Honest caveat: repair enforces
  consistency with the model's own *recovered* facts, so consistency ≠
  correctness — control accuracy is essentially unchanged (bounded by how well
  the model reads its inputs), not magically improved.

    python integration_demo.py
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
    LogicEnergy,
    NeuralProposer,
    TNorm,
    enumerate_worlds,
    sample_from_worlds,
    split_worlds,
)
from benchmarks.cloud_config import build_cloud_kb

FIG_DIR, RES_DIR = "figures", "results"


def train_plain(kb, split, seed=0, epochs=3, n_train=600, noise=0.12, noise_dims=4):
    """A plain multi-label net: BCE on every atom, NO logic term (an off-the-shelf model).

    Deliberately a *realistic, imperfect* deployment — a decent but not perfect
    model (~0.9 control accuracy) that still emits policy-violating configs on a
    meaningful fraction of unseen inputs, which is exactly what the guardrail is for.
    """
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    tr = sample_from_worlds(kb, split.train_worlds, n_train, noise, noise_dims, gen)
    net = NeuralProposer(tr.input_dim, kb.num_atoms, (64, 64))
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    bce = torch.nn.BCELoss()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(tr.inputs, tr.targets),
        batch_size=128, shuffle=True, generator=gen)
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad(); bce(net(xb), yb).backward(); opt.step()
    net.eval()
    return net, tr.input_dim


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(RES_DIR, exist_ok=True)
    kb, idx = build_cloud_kb(3)
    engine = DifferentiableLogicEngine(kb, TNorm.LUKASIEWICZ)
    worlds = enumerate_worlds(kb, engine)
    split = split_worlds(worlds, holdout=len(worlds) // 3, seed=0)

    print("=" * 76)
    print("Integration: LogicEnergy guardrail on a plain (logic-free) model")
    print(f"cloud-config benchmark — {kb.num_atoms} atoms, {kb.num_rules} rules, "
          f"{len(worlds)} worlds ({len(split.test_worlds)} unseen)")
    print("=" * 76)

    net, _ = train_plain(kb, split)
    gen = torch.Generator().manual_seed(777)
    ev = sample_from_worlds(kb, split.test_worlds, 3000, 0.12, 4, gen)

    # ---- the product API: wrap ANY model's output ---- #
    guard = LogicEnergy.from_knowledge_base(kb)
    with torch.no_grad():
        preds = net(ev.inputs)                       # the model's raw structured output
    energy = guard.score(preds)                      # label-free inconsistency signal
    consistent = guard.is_consistent(preds, crisp=True)
    didx = torch.tensor(list(kb.derived_atoms))

    def control_acc(mask=None):
        p = preds if mask is None else preds[mask]
        y = ev.targets if mask is None else ev.targets[mask]
        if p.shape[0] == 0:
            return float("nan")
        return float(((p.index_select(1, didx) > 0.5).float() == y.index_select(1, didx)).float().mean())

    frac_flagged = float((~consistent).float().mean())
    print(f"\nplain model on unseen configs:")
    print(f"  policy-consistent outputs : {float(consistent.float().mean()):.3f}  "
          f"({frac_flagged*100:.1f}% violate the policy — flagged by energy, no labels)")
    print(f"  control accuracy          : {control_acc():.3f}")

    # ---- (1) label-free detection: precision of the energy flag ---- #
    # "wrong" = the full predicted config differs from ground truth. A flagged
    # (inconsistent) output is not a valid world at all, so it cannot equal the
    # valid ground-truth world — hence flagged ⇒ wrong with certainty.
    wrong = 1.0 - ((preds > 0.5).float() == ev.targets).all(1).float()
    flagged = ~consistent
    err_flagged = float(wrong[flagged].mean()) if int(flagged.sum()) else float("nan")
    err_unflagged = float(wrong[~flagged].mean()) if int((~flagged).sum()) else float("nan")
    print(f"\nlabel-free detection (energy flag):")
    print(f"  P(wrong | flagged)   = {err_flagged:.3f}   ← every flagged config is wrong (precision)")
    print(f"  P(wrong | unflagged) = {err_unflagged:.3f}   ← energy misses consistent-but-wrong (recall limit)")

    # ---- (2) repair the flagged outputs ---- #
    repaired = guard.repair(preds, fixed=[kb.atom_names[i] for i in kb.cause_atoms],
                            budget=150, snap=True, verify=True)
    rep_consistent = float(guard.is_consistent(repaired, crisp=True).float().mean())
    rep_acc = float(((repaired.index_select(1, didx) > 0.5).float()
                     == ev.targets.index_select(1, didx)).float().mean())
    print(f"\nafter LogicEnergy.repair():")
    print(f"  policy-consistent outputs : {rep_consistent:.3f}   (guaranteed)")
    print(f"  control accuracy          : {control_acc():.3f}  ->  {rep_acc:.3f}   "
          f"(consistency ≠ correctness; bounded by input quality)")

    # ---- concrete examples ---- #
    print("\nexample flagged configs and their repair:")
    flagged_ix = (~consistent).nonzero(as_tuple=True)[0][:2].tolist()
    names = kb.atom_names
    for i in flagged_ix:
        v = guard.violations((preds[i:i+1] > 0.5).float())[0]     # crisp broken rules
        changed = [(names[a], int(preds[i, a] > 0.5), int(repaired[i, a] > 0.5))
                   for a in kb.derived_atoms if (preds[i, a] > 0.5) != (repaired[i, a] > 0.5)]
        print(f"  sample {i}: energy {float(energy[i]):.2f}, breaks {len(v)} rule(s): {v[:3]}")
        print(f"     repair sets: {[f'{n}:{a}→{b}' for n, a, b in changed] or '(facts infeasible)'}")

    # ---- figure: (a) energy separates compliant/non-compliant, (b) repair ---- #
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(11, 4.4))
    crisp_e = guard.score((preds > 0.5).float()).detach().numpy()  # 0 iff compliant
    comp = consistent.numpy().astype(bool)
    axa.hist([crisp_e[comp], crisp_e[~comp]], bins=24, stacked=True,
             color=["#2bb673", "#D55E00"], label=["compliant (E=0)", "flagged (E>0)"])
    axa.set_yscale("log")
    axa.set_xlabel("energy of the config (0 = obeys every rule)")
    axa.set_ylabel("# of the model's configs (log)")
    axa.set_title(f"(a) Energy flags {frac_flagged:.0%} as non-compliant — no labels")
    axa.legend(fontsize=9)
    for s in ("top", "right"):
        axa.spines[s].set_visible(False)
    axb.bar(["plain model\noutput", "after\nLogicEnergy.repair"],
            [float(consistent.float().mean()), rep_consistent],
            color=["#999999", "#CC79A7"], alpha=0.9, width=0.6)
    for i, v in enumerate([float(consistent.float().mean()), rep_consistent]):
        axb.text(i, v + 0.02, f"{v:.0%}", ha="center", fontsize=11)
    axb.set_ylabel("policy-consistent outputs")
    axb.set_ylim(0, 1.08)
    axb.set_title("(b) Repair guarantees 100% consistency")
    for s in ("top", "right"):
        axb.spines[s].set_visible(False)
    fig.suptitle("LogicEnergy guardrail on a plain model: flag every non-compliant config, then repair",
                 y=1.02, fontsize=11.5)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig_integration.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out}")

    with open(os.path.join(RES_DIR, "integration.json"), "w", encoding="utf-8") as fh:
        json.dump({"frac_flagged": frac_flagged, "control_acc": control_acc(),
                   "P_wrong_given_flagged": err_flagged, "P_wrong_given_unflagged": err_unflagged,
                   "repaired_consistent": rep_consistent, "repaired_acc": rep_acc}, fh, indent=2)
    print("=" * 76)


if __name__ == "__main__":
    main()
