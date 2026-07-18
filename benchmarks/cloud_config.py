"""A harder, first-order-*grounded* benchmark: cloud security configuration.

Moves beyond the 11-atom toy to a realistic domain with genuine external
relevance — the kind of policy checking cloud-posture tools (AWS Config, OPA,
CSPM) actually perform. Rules are written as **first-order templates** with a
variable over resources and **grounded** over `n_res` resources into a
propositional knowledge base (the standard way neuro-symbolic systems handle
first-order logic). With 3 resources this yields ~20 atoms, ~19 rules, and
**432 distinct valid worlds** — a real held-out generalization test.

Facts (observed) describe what a resource *is*; controls (derived) are what the
policy *forces*:

    template rules (∀ resource r):
      public(r)              → ¬has_pii(r)      # no PII in public buckets
      has_pii(r)             → encrypted(r)     # PII must be encrypted at rest
      has_pii(r)             → logging(r)       # PII access must be logged
      prod(r)                → backup(r)        # prod must be backed up
      prod(r)                → logging(r)       # prod must be logged
      compliance_required    → encrypted(r)     # compliance ⇒ encrypt everything
    global:
      compliance_required    → mfa_enabled

Minimal model (ground truth) for the controls:
    encrypted(r) = has_pii(r) ∨ compliance_required
    logging(r)   = has_pii(r) ∨ prod(r)
    backup(r)    = prod(r)
    mfa_enabled  = compliance_required
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from thermologic import ImplicationRule, KnowledgeBase, Literal

FACT_TEMPLATES = ["public", "has_pii", "prod"]      # observed per-resource facts
CONTROL_TEMPLATES = ["encrypted", "logging", "backup"]  # derived per-resource controls
GLOBAL_FACTS = ["compliance_required"]
GLOBAL_CONTROLS = ["mfa_enabled"]


def build_cloud_kb(n_res: int = 3) -> Tuple[KnowledgeBase, Dict[str, int]]:
    """Ground the first-order policy templates over ``n_res`` resources.

    Returns the propositional :class:`KnowledgeBase` and a name→index map.
    """
    names: List[str] = []
    idx: Dict[str, int] = {}

    def add(name: str) -> None:
        idx[name] = len(names)
        names.append(name)

    for r in range(n_res):
        for t in FACT_TEMPLATES:
            add(f"{t}[{r}]")
        for t in CONTROL_TEMPLATES:
            add(f"{t}[{r}]")
    for g in GLOBAL_FACTS + GLOBAL_CONTROLS:
        add(g)

    def L(name: str, neg: bool = False) -> Literal:
        return Literal(idx[name], neg)

    rules: List[ImplicationRule] = []
    for r in range(n_res):
        rules += [
            ImplicationRule((L(f"public[{r}]"),), L(f"has_pii[{r}]", True), f"public→¬pii[{r}]"),
            ImplicationRule((L(f"has_pii[{r}]"),), L(f"encrypted[{r}]"), f"pii→enc[{r}]"),
            ImplicationRule((L(f"has_pii[{r}]"),), L(f"logging[{r}]"), f"pii→log[{r}]"),
            ImplicationRule((L(f"prod[{r}]"),), L(f"backup[{r}]"), f"prod→backup[{r}]"),
            ImplicationRule((L(f"prod[{r}]"),), L(f"logging[{r}]"), f"prod→log[{r}]"),
            ImplicationRule((L("compliance_required"),), L(f"encrypted[{r}]"), f"comp→enc[{r}]"),
        ]
    rules.append(ImplicationRule((L("compliance_required"),), L("mfa_enabled"), "comp→mfa"))

    cause_atoms = tuple(idx[f"{t}[{r}]"] for r in range(n_res) for t in FACT_TEMPLATES) + \
        tuple(idx[g] for g in GLOBAL_FACTS)
    derived_atoms = tuple(idx[f"{t}[{r}]"] for r in range(n_res) for t in CONTROL_TEMPLATES) + \
        tuple(idx[g] for g in GLOBAL_CONTROLS)

    kb = KnowledgeBase(
        atom_names=tuple(names), rules=tuple(rules),
        cause_atoms=cause_atoms, derived_atoms=derived_atoms,
    )
    return kb, idx


def train_on_worlds(kb, engine, train_worlds, seed=0, epochs=40, noise=0.10,
                    noise_dims=4, n_train=6000, lr=1e-2, hidden=(128, 128), w_rule=1.0):
    """Train a Neural Proposer + EBM on samples drawn from ``train_worlds``."""
    import torch
    from thermologic import EnergyBasedModel, NeuralProposer, ThermoLogicLoss, sample_from_worlds

    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    tr = sample_from_worlds(kb, train_worlds, n_train, noise, noise_dims, gen)
    model = EnergyBasedModel(NeuralProposer(tr.input_dim, kb.num_atoms, hidden), engine, beta=4.0)
    criterion = ThermoLogicLoss(kb, w_supervised=1.0, w_rule=w_rule, w_parsimony=0.1)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(tr.inputs, tr.targets),
        batch_size=128, shuffle=True, generator=gen)
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            criterion(model(xb), yb).total.backward()
            opt.step()
    model.eval()
    return model, tr.input_dim


def run(n_res: int = 3, seed: int = 0) -> None:
    """Train on the grounded cloud KB with a world-level holdout; report metrics."""
    import torch
    from thermologic import (DifferentiableLogicEngine, TNorm, enumerate_worlds,
                             repair_beliefs, sample_from_worlds, split_worlds)

    kb, _ = build_cloud_kb(n_res)
    engine = DifferentiableLogicEngine(kb, TNorm.LUKASIEWICZ)
    worlds = enumerate_worlds(kb, engine)
    holdout = len(worlds) // 3
    split = split_worlds(worlds, holdout=holdout, seed=0)
    print("=" * 74)
    print(f"Cloud-config benchmark — {kb.num_atoms} atoms, {kb.num_rules} rules, "
          f"{len(worlds)} worlds ({len(split.train_worlds)} train / {len(split.test_worlds)} UNSEEN)")
    print("=" * 74)

    model, input_dim = train_on_worlds(kb, engine, split.train_worlds, seed=seed)
    gen = torch.Generator().manual_seed(123)
    ev = sample_from_worlds(kb, split.test_worlds, 2000, 0.10, 4, gen)
    didx = torch.tensor(list(kb.derived_atoms))

    with torch.no_grad():
        ff = model(ev.inputs).beliefs
    fixed = torch.zeros(kb.num_atoms); fixed[list(kb.cause_atoms)] = 1.0
    rep = repair_beliefs(model, ff, fixed, steps=150, lr=0.3, parsimony_weight=0.0).beliefs

    def acc(b):
        return float(((b.index_select(1, didx) > 0.5).float() == ev.targets.index_select(1, didx)).float().mean())

    def consistent(b):
        return float((engine.satisfaction((b > 0.5).float(), validate=False) > 0.5).all(1).float().mean())

    print(f"  unseen control accuracy  : feed-forward {acc(ff):.3f}  ->  + repair {acc(rep):.3f}")
    print(f"  unseen rule consistency  : feed-forward {consistent(ff):.3f}  ->  + repair {consistent(rep):.3f}")
    print("=" * 74)


if __name__ == "__main__":
    from thermologic import DifferentiableLogicEngine, TNorm, enumerate_worlds

    for n in (2, 3):
        kb, _ = build_cloud_kb(n)
        eng = DifferentiableLogicEngine(kb, TNorm.LUKASIEWICZ)
        worlds = enumerate_worlds(kb, eng)
        print(f"n_res={n}: {kb.num_atoms} atoms, {kb.num_rules} rules, "
              f"{len(worlds)} distinct valid worlds "
              f"({len(kb.cause_atoms)} facts / {len(kb.derived_atoms)} controls)")
    print()
    run(n_res=3)
