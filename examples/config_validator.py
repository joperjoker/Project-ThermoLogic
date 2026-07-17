"""Worked use case — a SaaS plan configurator guarded by ThermoLogic.

A realistic application of :class:`thermologic.LogicEnergy` to **structured
configuration validation**, a broad B2B problem where outputs (a set of feature
toggles) must satisfy hard interdependencies (business rules). A model or a user
proposes a configuration; the logic-energy layer

1. **scores** how much it violates the rules (a label-free validity signal),
2. **names** the specific rules broken, and
3. **repairs** it to the nearest valid configuration — keeping the plan tier the
   customer chose fixed, and changing as few feature flags as possible.

Run::

    python examples/config_validator.py

Roadmap (same mechanism, other domains): regulated tabular decisions
(finance/insurance/healthcare constraints) and data-integrity repair of noisy
records against schema rules. This example uses config validation because its
rules are cleanly propositional.
"""

from __future__ import annotations

from typing import Dict, List

import torch

from thermologic import LogicEnergy, implies

# --- the product's feature atoms ------------------------------------------- #
ATOMS: List[str] = [
    "plan_free", "plan_pro", "plan_enterprise",   # tier (mutually exclusive)
    "sso", "audit_logs", "priority_support",
    "custom_domain", "seats_gt_5", "data_residency_eu",
    "white_label", "uptime_sla_99_99",
]
PLAN_ATOMS = ["plan_free", "plan_pro", "plan_enterprise"]

# --- business rules (propositional constraints) ---------------------------- #
RULES = [
    # plan tiers are mutually exclusive
    implies(["plan_free"], "plan_pro", negate_consequent=True, name="free⇒¬pro"),
    implies(["plan_free"], "plan_enterprise", negate_consequent=True, name="free⇒¬enterprise"),
    implies(["plan_pro"], "plan_enterprise", negate_consequent=True, name="pro⇒¬enterprise"),
    # enterprise-only features
    implies(["sso"], "plan_enterprise", name="sso⇒enterprise"),
    implies(["white_label"], "plan_enterprise", name="white_label⇒enterprise"),
    implies(["data_residency_eu"], "plan_enterprise", name="eu_residency⇒enterprise"),
    implies(["uptime_sla_99_99"], "plan_enterprise", name="sla⇒enterprise"),
    # not available on the free tier
    implies(["audit_logs"], "plan_free", negate_consequent=True, name="audit_logs⇒¬free"),
    implies(["priority_support"], "plan_free", negate_consequent=True, name="support⇒¬free"),
    implies(["custom_domain"], "plan_free", negate_consequent=True, name="custom_domain⇒¬free"),
    implies(["seats_gt_5"], "plan_free", negate_consequent=True, name="seats>5⇒¬free"),
]


def config_to_tensor(config: Dict[str, bool]) -> torch.Tensor:
    """Turn a ``{atom: bool}`` config into a ``(1, num_atoms)`` belief tensor."""
    unknown = set(config) - set(ATOMS)
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    return torch.tensor([[1.0 if config.get(a, False) else 0.0 for a in ATOMS]])


def tensor_to_flags(t: torch.Tensor, threshold: float = 0.5) -> Dict[str, bool]:
    """Turn a belief tensor back into a ``{atom: bool}`` config."""
    return {a: bool(t[0, i] > threshold) for i, a in enumerate(ATOMS)}


def enabled(flags: Dict[str, bool]) -> List[str]:
    """List the enabled feature atoms (for readable printing)."""
    return [a for a, on in flags.items() if on]


def demo() -> None:
    guard = LogicEnergy(RULES, atom_names=ATOMS, beta=4.0)

    # Three customer requests, each a plan tier + a wishlist of features.
    requests = [
        ("Valid Enterprise request", {
            "plan_enterprise": True, "sso": True, "audit_logs": True,
            "data_residency_eu": True, "seats_gt_5": True}),
        ("Pro customer asking for SSO + audit logs", {
            "plan_pro": True, "sso": True, "audit_logs": True, "seats_gt_5": True}),
        ("Free customer overreaching", {
            "plan_free": True, "custom_domain": True, "priority_support": True,
            "seats_gt_5": True, "white_label": True}),
    ]

    print("=" * 78)
    print("ThermoLogic config validator — SaaS plan configurator")
    print(f"{len(ATOMS)} feature atoms, {len(RULES)} business rules")
    print("=" * 78)

    for title, req in requests:
        out = config_to_tensor(req)
        score = float(guard.score(out))
        violations = guard.violations(out)[0]

        print(f"\n### {title}")
        print(f"  requested   : {enabled(req)}")
        print(f"  validity score (energy): {score:8.4f}   "
              f"{'✅ VALID' if score < 1e-3 else '❌ INVALID'}")

        if score >= 1e-3:
            print(f"  rules broken : {violations}")
            # Repair, holding the customer's chosen plan tier fixed. snap=True
            # returns a discrete config; verify=True guarantees it is valid
            # (auto-raising the compute budget until every rule holds).
            chosen_plan = [p for p in PLAN_ATOMS if req.get(p)]
            repaired = guard.repair(out, fixed=chosen_plan, budget=60,
                                    snap=True, verify=True)
            fixed_flags = tensor_to_flags(repaired)
            removed = [a for a in enabled(req) if not fixed_flags.get(a)]
            added = [a for a in enabled(fixed_flags) if a not in enabled(req)]
            print(f"  repaired to : {enabled(fixed_flags)}")
            print(f"     removed   : {removed or '—'}")
            print(f"     added     : {added or '—'}")
            print(f"     new score : {float(guard.score(repaired)):8.4f}   "
                  f"(rules broken: {guard.violations(repaired)[0] or 'none'})")

    print("\n" + "=" * 78)
    print("Takeaway: the same energy layer that SCORES invalid configs also REPAIRS")
    print("them to the nearest valid one — no labelled training data required.")
    print("=" * 78)


if __name__ == "__main__":
    demo()
