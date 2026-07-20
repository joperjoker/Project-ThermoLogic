"""Drop-in guardrail for an LLM's *structured* (JSON) output.

The product pitch is "bolt `LogicEnergy` after any model." This example makes
that concrete for the most common real case: an LLM assistant emits a JSON
configuration, and we need it to obey hard policy rules — deterministically,
with no labels, and with an automatic fix when it doesn't.

The adapter is **model-agnostic**: it consumes a plain ``dict`` of fields, so it
works identically behind OpenAI, Anthropic, a local model, or a hand-written
form. (No live LLM is called here — this environment has no API key — so the
inputs below are *representative* assistant outputs, including the kind of
self-contradictory config an LLM plausibly produces. Swap them for
``json.loads(your_llm_response)`` and nothing else changes.)

Domain: a cloud storage/resource configuration. Boolean fields; five policy
rules. The user's **intent** fields (is it public? does it hold PII? is it an
admin role?) are held fixed during repair — the guard may only tighten the
**safety** settings (encryption, logging, MFA, backups) to satisfy policy.

    python examples/llm_json_guardrail.py
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import torch

from thermologic import LogicEnergy, implies

# ---- the schema: boolean fields, in a fixed order = the atom vector --------- #
FIELDS: Tuple[str, ...] = (
    "public_access",       # intent: resource is internet-reachable
    "stores_pii",          # intent: holds personal data
    "admin_role",          # intent: grants administrative privilege
    "encryption_at_rest",  # safety knob
    "encryption_in_transit",  # safety knob
    "logging_enabled",     # safety knob
    "mfa_required",        # safety knob
    "backups_enabled",     # safety knob
)
INTENT_FIELDS = ("public_access", "stores_pii", "admin_role")  # held fixed in repair

# ---- the policy: five hard rules over those fields -------------------------- #
RULES = [
    implies(["stores_pii"], "encryption_at_rest", name="PII must be encrypted at rest"),
    implies(["stores_pii"], "public_access", negate_consequent=True,
            name="PII must not be publicly accessible"),
    implies(["public_access"], "logging_enabled", name="public resources must be logged"),
    implies(["admin_role"], "mfa_required", name="admin roles must require MFA"),
    implies(["stores_pii"], "backups_enabled", name="PII must be backed up"),
]

GUARD = LogicEnergy(RULES, atom_names=list(FIELDS))


def to_vec(cfg: Dict[str, bool]) -> torch.Tensor:
    """JSON dict -> (1, num_atoms) belief tensor (softened off the 0/1 corners so
    energy repair has a usable gradient; the intent fields are held fixed anyway)."""
    missing = set(FIELDS) - set(cfg)
    if missing:
        raise ValueError(f"config missing fields: {sorted(missing)}")
    raw = torch.tensor([[1.0 if cfg[f] else 0.0 for f in FIELDS]])
    return raw.clamp(0.1, 0.9)


def to_json(vec: torch.Tensor) -> Dict[str, bool]:
    """(num_atoms,) crisp tensor -> JSON dict of booleans."""
    flat = vec.view(-1)
    return {f: bool(flat[i] > 0.5) for i, f in enumerate(FIELDS)}


def guard_config(cfg: Dict[str, bool]) -> Dict[str, object]:
    """Score, explain, and repair one LLM-produced config. Model-agnostic.

    Repair may only touch the *safety* settings; the user's intent is held fixed.
    That draws a useful line: a config that is inconsistent purely because two
    *intent* fields contradict each other (e.g. a public bucket that stores PII)
    cannot be auto-repaired without changing what was asked for — the guard flags
    it as an **intent conflict** for a human, rather than silently "fixing" it.
    """
    vec = to_vec(cfg)
    energy = float(GUARD.score(vec))
    consistent = bool(GUARD.is_consistent(vec, crisp=True).all())
    broken = GUARD.violations(vec)[0]
    # Discrete guarantee: repair only the safety knobs, hold intent fixed. The
    # result is either a *verified* valid config, or a *proof* that the request
    # itself is contradictory (with the conflicting rule named).
    res = GUARD.satisfiability(vec, fixed=list(INTENT_FIELDS))[0]
    repaired_json = to_json(torch.tensor([[1.0 if v else 0.0 for v in res.state]]))
    changed = {k: repaired_json[k] for k in FIELDS if repaired_json[k] != cfg[k]}
    conflicts = [GUARD.kb.rules[j].name for j in (res.core or [])]
    return {"energy": energy, "consistent": consistent, "violations": broken,
            "status": res.status, "repaired": repaired_json, "changed": changed,
            "repairable": res.is_valid, "intent_conflicts": conflicts}


# ---- representative assistant outputs (swap for json.loads(llm_response)) ---- #
LLM_OUTPUTS: List[Tuple[str, Dict[str, bool]]] = [
    ("a clean, compliant config", {
        "public_access": False, "stores_pii": True, "admin_role": False,
        "encryption_at_rest": True, "encryption_in_transit": True,
        "logging_enabled": True, "mfa_required": False, "backups_enabled": True}),
    ("a PII bucket left public and unencrypted (classic hallucinated config)", {
        "public_access": True, "stores_pii": True, "admin_role": False,
        "encryption_at_rest": False, "encryption_in_transit": False,
        "logging_enabled": False, "mfa_required": False, "backups_enabled": False}),
    ("an admin role with no MFA", {
        "public_access": False, "stores_pii": False, "admin_role": True,
        "encryption_at_rest": False, "encryption_in_transit": True,
        "logging_enabled": True, "mfa_required": False, "backups_enabled": False}),
    ("a public resource with logging switched off", {
        "public_access": True, "stores_pii": False, "admin_role": False,
        "encryption_at_rest": True, "encryption_in_transit": True,
        "logging_enabled": False, "mfa_required": True, "backups_enabled": True}),
]


def load_generated() -> List[Tuple[str, Dict[str, bool]]]:
    """Load configs generated by Claude in-session (if the JSON file is present).

    Returns ``[(request, cfg), …]`` with the non-schema keys stripped, so real
    model output flows through the exact same guard as the hand-written examples.
    """
    path = os.path.join(os.path.dirname(__file__), "llm_generated_configs.json")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    out = []
    for o in data.get("outputs", []):
        cfg = {k: bool(v) for k, v in o.items() if k in FIELDS}
        out.append((o.get("_request", "generated config"), cfg))
    return out


def _run_batch(items: List[Tuple[str, Dict[str, bool]]]) -> Tuple[int, int, int]:
    n_flagged = n_repaired = n_conflict = 0
    for label, cfg in items:
        r = guard_config(cfg)
        print(f"\n▶ {label}")
        print(f"  energy={r['energy']:.3f}  consistent={r['consistent']}")
        if r["consistent"]:
            print("  ✓ obeys every policy rule — passed through unchanged")
            continue
        n_flagged += 1
        print("  ✗ violations:")
        for v in r["violations"]:
            print(f"      - {v}")
        if r["repairable"]:
            n_repaired += 1
            print("  ↻ auto-repaired (safety settings only; intent held fixed):")
            for k, val in r["changed"].items():
                print(f"      {k}: {cfg[k]} → {val}")
            print("  ⇒ repaired config is VERIFIED policy-consistent (hard guarantee)")
        else:
            n_conflict += 1
            print("  ⚠ intent conflict — PROVEN unsatisfiable without changing the request:")
            for v in r["intent_conflicts"]:
                print(f"      - {v}  (needs a human decision, e.g. make it private OR drop PII)")
    return n_flagged, n_repaired, n_conflict


def _summary(tag: str, n: int, counts: Tuple[int, int, int]) -> None:
    fl, rp, cf = counts
    print("\n" + "-" * 74)
    print(f"{tag}: {fl}/{n} outputs flagged | {rp} auto-repaired (verified) | "
          f"{cf} intent conflict(s) proven unsatisfiable. No labels used.")


def main() -> None:
    print("=" * 74)
    print("LogicEnergy as a guardrail on an LLM's JSON output (model-agnostic)")
    print("=" * 74)
    print("\n### Hand-written representative outputs")
    _summary("summary", len(LLM_OUTPUTS), _run_batch(LLM_OUTPUTS))

    generated = load_generated()
    if generated:
        print("\n\n### Configs generated by Claude in this session (real model JSON)")
        _summary("summary", len(generated), _run_batch(generated))
    print("=" * 74)


if __name__ == "__main__":
    main()
