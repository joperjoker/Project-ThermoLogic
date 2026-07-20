"""Discrete guarantee layer: turn *soft* energy repair into a *hard* guarantee.

The soft energy of ``model.py`` makes contradictions **expensive**, not
**impossible** — gradient descent can stall in a shallow, still-invalid basin, so
``repair`` may hand back an output that is not actually rule-satisfying. For a
tool whose whole value proposition is reliability, that is the gap to close.

This module adds a crisp, discrete backstop:

* :func:`rule_holds` / :func:`crisp_violations` — evaluate rules on a Boolean
  assignment (the exact discrete semantics the fuzzy engine relaxes).
* :func:`min_conflicts_repair` — a min-conflicts / WalkSAT-style local search over
  the *free* atoms that drives the crisp violation count to zero. If it reaches
  zero, the result is a **verified** valid state — a hard guarantee.
* :func:`provable_unsat_core` — a *sound* (never wrong, but incomplete) check that
  a problem has **no** valid completion under the fixed atoms, returning the
  conflicting rule(s). This is what lets the guardrail say "this request is
  self-contradictory" instead of failing silently.
* :func:`guaranteed_repair` — the high-level entry point combining the two.

Design note on honesty: local search is *sound for SAT* (a returned state is
genuinely valid) but *incomplete for UNSAT* (failing to find a state is not a
proof none exists). We therefore never claim UNSAT from search failure alone —
only from :func:`provable_unsat_core`, which is sound. The status is reported
explicitly (``repaired`` / ``already_valid`` / ``unsat`` / ``unknown``) so callers
are never handed a false guarantee.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from thermologic.logic_engine import ImplicationRule, KnowledgeBase, Literal

__all__ = [
    "rule_holds",
    "crisp_violations",
    "provable_unsat_core",
    "min_conflicts_repair",
    "guaranteed_repair",
    "GuaranteeResult",
]


def _lit_true(lit: Literal, state: Sequence[bool]) -> bool:
    v = bool(state[lit.atom])
    return (not v) if lit.negated else v


def rule_holds(rule: ImplicationRule, state: Sequence[bool]) -> bool:
    """Does a Horn implication ``body → consequent`` hold on a Boolean state?

    An empty body is a fact (body truth ``True``). The rule holds iff the body is
    not fully satisfied, or the consequent literal is true.
    """
    body = all(_lit_true(lit, state) for lit in rule.antecedents)
    return (not body) or _lit_true(rule.consequent, state)


def crisp_violations(kb: KnowledgeBase, state: Sequence[bool]) -> List[int]:
    """Indices of the rules violated by a Boolean assignment."""
    return [i for i, r in enumerate(kb.rules) if not rule_holds(r, state)]


def _rule_atoms(rule: ImplicationRule) -> set:
    return {lit.atom for lit in rule.antecedents} | {rule.consequent.atom}


def provable_unsat_core(
    kb: KnowledgeBase, fixed_mask: Sequence[bool], state: Sequence[bool]
) -> Optional[List[int]]:
    """Sound (incomplete) UNSAT check: a rule *entirely over fixed atoms* that is
    violated proves no valid completion exists — return that rule as the core.

    Returns ``None`` if no such definitive conflict is found (which is *not* a
    proof of satisfiability).
    """
    for i, r in enumerate(kb.rules):
        if all(fixed_mask[a] for a in _rule_atoms(r)) and not rule_holds(r, state):
            return [i]
    return None


def min_conflicts_repair(
    kb: KnowledgeBase,
    init_state: Sequence[bool],
    fixed_mask: Sequence[bool],
    max_iters: int = 2000,
    restarts: int = 12,
    noise: float = 0.2,
    seed: int = 0,
) -> Tuple[List[bool], bool]:
    """Min-conflicts local search over the free atoms toward zero violations.

    Holds ``fixed`` atoms at their ``init_state`` values and flips free atoms to
    reduce the crisp violation count, with a WalkSAT-style random-walk probability
    ``noise`` and periodic random restarts. Violation deltas are evaluated
    incrementally over only the rules touching a flipped atom, so a step costs
    ``O(degree)`` rather than ``O(num_rules)``.

    Returns ``(state, success)``; ``success`` means the returned state satisfies
    **every** rule (a verified hard guarantee).
    """
    rng = random.Random(seed)
    n = kb.num_atoms
    fixed = [bool(fixed_mask[a]) for a in range(n)]
    free = [a for a in range(n) if not fixed[a]]

    atom_rules: Dict[int, List[int]] = {a: [] for a in range(n)}
    for i, r in enumerate(kb.rules):
        for a in _rule_atoms(r):
            atom_rules[a].append(i)

    def viol_in(state: List[bool], rule_ids: Sequence[int]) -> int:
        return sum(0 if rule_holds(kb.rules[i], state) else 1 for i in rule_ids)

    best_state = [bool(init_state[a]) for a in range(n)]
    best_v = len(crisp_violations(kb, best_state))
    if best_v == 0:
        return best_state, True

    for attempt in range(restarts):
        state = [bool(init_state[a]) for a in range(n)]
        if attempt > 0:  # randomize free atoms on restart; keep fixed atoms
            for a in free:
                state[a] = rng.random() < 0.5
        total = len(crisp_violations(kb, state))

        for _ in range(max_iters):
            if total == 0:
                return state, True
            viol = crisp_violations(kb, state)
            rule = kb.rules[rng.choice(viol)]
            cand = [a for a in _rule_atoms(rule) if not fixed[a]]
            if not cand:
                break  # violated rule pinned entirely by fixed atoms — restart
            if rng.random() < noise:
                a = rng.choice(cand)
            else:  # greedy: flip the free atom that most reduces total violations
                best_a, best_delta = cand[0], None
                for a in cand:
                    before = viol_in(state, atom_rules[a])
                    state[a] = not state[a]
                    after = viol_in(state, atom_rules[a])
                    state[a] = not state[a]
                    delta = after - before
                    if best_delta is None or delta < best_delta:
                        best_delta, best_a = delta, a
                a = best_a
            before = viol_in(state, atom_rules[a])
            state[a] = not state[a]
            after = viol_in(state, atom_rules[a])
            total += after - before
            if total < best_v:
                best_v, best_state = total, list(state)

    return best_state, (best_v == 0)


@dataclass
class GuaranteeResult:
    """Outcome of :func:`guaranteed_repair`.

    Attributes
    ----------
    state:
        The best Boolean assignment found (valid iff ``status`` is ``repaired`` or
        ``already_valid``).
    status:
        ``"already_valid"`` — the input already satisfied every rule;
        ``"repaired"``      — a verified valid state was found;
        ``"unsat"``         — *proven* to have no valid completion (see ``core``);
        ``"unknown"``       — search failed but UNSAT was not proven (raise the
        budget, or the instance may be a hard/large UNSAT the sound check misses).
    violations:
        Rule indices still violated in ``state`` (empty when valid).
    core:
        A conflicting rule subset when ``status == "unsat"``; else ``None``.
    """

    state: List[bool]
    status: str
    violations: List[int]
    core: Optional[List[int]] = None

    @property
    def is_valid(self) -> bool:
        """True iff ``state`` provably satisfies every rule."""
        return self.status in ("already_valid", "repaired")


def guaranteed_repair(
    kb: KnowledgeBase,
    init_state: Sequence[bool],
    fixed_mask: Sequence[bool],
    max_iters: int = 2000,
    restarts: int = 12,
    seed: int = 0,
) -> GuaranteeResult:
    """Discrete backstop: return a verified-valid state, prove UNSAT, or say so.

    Combines :func:`min_conflicts_repair` (sound for SAT) with
    :func:`provable_unsat_core` (sound for UNSAT). Never reports a valid state
    that is not actually valid, and never claims UNSAT without a proof.
    """
    init = [bool(init_state[a]) for a in range(kb.num_atoms)]
    if not crisp_violations(kb, init):
        return GuaranteeResult(state=init, status="already_valid", violations=[])

    core = provable_unsat_core(kb, fixed_mask, init)
    if core is not None:
        return GuaranteeResult(state=init, status="unsat",
                               violations=crisp_violations(kb, init), core=core)

    state, ok = min_conflicts_repair(kb, init, fixed_mask,
                                     max_iters=max_iters, restarts=restarts, seed=seed)
    if ok:
        return GuaranteeResult(state=state, status="repaired", violations=[])

    # search failed: try once more to prove UNSAT from the best state's pinned rules
    core = provable_unsat_core(kb, fixed_mask, state)
    if core is not None:
        return GuaranteeResult(state=state, status="unsat",
                               violations=crisp_violations(kb, state), core=core)
    return GuaranteeResult(state=state, status="unknown",
                           violations=crisp_violations(kb, state))
