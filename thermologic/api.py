"""High-level product API for Project ThermoLogic.

:class:`LogicEnergy` is the one class most users need. It turns a set of symbolic
rules over named propositions into a *differentiable logic-energy layer* that can
be dropped after any model to:

* **score** how much an output violates the rules (a label-free inconsistency /
  hallucination signal), and
* **repair** an output to the nearest rule-satisfying state, at a compute budget
  you control (test-time energy descent).

Example
-------
>>> from thermologic import LogicEnergy, implies
>>> atoms = ["plan_free", "seats_gt_1", "sso_enabled", "plan_enterprise"]
>>> rules = [
...     implies(["plan_free"], "seats_gt_1", negate_consequent=True, name="free⇒≤1 seat"),
...     implies(["sso_enabled"], "plan_enterprise", name="sso⇒enterprise"),
... ]
>>> guard = LogicEnergy(rules, atom_names=atoms)
>>> import torch
>>> out = torch.tensor([[0.9, 0.8, 0.7, 0.1]])   # a model's (inconsistent) output
>>> float(guard.score(out)) > 0                    # flagged as inconsistent
True
>>> fixed = guard.repair(out, budget=60)           # nearest valid config
"""

from __future__ import annotations

import json
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor

from thermologic.logic_engine import (
    DifferentiableLogicEngine,
    ImplicationRule,
    KnowledgeBase,
    Literal,
    TNorm,
)
from thermologic.model import EnergyBasedModel, NeuralProposer, RepairResult, repair_beliefs
from thermologic.solver import GuaranteeResult, crisp_violations, guaranteed_repair

__all__ = ["LogicEnergy", "implies", "atom_index", "UnsatisfiableError"]


class UnsatisfiableError(ValueError):
    """Raised by :meth:`LogicEnergy.solve` / guaranteed repair when no valid state
    exists under the fixed atoms. Carries the conflicting rule names in ``core``."""

    def __init__(self, message: str, core: Optional[List[str]] = None) -> None:
        super().__init__(message)
        self.core: List[str] = core or []

RuleLike = Union[ImplicationRule, "Tuple"]


def atom_index(name_or_idx: Union[int, str], atom_names: Sequence[str]) -> int:
    """Resolve an atom given by name or index to its integer index."""
    if isinstance(name_or_idx, int):
        return name_or_idx
    try:
        return list(atom_names).index(name_or_idx)
    except ValueError as exc:  # pragma: no cover - user error path
        raise ValueError(f"unknown atom {name_or_idx!r}; known: {list(atom_names)}") from exc


def implies(
    antecedents: Sequence[Union[int, str]],
    consequent: Union[int, str],
    *,
    negate_consequent: bool = False,
    negate_antecedents: Optional[Sequence[bool]] = None,
    name: str = "",
) -> Tuple:
    """Friendly builder for an implication rule, using atom **names or indices**.

    Returns an opaque spec resolved by :class:`LogicEnergy` once atom names are
    known. ``negate_consequent`` encodes ``… → ¬consequent`` (useful for mutual
    exclusion); ``negate_antecedents`` optionally negates individual body atoms.
    """
    if negate_antecedents is None:
        negate_antecedents = [False] * len(antecedents)
    if len(negate_antecedents) != len(antecedents):
        raise ValueError("negate_antecedents must match antecedents length")
    return ("implies", list(antecedents), list(negate_antecedents),
            consequent, negate_consequent, name)


def _resolve_rule(spec: RuleLike, atom_names: Sequence[str], i: int) -> ImplicationRule:
    if isinstance(spec, ImplicationRule):
        return spec
    kind = spec[0]
    if kind != "implies":  # pragma: no cover - defensive
        raise ValueError(f"unknown rule spec {kind!r}")
    _, ants, neg_ants, cons, neg_cons, name = spec
    antecedents = tuple(
        Literal(atom_index(a, atom_names), bool(n)) for a, n in zip(ants, neg_ants)
    )
    consequent = Literal(atom_index(cons, atom_names), bool(neg_cons))
    return ImplicationRule(antecedents, consequent, name or f"rule{i}")


class LogicEnergy:
    """A differentiable logic-energy layer over named propositional atoms.

    Parameters
    ----------
    rules:
        Rules over the atoms. Each is either an :class:`ImplicationRule` or a spec
        produced by :func:`implies` (recommended — lets you use atom names).
    atom_names:
        Names of the atoms (defines the output vector's columns and order).
    tnorm:
        Fuzzy t-norm family: ``"lukasiewicz"`` (default, smoothest gradients),
        ``"product"``, or ``"godel"``.
    beta:
        Inverse-temperature sharpness of the energy penalty (``> 0``).
    """

    def __init__(
        self,
        rules: Sequence[RuleLike],
        atom_names: Sequence[str],
        tnorm: Union[TNorm, str] = TNorm.LUKASIEWICZ,
        beta: float = 4.0,
    ) -> None:
        if not atom_names:
            raise ValueError("atom_names must be non-empty")
        resolved = tuple(_resolve_rule(r, atom_names, i) for i, r in enumerate(rules))
        self.kb = KnowledgeBase(
            atom_names=tuple(atom_names),
            rules=resolved,
            derived_atoms=tuple(range(len(atom_names))),
        )
        self.engine = DifferentiableLogicEngine(self.kb, tnorm=tnorm)
        # A dummy proposer lets us reuse the EBM's energy + repair machinery; it
        # is never trained and touches no user data.
        self._ebm = EnergyBasedModel(NeuralProposer(1, self.kb.num_atoms), self.engine, beta=beta)
        self.beta = float(beta)

    # -- introspection ------------------------------------------------------ #
    @property
    def atom_names(self) -> Tuple[str, ...]:
        """Ordered atom names (the output vector's columns)."""
        return self.kb.atom_names

    @property
    def num_atoms(self) -> int:
        """Number of atoms."""
        return self.kb.num_atoms

    @classmethod
    def from_knowledge_base(
        cls, kb: KnowledgeBase, tnorm: Union[TNorm, str] = TNorm.LUKASIEWICZ, beta: float = 4.0
    ) -> "LogicEnergy":
        """Build directly from an existing :class:`KnowledgeBase`."""
        obj = cls.__new__(cls)
        obj.kb = kb
        obj.engine = DifferentiableLogicEngine(kb, tnorm=tnorm)
        obj._ebm = EnergyBasedModel(NeuralProposer(1, kb.num_atoms), obj.engine, beta=beta)
        obj.beta = float(beta)
        return obj

    # -- serialization (rules as data, not code) ---------------------------- #
    def to_dict(self) -> Dict[str, Any]:
        """Serialize the atoms, rules, t-norm, and beta to a JSON-friendly dict.

        Atoms are referenced by name, so the rule set stays human-readable and can
        live in a config file rather than in Python.
        """
        names = self.atom_names

        def lit(li: Literal) -> Dict[str, Any]:
            return {"atom": names[li.atom], "negated": bool(li.negated)}

        return {
            "atom_names": list(names),
            "tnorm": self.engine.tnorm.value,
            "beta": self.beta,
            "rules": [
                {
                    "antecedents": [lit(a) for a in r.antecedents],
                    "consequent": lit(r.consequent),
                    "name": r.name,
                }
                for r in self.kb.rules
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LogicEnergy":
        """Reconstruct a :class:`LogicEnergy` from :meth:`to_dict` output."""
        atom_names = data["atom_names"]
        rules = [
            implies(
                [a["atom"] for a in r["antecedents"]],
                r["consequent"]["atom"],
                negate_consequent=bool(r["consequent"].get("negated", False)),
                negate_antecedents=[bool(a.get("negated", False)) for a in r["antecedents"]],
                name=r.get("name", ""),
            )
            for r in data["rules"]
        ]
        return cls(rules, atom_names=atom_names,
                   tnorm=data.get("tnorm", TNorm.LUKASIEWICZ.value),
                   beta=float(data.get("beta", 4.0)))

    def save(self, path: str) -> None:
        """Write the rule set to a JSON file."""
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "LogicEnergy":
        """Load a rule set previously written by :meth:`save`."""
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    # -- scoring ------------------------------------------------------------ #
    def _as_batch(self, outputs: Tensor) -> Tensor:
        if not isinstance(outputs, Tensor):
            outputs = torch.as_tensor(outputs, dtype=torch.float32)
        if outputs.dim() == 1:
            outputs = outputs.unsqueeze(0)
        if outputs.dim() != 2 or outputs.shape[1] != self.num_atoms:
            raise ValueError(
                f"outputs must have shape (batch, {self.num_atoms}); got {tuple(outputs.shape)}"
            )
        return outputs.float()

    def satisfaction(self, outputs: Tensor) -> Tensor:
        """Per-rule fuzzy satisfaction, shape ``(batch, num_rules)`` in ``[0, 1]``."""
        return self.engine.satisfaction(self._as_batch(outputs))

    def score(self, outputs: Tensor) -> Tensor:
        """Logical-energy score per sample (``0`` = fully consistent, higher = worse).

        This is a **label-free** measure of how much the output violates the rules
        — usable directly as a hallucination / confidence / abstention signal.
        """
        batch = self._as_batch(outputs)
        return self._ebm.energy_from_satisfaction(self.engine.satisfaction(batch))

    def is_consistent(self, outputs: Tensor, tol: float = 1e-3, crisp: bool = False) -> Tensor:
        """Boolean per-sample mask of whether every rule holds.

        Parameters
        ----------
        outputs:
            ``(batch, num_atoms)`` (or ``(num_atoms,)``) belief vector.
        tol:
            Soft-energy tolerance (used when ``crisp=False``).
        crisp:
            If ``True``, threshold the beliefs to ``{0, 1}`` first and require
            every rule to hold on the *discrete* state — the right check when the
            output represents a decision/configuration rather than a probability.
        """
        if crisp:
            return self._crisp_consistent(self._as_batch(outputs))
        return self.score(outputs) <= tol

    def _crisp_consistent(self, batch: Tensor) -> Tensor:
        """Per-sample mask: does the thresholded (0/1) state satisfy every rule?"""
        crisp = (batch > 0.5).float()
        sat = self.engine.satisfaction(crisp, validate=False)
        return (sat > 0.5).all(dim=1)

    def violations(self, outputs: Tensor, threshold: float = 0.5) -> List[List[str]]:
        """Names of the rules each sample violates (fuzzy satisfaction < threshold)."""
        sat = self.satisfaction(outputs)
        names = [r.name for r in self.kb.rules]
        return [
            [names[j] for j in range(sat.shape[1]) if float(sat[i, j]) < threshold]
            for i in range(sat.shape[0])
        ]

    # -- repair ------------------------------------------------------------- #
    def repair(
        self,
        outputs: Tensor,
        fixed: Optional[Sequence[Union[int, str]]] = None,
        budget: int = 60,
        lr: float = 0.3,
        optimizer: str = "adam",
        parsimony: float = 0.0,
        snap: bool = False,
        verify: bool = False,
        max_budget: int = 480,
        return_trace: bool = False,
        guarantee: bool = False,
    ) -> Union[Tensor, RepairResult]:
        """Repair an output to the nearest rule-satisfying state (energy descent).

        Parameters
        ----------
        outputs:
            ``(batch, num_atoms)`` (or ``(num_atoms,)``) belief/probability vector.
        fixed:
            Atoms (names or indices) to hold constant during repair — e.g. inputs
            you trust and do not want the repair to change. Defaults to none.
        budget:
            Number of energy-descent steps (your test-time compute budget). Because
            repair cost scales with constraint depth, deeper rule chains need a
            larger budget.
        lr:
            Optimizer learning rate.
        optimizer:
            ``"adam"`` (default) or ``"sgd"``.
        parsimony:
            Optional Occam pressure toward ``false`` on the free atoms; ``0`` (the
            default) repairs to the nearest state without a truth-minimizing bias.
        snap:
            If ``True``, threshold the repaired beliefs to a crisp ``{0, 1}`` state
            before returning — what you want when the output is a discrete
            decision/configuration rather than a probability.
        verify:
            If ``True``, check that the (thresholded) repaired state actually
            satisfies every rule, and automatically double the budget (up to
            ``max_budget``) until it does. Emits a warning if it still cannot —
            so the caller is never handed a silently-invalid "repaired" output.
        max_budget:
            Ceiling for ``verify`` budget escalation.
        return_trace:
            If ``True`` return the full :class:`RepairResult` (with the energy
            trajectory) instead of just the repaired tensor. Note its ``beliefs``
            field is the soft (un-snapped) state.

        guarantee:
            If ``True``, back the soft energy descent with a **discrete** solver
            (min-conflicts local search) so the returned state is *verified*
            crisp-valid — closing the "soft energy is expensive, not impossible"
            gap. Implies ``snap=True``. Raises :class:`UnsatisfiableError` if a
            sample is *proven* to have no valid completion under ``fixed`` (with
            the conflicting rule names), or ``RuntimeError`` if the solver cannot
            reach validity within its budget and UNSAT was not proven.

        Returns
        -------
        Tensor or RepairResult
            The repaired belief state — crisp if ``snap=True`` — or the full
            :class:`RepairResult` if ``return_trace=True``.
        """
        batch = self._as_batch(outputs)
        fixed_mask = torch.zeros(self.num_atoms)
        if fixed:
            for a in fixed:
                fixed_mask[atom_index(a, self.atom_names)] = 1.0

        def _run(steps: int) -> RepairResult:
            return repair_beliefs(
                self._ebm, batch, fixed_mask, steps=steps, lr=lr,
                parsimony_weight=parsimony, optimizer=optimizer,
                log_every=max(1, steps // 20),
            )

        if guarantee:
            warm = _run(budget).beliefs if budget > 0 else batch
            fmask = [bool(fixed_mask[a] > 0.5) for a in range(self.num_atoms)]
            rows: List[Tensor] = []
            for i in range(warm.shape[0]):
                init = [bool(warm[i, a] > 0.5) for a in range(self.num_atoms)]
                res = guaranteed_repair(self.kb, init, fmask)
                if res.status == "unsat":
                    names = [self.kb.rules[j].name for j in (res.core or [])]
                    raise UnsatisfiableError(
                        f"sample {i} has no valid completion under the fixed atoms; "
                        f"conflicting rule(s): {names}", core=names)
                if not res.is_valid:
                    raise RuntimeError(
                        f"guaranteed repair could not reach a valid state for sample {i} "
                        f"(residual violations: {len(res.violations)}); raise max_iters/restarts "
                        f"or check the rule set")
                rows.append(torch.tensor([1.0 if v else 0.0 for v in res.state]))
            return torch.stack(rows)

        result = _run(budget)
        if verify:
            used = budget
            while used < max_budget and not bool(self._crisp_consistent(result.beliefs).all()):
                used = min(max_budget, used * 2)
                result = _run(used)
            still_bad = int((~self._crisp_consistent(result.beliefs)).sum())
            if still_bad:
                warnings.warn(
                    f"repair could not reach a crisp-valid state within budget={max_budget} "
                    f"for {still_bad}/{batch.shape[0]} sample(s); returning the lowest-energy "
                    f"state found. Try a larger max_budget or lr.",
                    RuntimeWarning, stacklevel=2,
                )

        if return_trace:
            return result
        beliefs = result.beliefs
        return (beliefs > 0.5).float() if snap else beliefs

    # -- discrete guarantees ----------------------------------------------- #
    def satisfiability(
        self,
        outputs: Optional[Tensor] = None,
        fixed: Optional[Sequence[Union[int, str]]] = None,
        max_iters: int = 2000,
        restarts: int = 12,
        seed: int = 0,
    ) -> List[GuaranteeResult]:
        """Per-sample discrete satisfiability of the rules under the fixed atoms.

        Returns a :class:`GuaranteeResult` per sample whose ``status`` is one of
        ``already_valid`` / ``repaired`` / ``unsat`` / ``unknown`` (see
        :class:`thermologic.solver.GuaranteeResult`). Sound in both directions: a
        ``repaired``/``already_valid`` state is genuinely valid, and ``unsat`` is a
        proof — never a guess. ``outputs`` (thresholded) seeds the search; if
        omitted the all-false state is used.
        """
        if outputs is None:
            batch = torch.zeros(1, self.num_atoms)
        else:
            batch = self._as_batch(outputs)
        fmask = [False] * self.num_atoms
        if fixed:
            for a in fixed:
                fmask[atom_index(a, self.atom_names)] = True
        results: List[GuaranteeResult] = []
        for i in range(batch.shape[0]):
            init = [bool(batch[i, a] > 0.5) for a in range(self.num_atoms)]
            results.append(guaranteed_repair(self.kb, init, fmask,
                                             max_iters=max_iters, restarts=restarts, seed=seed))
        return results

    def solve(
        self,
        observations: Optional[Dict[str, bool]] = None,
        seed: int = 0,
        max_iters: int = 2000,
        restarts: int = 12,
    ) -> Tensor:
        """Find one **verified-valid** crisp assignment consistent with the given
        atom observations (held fixed), or raise :class:`UnsatisfiableError`.

        Unlike :meth:`repair`, this needs no model output — it is a small built-in
        constraint solver over the rule set, useful for "complete this partial
        configuration to a valid one" tasks.
        """
        observations = observations or {}
        init = torch.zeros(1, self.num_atoms)
        fmask = [False] * self.num_atoms
        for name, val in observations.items():
            idx = atom_index(name, self.atom_names)
            init[0, idx] = 1.0 if val else 0.0
            fmask[idx] = True
        state = [bool(init[0, a] > 0.5) for a in range(self.num_atoms)]
        res = guaranteed_repair(self.kb, state, fmask, max_iters=max_iters, restarts=restarts, seed=seed)
        if res.status == "unsat":
            names = [self.kb.rules[j].name for j in (res.core or [])]
            raise UnsatisfiableError(
                f"no valid assignment satisfies the observations; conflicting rule(s): {names}",
                core=names)
        if not res.is_valid:
            raise RuntimeError(
                f"solver could not find a valid assignment (residual violations: "
                f"{len(res.violations)}); raise max_iters/restarts")
        return torch.tensor([[1.0 if v else 0.0 for v in res.state]])
