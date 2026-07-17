"""Differentiable Theorem Proving (DTP) engine for Project ThermoLogic.

This module translates *discrete* symbolic logic into a *continuous,
differentiable* computational graph using fuzzy logic **t-norms**. Boolean truth
values ``{True, False}`` are relaxed to the real interval ``[0, 1]`` and the
logical connectives (``AND``, ``OR``, ``NOT``, ``IMPLIES``) are replaced by their
fuzzy counterparts, whose corner cases (``0``/``1``) agree exactly with Boolean
logic while remaining smooth (and therefore back-propagatable) in between.

The public entry point is :class:`DifferentiableLogicEngine`, an ``nn.Module``
that, given a batch of probabilistic atom beliefs ``p ∈ [0, 1]^A``, returns the
per-rule **satisfaction** tensor ``sat ∈ [0, 1]^R`` — the raw material the EBM in
``model.py`` turns into an energy landscape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Sequence, Tuple

import torch
from torch import Tensor, nn

__all__ = [
    "TNorm",
    "Literal",
    "ImplicationRule",
    "KnowledgeBase",
    "DifferentiableLogicEngine",
    "EPS",
]

#: Small constant used to keep truth values strictly inside ``(0, 1)`` and to
#: avoid division by zero in the Product residuum.
EPS: float = 1e-7


class TNorm(str, Enum):
    """Supported fuzzy-logic t-norm families.

    Each family fixes a consistent triple of (conjunction, disjunction,
    residuated implication). See :class:`DifferentiableLogicEngine` for the
    concrete tensor implementations.
    """

    PRODUCT = "product"
    LUKASIEWICZ = "lukasiewicz"
    GODEL = "godel"


# --------------------------------------------------------------------------- #
# Symbolic rule representation
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Literal:
    """A single (possibly negated) atomic proposition.

    Parameters
    ----------
    atom:
        Index of the atom within the knowledge base (``0 <= atom < num_atoms``).
    negated:
        If ``True`` the literal denotes ``NOT atom``; the fuzzy truth of the
        literal is then ``1 - p[atom]``.
    """

    atom: int
    negated: bool = False

    def truth(self, probs: Tensor) -> Tensor:
        """Return the fuzzy truth of this literal for a batch of beliefs.

        Parameters
        ----------
        probs:
            Tensor of shape ``(batch, num_atoms)`` with values in ``[0, 1]``.

        Returns
        -------
        Tensor
            Shape ``(batch,)`` fuzzy truth values in ``[0, 1]``.
        """
        value = probs[:, self.atom]
        return (1.0 - value) if self.negated else value

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"{'¬' if self.negated else ''}x{self.atom}"


@dataclass(frozen=True)
class ImplicationRule:
    """A Horn-style implication ``L1 ∧ L2 ∧ ... ∧ Lk  ->  C``.

    An empty ``antecedents`` list encodes a *fact* (antecedent truth ``1``),
    i.e. an unconditional assertion of the consequent.

    Parameters
    ----------
    antecedents:
        Conjunction of literals forming the rule body.
    consequent:
        The single literal implied by the body.
    name:
        Human-readable label used in diagnostics.
    """

    antecedents: Tuple[Literal, ...]
    consequent: Literal
    name: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.consequent, Literal):
            raise TypeError("consequent must be a Literal")
        for lit in self.antecedents:
            if not isinstance(lit, Literal):
                raise TypeError("every antecedent must be a Literal")

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        body = " ∧ ".join(str(lit) for lit in self.antecedents) or "⊤"
        return f"{self.name or 'rule'}: {body} → {self.consequent}"


@dataclass
class KnowledgeBase:
    """A validated collection of atoms and implication rules.

    Parameters
    ----------
    atom_names:
        Ordered names of the atomic propositions; ``len`` defines ``num_atoms``.
    rules:
        The symbolic implication rules over those atoms.
    cause_atoms:
        Indices of *observed* atoms (supervised from the input).
    derived_atoms:
        Indices of *inferred* atoms (constrained only through the energy).
    """

    atom_names: Tuple[str, ...]
    rules: Tuple[ImplicationRule, ...]
    cause_atoms: Tuple[int, ...] = field(default_factory=tuple)
    derived_atoms: Tuple[int, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        self.validate()

    @property
    def num_atoms(self) -> int:
        """Number of atomic propositions in the knowledge base."""
        return len(self.atom_names)

    @property
    def num_rules(self) -> int:
        """Number of implication rules in the knowledge base."""
        return len(self.rules)

    def validate(self) -> None:
        """Check that every literal references a valid atom index.

        Raises
        ------
        ValueError
            If any literal (in an antecedent or consequent) points outside the
            range ``[0, num_atoms)``, or if the knowledge base is empty.
        """
        if self.num_atoms == 0:
            raise ValueError("KnowledgeBase must contain at least one atom")
        if self.num_rules == 0:
            raise ValueError("KnowledgeBase must contain at least one rule")

        def _check(lit: Literal, where: str) -> None:
            if not (0 <= lit.atom < self.num_atoms):
                raise ValueError(
                    f"atom index {lit.atom} in {where} is out of range "
                    f"[0, {self.num_atoms})"
                )

        for rule in self.rules:
            for lit in rule.antecedents:
                _check(lit, f"antecedent of {rule.name!r}")
            _check(rule.consequent, f"consequent of {rule.name!r}")

        for idx in (*self.cause_atoms, *self.derived_atoms):
            if not (0 <= idx < self.num_atoms):
                raise ValueError(f"atom index {idx} is out of range")


# --------------------------------------------------------------------------- #
# Differentiable logic engine
# --------------------------------------------------------------------------- #
class DifferentiableLogicEngine(nn.Module):
    """Evaluate a :class:`KnowledgeBase` under differentiable fuzzy logic.

    The engine is stateless (it holds no learnable parameters); it purely maps a
    batch of probabilistic beliefs to a batch of per-rule satisfaction values via
    a chosen t-norm family. Because every operation is a smooth (or almost-
    everywhere-smooth) tensor op, gradients flow back to whatever produced the
    beliefs — e.g. the Neural Proposer.

    Parameters
    ----------
    kb:
        The knowledge base to evaluate.
    tnorm:
        Which t-norm family to use for conjunction and implication. Defaults to
        Łukasiewicz, whose residuum has bounded, division-free gradients.
    """

    def __init__(self, kb: KnowledgeBase, tnorm: TNorm | str = TNorm.LUKASIEWICZ) -> None:
        super().__init__()
        if not isinstance(kb, KnowledgeBase):
            raise TypeError("kb must be a KnowledgeBase instance")
        kb.validate()
        self.kb: KnowledgeBase = kb
        self.tnorm: TNorm = TNorm(tnorm)

        # Dispatch tables keep the forward pass branch-free and readable.
        self._conj: Dict[TNorm, Callable[[Tensor, Tensor], Tensor]] = {
            TNorm.PRODUCT: self._conj_product,
            TNorm.LUKASIEWICZ: self._conj_lukasiewicz,
            TNorm.GODEL: self._conj_godel,
        }
        self._impl: Dict[TNorm, Callable[[Tensor, Tensor], Tensor]] = {
            TNorm.PRODUCT: self._residuum_product,
            TNorm.LUKASIEWICZ: self._residuum_lukasiewicz,
            TNorm.GODEL: self._residuum_godel,
        }

    # ---- fuzzy conjunction (t-norm) --------------------------------------- #
    @staticmethod
    def _conj_product(a: Tensor, b: Tensor) -> Tensor:
        return a * b

    @staticmethod
    def _conj_lukasiewicz(a: Tensor, b: Tensor) -> Tensor:
        return torch.clamp(a + b - 1.0, min=0.0)

    @staticmethod
    def _conj_godel(a: Tensor, b: Tensor) -> Tensor:
        return torch.minimum(a, b)

    # ---- fuzzy residuated implication  a => c ----------------------------- #
    @staticmethod
    def _residuum_product(a: Tensor, c: Tensor) -> Tensor:
        # Goguen residuum: 1 if a <= c else c / a. Clamp guards the division.
        ratio = c / torch.clamp(a, min=EPS)
        return torch.clamp(torch.where(a <= c, torch.ones_like(a), ratio), max=1.0)

    @staticmethod
    def _residuum_lukasiewicz(a: Tensor, c: Tensor) -> Tensor:
        return torch.clamp(1.0 - a + c, max=1.0)

    @staticmethod
    def _residuum_godel(a: Tensor, c: Tensor) -> Tensor:
        return torch.where(a <= c, torch.ones_like(a), c)

    # ---- rule / knowledge-base evaluation --------------------------------- #
    def _antecedent_truth(self, rule: ImplicationRule, probs: Tensor) -> Tensor:
        """Fuzzy conjunction of a rule's antecedent literals.

        An empty antecedent (a fact) yields constant truth ``1``.
        """
        conj = self._conj[self.tnorm]
        if not rule.antecedents:
            return torch.ones(probs.shape[0], device=probs.device, dtype=probs.dtype)
        acc = rule.antecedents[0].truth(probs)
        for lit in rule.antecedents[1:]:
            acc = conj(acc, lit.truth(probs))
        return acc

    def rule_satisfaction(self, rule: ImplicationRule, probs: Tensor) -> Tensor:
        """Satisfaction of a single implication rule for a batch of beliefs.

        Returns a tensor of shape ``(batch,)`` in ``[0, 1]`` where ``1`` means
        the rule is fully satisfied and ``0`` a hard contradiction.
        """
        antecedent = self._antecedent_truth(rule, probs)
        consequent = rule.consequent.truth(probs)
        return self._impl[self.tnorm](antecedent, consequent)

    def satisfaction(self, probs: Tensor, validate: bool = True) -> Tensor:
        """Per-rule satisfaction for a batch of beliefs.

        Parameters
        ----------
        probs:
            Tensor of shape ``(batch, num_atoms)`` with values in ``[0, 1]``.
        validate:
            Whether to range-check ``probs``. Defaults to ``True``. Hot inner
            loops (e.g. test-time repair, where beliefs are already sigmoid-bounded)
            pass ``False`` to skip a per-call device synchronization.

        Returns
        -------
        Tensor
            Shape ``(batch, num_rules)`` satisfaction values in ``[0, 1]``.

        Raises
        ------
        ValueError
            If ``probs`` has the wrong shape or values outside ``[0, 1]``.
        """
        if validate:
            self._check_probs(probs)
        columns: List[Tensor] = [
            self.rule_satisfaction(rule, probs) for rule in self.kb.rules
        ]
        return torch.stack(columns, dim=1)

    def forward(self, probs: Tensor) -> Tensor:  # noqa: D401 - see satisfaction
        """Alias for :meth:`satisfaction` so the engine composes as a module."""
        return self.satisfaction(probs)

    # ---- validation ------------------------------------------------------- #
    def _check_probs(self, probs: Tensor) -> None:
        if not isinstance(probs, Tensor):
            raise TypeError("probs must be a torch.Tensor")
        if probs.dim() != 2 or probs.shape[1] != self.kb.num_atoms:
            raise ValueError(
                f"probs must have shape (batch, {self.kb.num_atoms}); "
                f"got {tuple(probs.shape)}"
            )
        # Only validate ranges on finite tensors to avoid tripping on NaNs that
        # the caller may want surfaced by the loss instead. Detach first so the
        # cheap range check never touches the autograd graph.
        with torch.no_grad():
            if torch.isfinite(probs).all():
                mn = float(probs.min())
                mx = float(probs.max())
            else:
                return
            if mn < -EPS or mx > 1.0 + EPS:
                raise ValueError(
                    f"probs must lie in [0, 1]; observed range [{mn:.4f}, {mx:.4f}]"
                )


def forward_chaining(kb: KnowledgeBase, causes: Sequence[bool]) -> List[bool]:
    """Compute the crisp *minimal model* of ``kb`` given Boolean cause values.

    A least-fixed-point forward-chaining pass over positive Horn rules: an atom
    is ``True`` iff some rule with a fully-satisfied positive body forces it,
    starting from the provided cause atoms. This yields the deterministic ground
    truth against which the neural derivation is measured in ``train.py``.

    Parameters
    ----------
    kb:
        The knowledge base.
    causes:
        Boolean values aligned with ``kb.cause_atoms``.

    Returns
    -------
    list of bool
        The full ``num_atoms``-length truth assignment of the minimal model.

    Raises
    ------
    ValueError
        If ``len(causes)`` does not match ``len(kb.cause_atoms)``.
    """
    if len(causes) != len(kb.cause_atoms):
        raise ValueError("number of cause values must match kb.cause_atoms")

    state: List[bool] = [False] * kb.num_atoms
    for idx, value in zip(kb.cause_atoms, causes):
        state[idx] = bool(value)

    changed = True
    while changed:
        changed = False
        for rule in kb.rules:
            # Only positive-body, positive-head rules propagate truth in the
            # minimal model; negative constraints act as filters on causes.
            if rule.consequent.negated:
                continue
            if any(lit.negated for lit in rule.antecedents):
                continue
            body_holds = all(state[lit.atom] for lit in rule.antecedents)
            if body_holds and not state[rule.consequent.atom]:
                state[rule.consequent.atom] = True
                changed = True
    return state
