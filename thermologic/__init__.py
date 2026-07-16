"""Project ThermoLogic — a differentiable logic-energy layer for AI outputs.

Fuses Differentiable Theorem Proving (fuzzy-logic t-norms) with an Energy-Based
Model so that logically valid states sit at low energy and contradictions cost
exponentially more. Use :class:`LogicEnergy` to *score* how much any model's
output violates a rule set (a label-free inconsistency signal) and to *repair*
it to the nearest valid state at a compute budget you control.

Quick start
-----------
>>> from thermologic import LogicEnergy, implies
>>> guard = LogicEnergy(
...     rules=[implies(["a"], "b", name="a⇒b")],
...     atom_names=["a", "b"],
... )
>>> import torch
>>> guard.score(torch.tensor([[1.0, 0.0]])) > 0   # a true, b false ⇒ violation
tensor([True])
"""

from thermologic.api import LogicEnergy, atom_index, implies
from thermologic.data import (
    Dataset,
    WorldSplit,
    enumerate_worlds,
    sample_from_worlds,
    split_worlds,
)
from thermologic.logic_engine import (
    DifferentiableLogicEngine,
    ImplicationRule,
    KnowledgeBase,
    Literal,
    TNorm,
    forward_chaining,
)
from thermologic.model import (
    EBMOutput,
    EnergyBasedModel,
    LossBreakdown,
    NeuralProposer,
    RepairResult,
    ThermoLogicLoss,
    repair_beliefs,
)

__version__ = "0.1.0"

__all__ = [
    "LogicEnergy",
    "implies",
    "atom_index",
    "TNorm",
    "Literal",
    "ImplicationRule",
    "KnowledgeBase",
    "DifferentiableLogicEngine",
    "forward_chaining",
    "NeuralProposer",
    "EnergyBasedModel",
    "EBMOutput",
    "ThermoLogicLoss",
    "LossBreakdown",
    "repair_beliefs",
    "RepairResult",
    "Dataset",
    "WorldSplit",
    "enumerate_worlds",
    "split_worlds",
    "sample_from_worlds",
    "__version__",
]
