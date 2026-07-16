"""Neural Proposer and Energy-Based Model (EBM) for Project ThermoLogic.

This module holds the two neural halves of the hybrid architecture:

* :class:`NeuralProposer` — an MLP that maps an input encoding to *probabilistic
  truth values* (beliefs) over the atomic propositions. This is the
  "generative" / exploratory component: it proposes a belief state in the
  continuous latent space ``[0, 1]^A``.

* :class:`EnergyBasedModel` — composes the proposer with the differentiable
  logic engine (the DTP layer) and turns per-rule satisfaction into a scalar
  **energy**. Logically valid belief states sit at the bottom of the energy well
  (``E ≈ 0``); contradictions incur an *exponential* penalty.

* :class:`ThermoLogicLoss` — the composite training objective: supervised
  evidence (anchoring beliefs to the input) + logical energy (constraining the
  derived atoms) + a parsimony term (making the minimal model the unique
  minimum).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
from torch import Tensor, nn

from thermologic.logic_engine import DifferentiableLogicEngine, KnowledgeBase, TNorm

__all__ = [
    "NeuralProposer",
    "EnergyBasedModel",
    "EBMOutput",
    "ThermoLogicLoss",
    "LossBreakdown",
    "repair_beliefs",
    "RepairResult",
]


class NeuralProposer(nn.Module):
    """MLP that proposes probabilistic truth values for the atomic propositions.

    The final ``Sigmoid`` guarantees every output lies in ``(0, 1)``, so the
    proposal is always a valid fuzzy belief state that the logic engine can
    consume directly.

    Parameters
    ----------
    input_dim:
        Dimensionality of the input encoding.
    num_atoms:
        Number of atoms (size of the output belief vector).
    hidden_dims:
        Width of each hidden layer.
    """

    def __init__(
        self,
        input_dim: int,
        num_atoms: int,
        hidden_dims: Sequence[int] = (64, 64),
    ) -> None:
        super().__init__()
        if input_dim <= 0 or num_atoms <= 0:
            raise ValueError("input_dim and num_atoms must be positive")

        layers: list[nn.Module] = []
        prev = input_dim
        for width in hidden_dims:
            if width <= 0:
                raise ValueError("hidden layer widths must be positive")
            layers.append(nn.Linear(prev, width))
            layers.append(nn.ReLU())
            prev = width
        layers.append(nn.Linear(prev, num_atoms))
        layers.append(nn.Sigmoid())
        self.net: nn.Sequential = nn.Sequential(*layers)
        self.num_atoms: int = num_atoms
        self.input_dim: int = input_dim

    def forward(self, x: Tensor) -> Tensor:
        """Map an input batch to a batch of belief vectors in ``(0, 1)``.

        Parameters
        ----------
        x:
            Tensor of shape ``(batch, input_dim)``.

        Returns
        -------
        Tensor
            Shape ``(batch, num_atoms)`` probabilistic beliefs.
        """
        if x.dim() != 2 or x.shape[1] != self.input_dim:
            raise ValueError(
                f"expected input of shape (batch, {self.input_dim}); "
                f"got {tuple(x.shape)}"
            )
        return self.net(x)


@dataclass
class EBMOutput:
    """Container for a forward pass through the :class:`EnergyBasedModel`.

    Attributes
    ----------
    beliefs:
        ``(batch, num_atoms)`` probabilistic truth values from the proposer.
    satisfaction:
        ``(batch, num_rules)`` per-rule satisfaction from the DTP layer.
    energy:
        ``(batch,)`` scalar logical energy per sample.
    """

    beliefs: Tensor
    satisfaction: Tensor
    energy: Tensor


class EnergyBasedModel(nn.Module):
    """Neural Proposer + DTP logic engine wired into an energy landscape.

    The energy of a belief state is derived from per-rule satisfaction ``s_r``:

    .. math::

        E = \\operatorname{mean}_r \\big( \\exp(\\beta (1 - s_r)) - 1 \\big)

    which is ``0`` when every rule is satisfied and grows exponentially as rules
    are violated — the "false statements cost (near-)infinite energy" principle.

    Parameters
    ----------
    proposer:
        The belief-proposing network.
    engine:
        The differentiable logic engine (DTP layer).
    beta:
        Inverse-temperature sharpness of the energy penalty (``> 0``).
    """

    def __init__(
        self,
        proposer: NeuralProposer,
        engine: DifferentiableLogicEngine,
        beta: float = 4.0,
    ) -> None:
        super().__init__()
        if proposer.num_atoms != engine.kb.num_atoms:
            raise ValueError(
                "proposer.num_atoms must match the knowledge base's atom count"
            )
        if beta <= 0.0:
            raise ValueError("beta must be positive")
        self.proposer: NeuralProposer = proposer
        self.engine: DifferentiableLogicEngine = engine
        self.beta: float = float(beta)

    def energy_from_satisfaction(self, satisfaction: Tensor) -> Tensor:
        """Exponential energy from a per-rule satisfaction tensor.

        Parameters
        ----------
        satisfaction:
            ``(batch, num_rules)`` values in ``[0, 1]``.

        Returns
        -------
        Tensor
            ``(batch,)`` per-sample energy, ``>= 0``.
        """
        violation = 1.0 - satisfaction
        per_rule = torch.expm1(self.beta * violation)  # exp(x) - 1, stable
        return per_rule.mean(dim=1)

    def forward(self, x: Tensor) -> EBMOutput:
        """Run proposer → logic engine → energy for an input batch."""
        beliefs = self.proposer(x)
        satisfaction = self.engine.satisfaction(beliefs)
        energy = self.energy_from_satisfaction(satisfaction)
        return EBMOutput(beliefs=beliefs, satisfaction=satisfaction, energy=energy)


@dataclass
class LossBreakdown:
    """Scalar components of the composite loss, for logging.

    Attributes
    ----------
    total:
        The full objective actually back-propagated.
    supervised:
        BCE anchoring the observed (cause) atoms to the input.
    rule_energy:
        Mean logical energy from the EBM.
    parsimony:
        Mean Occam pressure on the derived atoms.
    """

    total: Tensor
    supervised: Tensor
    rule_energy: Tensor
    parsimony: Tensor


class ThermoLogicLoss(nn.Module):
    """Composite objective fusing supervised evidence with logical energy.

    .. math::

        L = w_{sup}\\,\\mathrm{BCE}(p_{causes}, y_{causes})
          + w_{rule}\\,\\overline{E}
          + w_{par}\\,\\operatorname{mean}(p_{derived})

    The supervision term anchors beliefs about *observed* atoms to the input; the
    energy term forces the *derived* atoms to obey the rule base; the parsimony
    term defaults unforced atoms to false so that the logical **minimal model**
    is the unique energy minimum.

    Parameters
    ----------
    kb:
        Knowledge base (supplies cause/derived atom indices).
    w_supervised, w_rule, w_parsimony:
        Non-negative weights for the three terms.
    """

    def __init__(
        self,
        kb: KnowledgeBase,
        w_supervised: float = 1.0,
        w_rule: float = 1.0,
        w_parsimony: float = 0.1,
    ) -> None:
        super().__init__()
        for name, value in (
            ("w_supervised", w_supervised),
            ("w_rule", w_rule),
            ("w_parsimony", w_parsimony),
        ):
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        self.kb: KnowledgeBase = kb
        self.w_supervised: float = float(w_supervised)
        self.w_rule: float = float(w_rule)
        self.w_parsimony: float = float(w_parsimony)
        self._bce = nn.BCELoss()

        cause_idx = torch.tensor(kb.cause_atoms, dtype=torch.long)
        derived_idx = torch.tensor(kb.derived_atoms, dtype=torch.long)
        # Registered as buffers so device moves (``.to``) keep them aligned.
        self.register_buffer("_cause_idx", cause_idx, persistent=False)
        self.register_buffer("_derived_idx", derived_idx, persistent=False)

    def forward(self, output: EBMOutput, targets: Tensor) -> LossBreakdown:
        """Compute the composite loss for one batch.

        Parameters
        ----------
        output:
            The :class:`EBMOutput` from the model's forward pass.
        targets:
            ``(batch, num_atoms)`` ground-truth truth values in ``{0, 1}``. Only
            the columns for ``kb.cause_atoms`` are used for supervision.

        Returns
        -------
        LossBreakdown
            The total loss and its individual components.
        """
        beliefs = output.beliefs
        if targets.shape != beliefs.shape:
            raise ValueError(
                f"targets shape {tuple(targets.shape)} must match beliefs "
                f"shape {tuple(beliefs.shape)}"
            )

        device = beliefs.device
        cause_idx = self._cause_idx.to(device)
        derived_idx = self._derived_idx.to(device)

        if cause_idx.numel() > 0:
            supervised = self._bce(
                beliefs.index_select(1, cause_idx),
                targets.index_select(1, cause_idx),
            )
        else:
            supervised = beliefs.new_zeros(())

        rule_energy = output.energy.mean()

        if derived_idx.numel() > 0:
            parsimony = beliefs.index_select(1, derived_idx).mean()
        else:
            parsimony = beliefs.new_zeros(())

        total = (
            self.w_supervised * supervised
            + self.w_rule * rule_energy
            + self.w_parsimony * parsimony
        )
        return LossBreakdown(
            total=total,
            supervised=supervised.detach(),
            rule_energy=rule_energy.detach(),
            parsimony=parsimony.detach(),
        )


@dataclass
class RepairResult:
    """Outcome of test-time energy repair (:func:`repair_beliefs`).

    Attributes
    ----------
    beliefs:
        ``(batch, num_atoms)`` repaired belief state.
    energy_trace:
        Mean energy recorded at each logged descent step, ``[(step, energy), …]``.
    steps:
        Number of gradient steps actually taken.
    """

    beliefs: Tensor
    energy_trace: List[Tuple[int, float]]
    steps: int


@torch.enable_grad()
def repair_beliefs(
    ebm: EnergyBasedModel,
    beliefs: Tensor,
    fixed_mask: Tensor,
    steps: int = 80,
    lr: float = 0.2,
    parsimony_weight: float = 0.1,
    parsimony_mask: Tensor | None = None,
    optimizer: str = "adam",
    log_every: int = 10,
) -> RepairResult:
    """Test-time logical inference by gradient descent on the energy landscape.

    This is the EBM's *native* (iterative) inference, as opposed to the Neural
    Proposer's single amortized feed-forward pass. Given an initial belief state
    — typically the proposer's output on a novel input — the free atoms are
    optimized to minimize ``energy + parsimony_weight · mean(free beliefs)`` while
    the ``fixed`` atoms (e.g. confidently-recovered cause atoms) are held
    constant. Optimization runs in logit space so beliefs stay in ``(0, 1)``.

    Because minimizing energy enforces every rule, this *repairs* a logically
    inconsistent proposal toward the nearest valid state without any labels.

    Parameters
    ----------
    ebm:
        Trained (or untrained) energy-based model providing the engine + energy.
    beliefs:
        ``(batch, num_atoms)`` initial belief state to repair.
    fixed_mask:
        ``(num_atoms,)`` mask with ``1`` for atoms to hold constant, ``0`` for
        atoms to optimize.
    steps:
        Number of gradient-descent steps.
    lr:
        Learning rate of the internal optimizer.
    parsimony_weight:
        Weight of the Occam pressure on free atoms (selects the minimal model).
    parsimony_mask:
        ``(num_atoms,)`` mask selecting which atoms the parsimony term applies to;
        defaults to the free atoms (``1 - fixed_mask``).
    optimizer:
        ``"adam"`` (default; robust for repairing a trained model's output) or
        ``"sgd"`` (momentum-free; gives the cleanest hop-by-hop propagation wave).
    log_every:
        Record the mean energy every ``log_every`` steps (and at the end).

    Returns
    -------
    RepairResult
        The repaired beliefs and the recorded energy trajectory.
    """
    if beliefs.dim() != 2:
        raise ValueError("beliefs must have shape (batch, num_atoms)")
    if fixed_mask.shape[-1] != beliefs.shape[1]:
        raise ValueError("fixed_mask must have length num_atoms")
    if steps <= 0:
        raise ValueError("steps must be positive")

    device = beliefs.device
    fixed = fixed_mask.to(device=device, dtype=beliefs.dtype).view(1, -1)
    free = 1.0 - fixed
    if parsimony_mask is None:
        par = free
    else:
        par = parsimony_mask.to(device=device, dtype=beliefs.dtype).view(1, -1)

    base = beliefs.detach().clamp(1e-4, 1.0 - 1e-4)
    logit = torch.logit(base).clone().requires_grad_(True)
    opt_name = optimizer.lower()
    if opt_name == "adam":
        opt = torch.optim.Adam([logit], lr=lr)
    elif opt_name == "sgd":
        opt = torch.optim.SGD([logit], lr=lr)
    else:
        raise ValueError(f"optimizer must be 'adam' or 'sgd'; got {optimizer!r}")

    def current() -> Tensor:
        return base * fixed + torch.sigmoid(logit) * free

    trace: List[Tuple[int, float]] = []
    for step in range(steps):
        opt.zero_grad()
        state = current()
        energy = ebm.energy_from_satisfaction(ebm.engine.satisfaction(state)).mean()
        # Mean belief over the parsimony-masked atoms, averaged over the batch.
        denom = (par.sum() * state.shape[0]).clamp(min=1.0)
        parsimony = (state * par).sum() / denom
        (energy + parsimony_weight * parsimony).backward()
        opt.step()
        if step % log_every == 0:
            trace.append((step, float(energy.detach())))

    with torch.no_grad():
        final = current()
        final_energy = float(
            ebm.energy_from_satisfaction(ebm.engine.satisfaction(final)).mean()
        )
    trace.append((steps, final_energy))
    return RepairResult(beliefs=final.detach(), energy_trace=trace, steps=steps)
