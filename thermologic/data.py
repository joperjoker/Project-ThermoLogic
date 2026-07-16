"""Native synthetic-data generation with world-level train/test holdout.

The benchmark's ground truth is the set of logically-consistent *minimal-model
worlds* implied by a :class:`~logic_engine.KnowledgeBase`. To test genuine
generalization (rather than memorization of a handful of patterns), we split the
distinct worlds themselves into disjoint train/test sets, so evaluation happens
on premise combinations the network has **never** seen during training.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
from torch import Tensor

from thermologic.logic_engine import (
    DifferentiableLogicEngine,
    KnowledgeBase,
    forward_chaining,
)

__all__ = ["Dataset", "WorldSplit", "enumerate_worlds", "split_worlds", "sample_from_worlds"]

World = Tuple[int, ...]


@dataclass
class Dataset:
    """A batch of noisy inputs and crisp ground-truth targets.

    Attributes
    ----------
    inputs:
        ``(n, input_dim)`` noisy encodings of the cause atoms.
    targets:
        ``(n, num_atoms)`` crisp minimal-model truth assignments.
    input_dim:
        Dimensionality of ``inputs``.
    """

    inputs: Tensor
    targets: Tensor
    input_dim: int


@dataclass
class WorldSplit:
    """A disjoint partition of the distinct worlds into train and test sets."""

    train_worlds: List[World]
    test_worlds: List[World]
    all_worlds: List[World]


def enumerate_worlds(kb: KnowledgeBase, engine: DifferentiableLogicEngine) -> List[World]:
    """Enumerate every distinct, logically-consistent minimal-model world.

    For each Boolean cause assignment we compute the minimal model by forward
    chaining and keep it only if the resulting crisp world satisfies *every* rule
    (including negative constraints such as ``Rain -> not Sprinkler``), i.e. it
    has zero energy. Duplicate worlds (different causes, same closure) collapse.

    Parameters
    ----------
    kb:
        The knowledge base.
    engine:
        A logic engine over ``kb`` used to test crisp consistency.

    Returns
    -------
    list of tuple
        Distinct valid worlds, each a ``num_atoms``-length tuple of ``0/1``.
    """
    worlds: List[World] = []
    seen: set[World] = set()
    for cbits in itertools.product([0, 1], repeat=len(kb.cause_atoms)):
        model = forward_chaining(kb, [bool(b) for b in cbits])
        world: World = tuple(int(b) for b in model)
        if world in seen:
            continue
        sat = engine.satisfaction(torch.tensor([[float(b) for b in world]]))
        if bool((sat > 0.999).all()):
            seen.add(world)
            worlds.append(world)
    return worlds


def split_worlds(worlds: Sequence[World], holdout: int, seed: int) -> WorldSplit:
    """Randomly hold out ``holdout`` worlds for testing, rest for training.

    Raises
    ------
    ValueError
        If ``holdout`` is not strictly between ``0`` and ``len(worlds)``.
    """
    if not (0 < holdout < len(worlds)):
        raise ValueError(
            f"holdout must be in (0, {len(worlds)}); got {holdout}"
        )
    order = list(range(len(worlds)))
    random.Random(seed).shuffle(order)
    test = [tuple(worlds[i]) for i in order[:holdout]]
    train = [tuple(worlds[i]) for i in order[holdout:]]
    return WorldSplit(train_worlds=train, test_worlds=test, all_worlds=list(map(tuple, worlds)))


def sample_from_worlds(
    kb: KnowledgeBase,
    worlds: Sequence[World],
    num_samples: int,
    noise_std: float,
    noise_dims: int,
    generator: torch.Generator,
) -> Dataset:
    """Draw noisy training/eval instances from a given set of worlds.

    Each sample picks a world uniformly at random; the input is a Gaussian-noised
    encoding of that world's *cause* bits, padded with ``noise_dims`` pure-noise
    distractor features. The target is the full crisp world. The network must
    therefore recover the causes from noise and infer the derived atoms.

    Parameters
    ----------
    kb:
        The knowledge base (defines cause atoms and atom count).
    worlds:
        The pool of worlds to sample from (e.g. only training worlds).
    num_samples:
        Number of instances to generate.
    noise_std:
        Std. dev. of Gaussian noise added to the cause bits.
    noise_dims:
        Number of pure-noise distractor input features.
    generator:
        Seeded RNG for reproducibility.

    Returns
    -------
    Dataset
        Inputs of shape ``(num_samples, len(cause_atoms) + noise_dims)`` and
        targets of shape ``(num_samples, num_atoms)``.
    """
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if noise_std < 0.0:
        raise ValueError("noise_std must be non-negative")
    if noise_dims < 0:
        raise ValueError("noise_dims must be non-negative")
    if len(worlds) == 0:
        raise ValueError("worlds pool must be non-empty")

    num_causes = len(kb.cause_atoms)
    input_dim = num_causes + noise_dims
    inputs = torch.empty((num_samples, input_dim))
    targets = torch.empty((num_samples, kb.num_atoms))
    pool = torch.tensor([list(w) for w in worlds], dtype=torch.float32)

    for i in range(num_samples):
        w_idx = int(torch.randint(len(worlds), (1,), generator=generator))
        world = pool[w_idx]
        targets[i] = world
        cause_vec = world[list(kb.cause_atoms)]
        noisy = cause_vec + noise_std * torch.randn(num_causes, generator=generator)
        distract = (
            torch.randn(noise_dims, generator=generator) if noise_dims else torch.empty(0)
        )
        inputs[i] = torch.cat([noisy, distract])

    return Dataset(inputs=inputs, targets=targets, input_dim=input_dim)
