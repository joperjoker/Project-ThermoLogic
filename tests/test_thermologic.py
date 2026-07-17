"""Unit tests for Project ThermoLogic.

Runnable with either::

    python -m unittest discover -s tests
    pytest tests/

Covers the correctness-critical core: fuzzy t-norm operators and residuated
implications at their Boolean corners, emergent modus ponens, the exponential
energy, forward-chaining ground truth, test-time repair convergence, and the
high-level :class:`LogicEnergy` API.
"""

from __future__ import annotations

import math
import unittest

import torch

from thermologic import (
    DifferentiableLogicEngine,
    EnergyBasedModel,
    ImplicationRule,
    KnowledgeBase,
    Literal,
    LogicEnergy,
    NeuralProposer,
    TNorm,
    forward_chaining,
    implies,
    repair_beliefs,
)


def _engine(tnorm: TNorm) -> DifferentiableLogicEngine:
    """A→B engine over 2 atoms for the given t-norm."""
    kb = KnowledgeBase(
        atom_names=("A", "B"),
        rules=(ImplicationRule((Literal(0),), Literal(1), "A→B"),),
        cause_atoms=(0,),
        derived_atoms=(1,),
    )
    return DifferentiableLogicEngine(kb, tnorm)


class TestTNormConjunction(unittest.TestCase):
    def test_boolean_corners(self) -> None:
        for tnorm in TNorm:
            eng = _engine(tnorm)
            conj = eng._conj[tnorm]
            a = torch.tensor([1.0, 1.0, 0.0, 0.0])
            b = torch.tensor([1.0, 0.0, 1.0, 0.0])
            out = conj(a, b)
            self.assertTrue(torch.allclose(out, torch.tensor([1.0, 0.0, 0.0, 0.0])),
                            msg=f"{tnorm} conjunction wrong at corners: {out}")

    def test_conjunction_ranges(self) -> None:
        eng_p = _engine(TNorm.PRODUCT)
        self.assertAlmostEqual(float(eng_p._conj_product(torch.tensor(0.5), torch.tensor(0.5))), 0.25)
        eng_l = _engine(TNorm.LUKASIEWICZ)
        self.assertAlmostEqual(float(eng_l._conj_lukasiewicz(torch.tensor(0.4), torch.tensor(0.4))), 0.0)
        eng_g = _engine(TNorm.GODEL)
        self.assertAlmostEqual(float(eng_g._conj_godel(torch.tensor(0.3), torch.tensor(0.7))), 0.3)


class TestResiduum(unittest.TestCase):
    def test_residuum_boolean_corners(self) -> None:
        # I(1,0)=0 (only violating case), I(1,1)=I(0,0)=I(0,1)=1
        for tnorm in TNorm:
            eng = _engine(tnorm)
            impl = eng._impl[tnorm]
            a = torch.tensor([1.0, 1.0, 0.0, 0.0])
            c = torch.tensor([0.0, 1.0, 0.0, 1.0])
            out = impl(a, c)
            self.assertTrue(torch.allclose(out, torch.tensor([0.0, 1.0, 1.0, 1.0])),
                            msg=f"{tnorm} residuum wrong: {out}")

    def test_residuum_monotone_in_consequent(self) -> None:
        # With antecedent true, satisfaction increases with the consequent belief.
        for tnorm in TNorm:
            eng = _engine(tnorm)
            impl = eng._impl[tnorm]
            a = torch.ones(5)
            c = torch.linspace(0, 1, 5)
            out = impl(a, c)
            self.assertTrue(torch.all(out[1:] >= out[:-1] - 1e-6),
                            msg=f"{tnorm} residuum not monotone: {out}")

    def test_product_residuum_no_blowup(self) -> None:
        # c/a must be guarded near a→0.
        eng = _engine(TNorm.PRODUCT)
        out = eng._residuum_product(torch.tensor([1e-9]), torch.tensor([1.0]))
        self.assertTrue(torch.isfinite(out).all())
        self.assertLessEqual(float(out), 1.0 + 1e-6)


class TestModusPonens(unittest.TestCase):
    def test_gradient_derives_consequent(self) -> None:
        # A=1, rule A→B: minimizing energy must push B up (∂E/∂B < 0 at B<1).
        eng = _engine(TNorm.LUKASIEWICZ)
        ebm = EnergyBasedModel(NeuralProposer(1, 2), eng, beta=4.0)
        probs = torch.tensor([[1.0, 0.3]], requires_grad=True)
        energy = ebm.energy_from_satisfaction(eng.satisfaction(probs)).sum()
        energy.backward()
        grad_B = float(probs.grad[0, 1])
        self.assertLess(grad_B, 0.0, "energy gradient should push consequent B upward")


class TestNegation(unittest.TestCase):
    def test_literal_truth(self) -> None:
        probs = torch.tensor([[0.2, 0.9]])
        self.assertAlmostEqual(float(Literal(0, False).truth(probs)), 0.2)
        self.assertAlmostEqual(float(Literal(0, True).truth(probs)), 0.8)


class TestEnergy(unittest.TestCase):
    def test_valid_zero_contradiction_positive(self) -> None:
        eng = _engine(TNorm.LUKASIEWICZ)
        ebm = EnergyBasedModel(NeuralProposer(1, 2), eng, beta=4.0)
        valid = torch.tensor([[1.0, 1.0]])       # A→B satisfied
        contra = torch.tensor([[1.0, 0.0]])      # A→B violated
        e_valid = float(ebm.energy_from_satisfaction(eng.satisfaction(valid)))
        e_contra = float(ebm.energy_from_satisfaction(eng.satisfaction(contra)))
        self.assertAlmostEqual(e_valid, 0.0, places=5)
        self.assertAlmostEqual(e_contra, math.expm1(4.0), places=4)  # exp(β)-1


class TestForwardChaining(unittest.TestCase):
    def test_minimal_model(self) -> None:
        kb = KnowledgeBase(
            atom_names=("Rain", "Cloudy", "Wet", "Sprinkler", "Slippery"),
            rules=(
                ImplicationRule((Literal(0),), Literal(1), "r"),
                ImplicationRule((Literal(0),), Literal(2), "r"),
                ImplicationRule((Literal(3),), Literal(2), "r"),
                ImplicationRule((Literal(2),), Literal(4), "r"),
            ),
            cause_atoms=(0, 3),
            derived_atoms=(1, 2, 4),
        )
        # Rain only -> Cloudy, Wet, Slippery all true
        self.assertEqual(forward_chaining(kb, [True, False]), [True, True, True, False, True])
        # Sprinkler only -> Wet, Slippery true; Cloudy false
        self.assertEqual(forward_chaining(kb, [False, True]), [False, False, True, True, True])
        # Neither -> all false
        self.assertEqual(forward_chaining(kb, [False, False]), [False, False, False, False, False])


class TestValidation(unittest.TestCase):
    def test_bad_atom_index_raises(self) -> None:
        with self.assertRaises(ValueError):
            KnowledgeBase(
                atom_names=("A", "B"),
                rules=(ImplicationRule((Literal(0),), Literal(5), "bad"),),
            )

    def test_empty_kb_raises(self) -> None:
        with self.assertRaises(ValueError):
            KnowledgeBase(atom_names=("A",), rules=())

    def test_proposer_bad_input_shape(self) -> None:
        p = NeuralProposer(4, 3)
        with self.assertRaises(ValueError):
            p(torch.zeros(2, 7))

    def test_engine_out_of_range_probs(self) -> None:
        eng = _engine(TNorm.LUKASIEWICZ)
        with self.assertRaises(ValueError):
            eng.satisfaction(torch.tensor([[1.5, 0.2]]))


class TestRepair(unittest.TestCase):
    def test_repair_reduces_energy(self) -> None:
        eng = _engine(TNorm.LUKASIEWICZ)
        ebm = EnergyBasedModel(NeuralProposer(1, 2), eng, beta=4.0)
        contra = torch.tensor([[1.0, 0.0]])                       # A→B violated
        fixed = torch.tensor([1.0, 0.0])                          # hold A fixed
        e_before = float(ebm.energy_from_satisfaction(eng.satisfaction(contra)).mean())
        res = repair_beliefs(ebm, contra, fixed, steps=150, lr=0.3,
                             parsimony_weight=0.0, optimizer="adam")
        # Energy drops by orders of magnitude and the consequent is derived true.
        self.assertLess(res.energy_trace[-1][1], 0.05)
        self.assertLess(res.energy_trace[-1][1], e_before / 100.0)
        self.assertGreater(float(res.beliefs[0, 1]), 0.9)         # B derived to true

    def test_sgd_optimizer_supported(self) -> None:
        eng = _engine(TNorm.LUKASIEWICZ)
        ebm = EnergyBasedModel(NeuralProposer(1, 2), eng, beta=4.0)
        res = repair_beliefs(ebm, torch.tensor([[1.0, 0.02]]), torch.tensor([1.0, 0.0]),
                             steps=60, lr=5.0, optimizer="sgd")
        self.assertLess(res.energy_trace[-1][1], 1e-1)


class TestLogicEnergyAPI(unittest.TestCase):
    def setUp(self) -> None:
        self.atoms = ["plan_free", "seats_gt_1", "sso_enabled", "plan_enterprise"]
        self.guard = LogicEnergy(
            rules=[
                implies(["plan_free"], "seats_gt_1", negate_consequent=True, name="free⇒≤1seat"),
                implies(["sso_enabled"], "plan_enterprise", name="sso⇒enterprise"),
            ],
            atom_names=self.atoms,
        )

    def test_score_and_flags(self) -> None:
        bad = torch.tensor([[0.9, 0.9, 0.9, 0.1]])   # free+many seats+sso, not enterprise
        self.assertGreater(float(self.guard.score(bad)), 0.0)
        self.assertFalse(bool(self.guard.is_consistent(bad)))
        self.assertEqual(set(self.guard.violations(bad)[0]), {"free⇒≤1seat", "sso⇒enterprise"})

    def test_repair_yields_consistent(self) -> None:
        bad = torch.tensor([[0.9, 0.9, 0.9, 0.1]])
        fixed = self.guard.repair(bad, budget=120, lr=0.3)
        self.assertTrue(bool(self.guard.is_consistent(fixed, tol=1e-2)))

    def test_accepts_1d_and_names(self) -> None:
        self.assertEqual(self.guard.score(torch.tensor([0.0, 0.0, 0.0, 0.0])).shape, (1,))
        # repair honouring a fixed atom keeps it unchanged
        out = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
        rep = self.guard.repair(out, fixed=["sso_enabled"], budget=80)
        self.assertGreater(float(rep[0, self.atoms.index("sso_enabled")]), 0.5)

    def test_snap_returns_crisp(self) -> None:
        bad = torch.tensor([[0.9, 0.9, 0.9, 0.1]])
        rep = self.guard.repair(bad, budget=120, snap=True)
        # every value is exactly 0 or 1
        self.assertTrue(bool(((rep == 0) | (rep == 1)).all()))
        self.assertTrue(bool(self.guard.is_consistent(rep, crisp=True)))

    def test_verify_reaches_valid_from_tiny_budget(self) -> None:
        # A deliberately tiny starting budget must still yield a crisp-valid
        # result thanks to verify=True auto-escalation.
        bad = torch.tensor([[0.9, 0.9, 0.9, 0.1]])
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # fail if repair gives up
            rep = self.guard.repair(bad, budget=1, snap=True, verify=True, max_budget=400)
        self.assertTrue(bool(self.guard.is_consistent(rep, crisp=True)))

    def test_crisp_vs_soft_consistency(self) -> None:
        valid = torch.tensor([[0.0, 1.0, 0.0, 1.0]])   # enterprise, no sso — valid
        self.assertTrue(bool(self.guard.is_consistent(valid, crisp=True)))
        self.assertTrue(bool(self.guard.is_consistent(valid)))

    def test_json_round_trip(self) -> None:
        d = self.guard.to_dict()
        clone = LogicEnergy.from_dict(d)
        self.assertEqual(clone.atom_names, self.guard.atom_names)
        probe = torch.tensor([[0.9, 0.9, 0.9, 0.1]])
        self.assertAlmostEqual(float(clone.score(probe)), float(self.guard.score(probe)), places=5)
        self.assertEqual(clone.violations(probe), self.guard.violations(probe))

    def test_save_load(self) -> None:
        import os, tempfile
        path = os.path.join(tempfile.gettempdir(), "thermologic_rules_test.json")
        self.guard.save(path)
        clone = LogicEnergy.load(path)
        self.assertEqual(clone.to_dict(), self.guard.to_dict())
        os.remove(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
