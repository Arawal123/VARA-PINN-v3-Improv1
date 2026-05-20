from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.controllers import VARAController
from src.controllers.local_controller import LocalControllerConfig, LocalVARAController
from src.diagnostics import PatchGrid, WeakRegion


def test_controller_outputs_actions():
    grid = PatchGrid(bounds=(0, 1, 0, 1), nx_patches=2, ny_patches=2)
    wr = WeakRegion(0, "p_error_mean_centered", 0.9, 0.7, grid.get_patch(0).bounds, "pressure_dominant")
    controller = VARAController({"p": 1.0, "u": 1.0})
    record = controller.step(0, [wr], np.ones((1, 4)), ["p_error_mean_centered"])
    assert record["actions"]
    assert "p" in controller.state.local_weights


def test_local_controller_state_rollback():
    grid = PatchGrid(bounds=(0, 1, 0, 1), nx_patches=2, ny_patches=2)
    wr = WeakRegion(1, "p_error_mean_centered", 1.0, 0.8, grid.get_patch(1).bounds, "pressure_dominant")
    controller = LocalVARAController({"p": 1.0}, LocalControllerConfig())
    snapshot = controller.snapshot()
    actions = controller.propose([wr])
    controller.apply(actions)
    assert controller.state.local_weights
    controller.rollback(snapshot)
    controller.mark_rejected(actions[0], {"target_local_improvement": 0.0, "max_collateral_damage": 0.0})
    assert controller.state.local_weights == {}
    pair_state = controller.pair_state[("p_error_mean_centered", 1)]
    assert pair_state.rejected_count == 1
    assert pair_state.strength < controller.config.initial_strength


def test_local_constrained_accept_reject_rule():
    controller = LocalVARAController({"p": 1.0}, LocalControllerConfig())
    before = {"u_rel_l2": 1.0, "v_rel_l2": 1.0, "p_rel_l2_centered": 1.0, "omega_rel_l2": 1.0, "pde_residual_mean": 1.0}
    after_good = {"u_rel_l2": 0.99, "v_rel_l2": 1.0, "p_rel_l2_centered": 0.99, "omega_rel_l2": 1.0, "pde_residual_mean": 0.99}
    raw_before = np.array([[1.0]])
    raw_after = np.array([[0.9]])
    wr = WeakRegion(0, "p_error_mean_centered", 1.0, 0.8, (0, 1, 0, 1, None, None), "pressure_dominant")
    actions = controller.propose([wr])
    accepted, _ = controller.evaluate_acceptance(actions, raw_before, raw_after, ["p_error_mean_centered"], before, after_good, True)
    assert accepted

    after_bad = {"u_rel_l2": 1.0, "v_rel_l2": 1.0, "p_rel_l2_centered": 1.2, "omega_rel_l2": 1.0, "pde_residual_mean": 1.0}
    accepted, metrics = controller.evaluate_acceptance(actions, raw_before, raw_after, ["p_error_mean_centered"], before, after_bad, True)
    assert not accepted
    assert metrics["pressure_collateral_damage"] > 0.05


def test_local_pair_strength_damps_after_rejection():
    grid = PatchGrid(bounds=(0, 1, 0, 1), nx_patches=2, ny_patches=2)
    wr = WeakRegion(2, "omega_error", 1.0, 1.0, grid.get_patch(2).bounds, "omega_dominant")
    controller = LocalVARAController({"omega": 1.0}, LocalControllerConfig(initial_strength=1.0, damping_factor=0.5))
    first = controller.propose([wr])[0]
    controller.mark_rejected(first, {"target_local_improvement": 0.0, "max_collateral_damage": 0.0})
    second = controller.propose([wr])[0]
    assert second.strength < first.strength
