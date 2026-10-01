import numpy as np

from reachy_sim.safety.shield import SafetyShield

LIMITS = np.radians([[-180, 180], [-180, 0], [-180, 180], [-128.9, 5.7], [-20, 20], [-20, 20], [-90, 90]])
BOX = [[0.15, 0.65], [-0.60, 0.05], [-0.40, 0.25]]
d = np.radians


def make(box=None):
    return SafetyShield(LIMITS, max_delta_deg=5, margin_deg=5, workspace_box=box)


def test_small_change_passes_unchanged():
    q0 = d([-30, -20, 0, -40, 0, 0, 0])
    q, info = make().filter(q0, q0 + d(2))
    assert np.allclose(q, q0 + d(2)) and info["interventions"] == 0


def test_big_change_is_cut_to_5_degrees():
    q0 = d([-30, -20, 0, -40, 0, 0, 0])
    q, info = make().filter(q0, q0 + d(30))
    assert np.allclose(q, q0 + d(5)) and info["reasons"] == ["step_limit"]


def test_joint_limit_is_clipped_with_margin():
    q0 = d([0, -20, 0, -40, 0, 12, 0])            # wrist pitch at 12 deg, soft limit 15
    q, info = make().filter(q0, q0 + d([0, 0, 0, 0, 0, 5, 0]))
    assert np.isclose(q[5], d(15)) and "joint_limit" in info["reasons"]


def test_home_at_the_limit_edge_does_not_force_a_jump():
    q0 = np.zeros(7)                                # shoulder_roll = 0 is outside its soft limit (-5)
    q, info = make().filter(q0, q0)
    assert np.allclose(q, q0) and info["interventions"] == 0


def test_moving_further_away_from_box_is_blocked():
    shield = make(BOX)
    fake_fk = lambda q: np.array([0.4, -0.3, -0.5 + q[0]])   # hand below the box; q[0] moves it up/down
    q0 = np.zeros(7)
    q, info = shield.filter(q0, q0 + d([-5, 0, 0, 0, 0, 0, 0]), predict_ee=fake_fk)  # goes down
    assert np.allclose(q, q0) and info["reasons"] == ["workspace"]


def test_moving_toward_box_from_home_is_allowed():
    shield = make(BOX)
    fake_fk = lambda q: np.array([0.4, -0.3, -0.5 + q[0]])
    q0 = np.zeros(7)
    q, info = shield.filter(q0, q0 + d([5, 0, 0, 0, 0, 0, 0]), predict_ee=fake_fk)    # goes up
    assert np.allclose(q, q0 + d([5, 0, 0, 0, 0, 0, 0])) and info["interventions"] == 0


def test_several_rules_are_counted():
    q0 = d([0, -20, 0, -40, 0, 12, 0])
    q, info = make().filter(q0, q0 + d([0, 0, 0, 0, 0, 30, 0]))   # too big AND past the limit
    assert info["interventions"] == 2