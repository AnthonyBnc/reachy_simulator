import numpy as np

from reachy_sim.backends.mujoco_backend import MujocoBackend


def test_reset_puts_arm_at_q_init():
    b = MujocoBackend()
    q = np.radians([-20, -10, 0, -30, 0, 0, 0])
    b.reset(q)
    assert np.all(np.abs(np.degrees(b.get_joint_positions() - q)) < 2)


def test_targets_are_reached():
    b = MujocoBackend()
    b.reset()
    q = np.radians([-45, -15, 0, -80, 0, 0, 20])
    b.send_joint_targets(q)
    for _ in range(20):  # 1 s
        b.step()
    assert np.all(np.abs(np.degrees(b.get_joint_positions() - q)) < 2)


def test_hand_position_at_zero_matches_compare_fk():
    b = MujocoBackend()
    b.reset()
    assert np.allclose(b.get_ee_pose()[:3, 3], [0.030, -0.368, -0.638], atol=0.005)


def test_predict_ee_matches_real_pose_and_does_not_move_robot():
    b = MujocoBackend()
    b.reset()
    before = b.data.qpos.copy()
    assert np.allclose(b.predict_ee(b.get_joint_positions()), b.get_ee_pose()[:3, 3], atol=1e-6)
    b.predict_ee(np.radians([-60, -20, 0, -90, 0, 0, 0]))
    assert np.array_equal(b.data.qpos, before)


def test_torso_does_not_move_after_settling():
    b = MujocoBackend()
    b.reset()
    p0 = b.data.xpos[b.torso].copy()
    b.send_joint_targets(np.radians([-60, -20, 0, -90, 0, 0, 40]))
    for _ in range(40):
        b.step()
    assert np.linalg.norm(b.data.xpos[b.torso] - p0) < 0.001