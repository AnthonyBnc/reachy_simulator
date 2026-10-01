"""MujocoBackend: the only place that talks to MuJoCo. Everything is in radians."""

import mujoco
import numpy as np

from reachy_sim.robot import ASSETS, GOAL_BODY, HAND_BODY, R_ARM, TORSO_BODY


class MujocoBackend:
    def __init__(self, scene="reach_scene.xml", control_hz=20, settle_time=0.5):
        # Absolute path: MuJoCo 3.14 can't find the meshes with a relative one
        self.model = mujoco.MjModel.from_xml_path(str(ASSETS / scene))
        self.data = mujoco.MjData(self.model)
        self._fk_data = mujoco.MjData(self.model)  # scratch copy for predict_ee()

        # Look everything up by NAME once (motor order != joint order)
        joints = [self._id(mujoco.mjtObj.mjOBJ_JOINT, n) for n in R_ARM]
        self.qadr = self.model.jnt_qposadr[joints]
        self.dadr = self.model.jnt_dofadr[joints]
        self.act = np.array([self._id(mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in R_ARM])
        self.joint_limits = self.model.jnt_range[joints].copy()  # (7, 2) rad
        self.torso = self._id(mujoco.mjtObj.mjOBJ_BODY, TORSO_BODY)
        self.hand = self._id(mujoco.mjtObj.mjOBJ_BODY, HAND_BODY)
        goal = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, GOAL_BODY)
        self.goal_mocap = self.model.body_mocapid[goal] if goal >= 0 else -1

        # 20 Hz control: 25 physics steps of 2 ms per control step
        self.frame_skip = round(1 / (control_hz * self.model.opt.timestep))
        self.dt = self.frame_skip * self.model.opt.timestep
        self.settle_steps = round(settle_time / self.model.opt.timestep)
        self._renderer = None

    def _id(self, kind, name):
        i = mujoco.mj_name2id(self.model, kind, name)
        if i < 0:
            raise ValueError(f"'{name}' not found in the model")
        return i

    def reset(self, q_init=None):
        """Put the arm at q_init, hold it there, and let the robot settle."""
        mujoco.mj_resetData(self.model, self.data)
        q = np.zeros(7) if q_init is None else np.asarray(q_init, dtype=float)
        self.data.qpos[self.qadr] = q
        self.data.ctrl[self.act] = q  # otherwise the motors pull back to 0
        mujoco.mj_forward(self.model, self.data)
        mujoco.mj_step(self.model, self.data, nstep=self.settle_steps)  # tripod settles ~11 mm

    def get_joint_positions(self):
        return self.data.qpos[self.qadr].copy()

    def get_joint_velocities(self):
        return self.data.qvel[self.dadr].copy()

    def _hand_pose(self, data):
        R_t = data.xmat[self.torso].reshape(3, 3)
        R_h = data.xmat[self.hand].reshape(3, 3)
        T = np.eye(4)
        T[:3, :3] = R_t.T @ R_h
        T[:3, 3] = R_t.T @ (data.xpos[self.hand] - data.xpos[self.torso])
        return T

    def get_ee_pose(self):
        """4x4 pose of the hand in the torso frame (same frame as the SDK)."""
        return self._hand_pose(self.data)

    def predict_ee(self, q):
        """Hand position (torso frame) IF the arm were at q. Doesn't touch the real sim."""
        self._fk_data.qpos[:] = self.data.qpos
        self._fk_data.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.model, self._fk_data)
        return self._hand_pose(self._fk_data)[:3, 3]

    def send_joint_targets(self, q):
        self.data.ctrl[self.act] = q

    def set_goal_marker(self, pos_torso):
        if self.goal_mocap < 0:
            return
        R_t = self.data.xmat[self.torso].reshape(3, 3)
        self.data.mocap_pos[self.goal_mocap] = self.data.xpos[self.torso] + R_t @ pos_torso

    def step(self):
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)

    def render(self, width=640, height=480):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=height, width=width)
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(self.model, camera)
        camera.distance, camera.elevation = 1.8, -15
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None