"""ReachyReach-v0: move the Reachy 2 right hand to a random 3D goal."""

import gymnasium as gym
import numpy as np
import yaml
from gymnasium import spaces

from reachy_sim.backends.mujoco_backend import MujocoBackend
from reachy_sim.robot import CONFIGS
from reachy_sim.safety.shield import SafetyShield


class ReachEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, render_mode=None, reward_type="dense", max_steps=100, success_threshold=0.03):
        cfg = yaml.safe_load((CONFIGS / "safety" / "default.yaml").read_text())
        self.backend = MujocoBackend(control_hz=20)
        self.shield = SafetyShield.from_config(cfg, self.backend.joint_limits)
        self.max_delta = np.radians(cfg["max_delta_per_step_deg"])
        self.render_mode = render_mode
        self.reward_type = reward_type
        self.max_steps = max_steps
        self.success_threshold = success_threshold

        # What the policy can DO: a change for each of the 7 joints, scaled to +-5 deg
        self.action_space = spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32)
        # What the policy SEES (Dict so that HER works later)
        self.observation_space = spaces.Dict({
            "observation": spaces.Box(-np.inf, np.inf, shape=(24,), dtype=np.float32),  # q, q_dot, hand, prev action
            "achieved_goal": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),  # hand position
            "desired_goal": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),   # goal position
        })

    # ---------- helpers ----------
    def _sample_goal(self):
        """A goal the hand can actually reach: random arm pose -> hand position, kept if inside the box."""
        lo, hi = self.shield.lo, self.shield.hi
        for _ in range(200):
            p = self.backend.predict_ee(self.np_random.uniform(lo, hi))
            if self.shield.distance_outside_box(p) == 0.0:
                return p
        return self.shield.box.mean(axis=1)  # fallback: center of the box

    def _get_obs(self):
        hand = self.backend.get_ee_pose()[:3, 3]
        observation = np.concatenate([
            self.backend.get_joint_positions(),
            self.backend.get_joint_velocities(),
            hand,
            self.prev_action,
        ])
        return {
            "observation": observation.astype(np.float32),
            "achieved_goal": hand.astype(np.float32),
            "desired_goal": self.goal.astype(np.float32),
        }

    # ---------- Gymnasium API ----------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)  # same seed -> same episode
        lo, hi = self.backend.joint_limits.T
        q0 = np.clip(self.np_random.uniform(-1, 1, 7) * np.radians(3), lo, hi)  # home +- 3 deg
        self.backend.reset(q0)
        self.q_cmd = q0.copy()  # last commanded target (NOT the measured angle, or the arm sags)
        self.goal = self._sample_goal()
        self.backend.set_goal_marker(self.goal)
        self.prev_action = np.zeros(7)
        self.steps = 0
        return self._get_obs(), {}

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)

        # action -> target -> safety shield -> robot
        target = self.q_cmd + a * self.max_delta
        self.q_cmd, shield_info = self.shield.filter(self.q_cmd, target, predict_ee=self.backend.predict_ee)
        self.backend.send_joint_targets(self.q_cmd)
        self.backend.step()  # 50 ms of physics

        obs = self._get_obs()
        distance = float(np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"]))
        n = shield_info["interventions"]
        if self.reward_type == "dense":
            reward = (-distance
                      - 0.01 * float(np.sum(a ** 2))                       # small moves
                      - 0.01 * float(np.sum((a - self.prev_action) ** 2))  # smooth moves
                      - 0.1 * n)                                           # respect the shield
        else:
            reward = float(self.compute_reward(obs["achieved_goal"], obs["desired_goal"], {}))

        self.prev_action = a
        self.steps += 1
        info = {"is_success": distance < self.success_threshold, "distance": distance,
                "shield_interventions": n, "shield_reasons": shield_info["reasons"]}
        return obs, reward, False, self.steps >= self.max_steps, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        """Vectorised so HER can recompute rewards for imagined goals."""
        d = np.linalg.norm(np.asarray(achieved_goal) - np.asarray(desired_goal), axis=-1)
        if self.reward_type == "sparse":
            return -(d > self.success_threshold).astype(np.float32)
        return -d

    def render(self):
        if self.render_mode == "rgb_array":
            return self.backend.render()

    def close(self):
        self.backend.close()