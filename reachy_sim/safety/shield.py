"""SafetyShield: filters every joint target before it reaches the robot.

Pure numpy (no MuJoCo), so the exact same code will run on the real robot.
"""

import numpy as np


class SafetyShield:
    def __init__(self, joint_limits, max_delta_deg=5.0, margin_deg=5.0, workspace_box=None, safe_range=None):
        limits = np.asarray(joint_limits, dtype=float)  # (7, 2) rad
        margin = np.radians(margin_deg)
        self.lo = limits[:, 0] + margin
        self.hi = limits[:, 1] - margin
        if safe_range is not None:
            safe = np.asarray(safe_range, dtype=float)
            self.lo = np.maximum(self.lo, safe[:, 0])
            self.hi = np.minimum(self.hi, safe[:, 1])
        self.max_delta = np.radians(max_delta_deg)
        self.box = None if workspace_box is None else np.asarray(workspace_box, dtype=float)  # (3, 2)

    @classmethod
    def from_config(cls, cfg, joint_limits):
        box = cfg.get("workspace_box_m")
        safe = cfg.get("joint_limits_deg") #SDK joint order 
        return cls(
            joint_limits,
            max_delta_deg=cfg["max_delta_per_step_deg"],
            margin_deg=cfg["joint_margin_deg"],
            workspace_box=None if box is None else [box["x"], box["y"], box["z"]],
            safe_range=None if safe is None else np.radians(list(safe.values())),
        )

    def distance_outside_box(self, p):
        """0 inside the box, otherwise the distance to it (m)."""
        below = self.box[:, 0] - p
        above = p - self.box[:, 1]
        return float(np.linalg.norm(np.maximum(0.0, np.maximum(below, above))))

    def filter(self, q_current, q_target, predict_ee=None):
        """Return (safe target, {"interventions": n, "reasons": [...]})."""
        q_current = np.asarray(q_current, dtype=float)
        reasons = []

        # 1. Step limit: each joint changes by at most max_delta
        delta = np.asarray(q_target, dtype=float) - q_current
        clipped = np.clip(delta, -self.max_delta, self.max_delta)
        if not np.allclose(clipped, delta):
            reasons.append("step_limit")
        q = q_current + clipped

        # 2. Joint limits: never move FURTHER outside the soft limits
        #    (home is at the edge for shoulder_roll, so "must be inside" would force a jump)
        lo = np.minimum(self.lo, q_current)
        hi = np.maximum(self.hi, q_current)
        q_clipped = np.clip(q, lo, hi)
        if not np.allclose(q_clipped, q):
            reasons.append("joint_limit")
        q = q_clipped

        # 3. Workspace box: never move the hand FURTHER away from the box
        #    (home starts below the box, so "must be inside" would freeze the arm)
        if self.box is not None and predict_ee is not None:
            if self.distance_outside_box(predict_ee(q)) > self.distance_outside_box(predict_ee(q_current)) + 1e-6:
                q = q_current.copy()
                reasons.append("workspace")

        return q, {"interventions": len(reasons), "reasons": reasons}