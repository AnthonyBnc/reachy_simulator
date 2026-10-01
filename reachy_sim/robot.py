"""Constants shared by the whole package: paths, joint names, body names."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
CONFIGS = ROOT / "configs"

# Right arm joints, in reachy2-sdk order
R_ARM = ["r_shoulder_pitch", "r_shoulder_roll", "r_elbow_yaw", "r_elbow_pitch",
         "r_wrist_roll", "r_wrist_pitch", "r_wrist_yaw"]
HAND_BODY = "r_arm_tip"
TORSO_BODY = "torso"
GOAL_BODY = "goal"