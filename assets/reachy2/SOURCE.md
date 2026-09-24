# Reachy 2 MuJoCo model (vendored)

Copied from Pollen Robotics' [reachy2_mujoco](https://github.com/pollen-robotics/reachy2_mujoco)
at commit `f6d8284e812d3b96b557e2e844d55bd09d6e3ee6` (2025-07-29). License: Apache 2.0 (see `LICENSE`).

| File here | Original path in reachy2_mujoco |
|---|---|
| `reachy2_asset.xml` | `reachy2_mujoco/description/mjcf/assets/reachy2_asset.xml` |
| `reachy2_body.xml` | `reachy2_mujoco/description/mjcf/objects/reachy2_body.xml` |
| `meshes/reachy2/` | `reachy2_mujoco/description/mjcf/assets/meshes/reachy2/` |
| `../test_scene.xml` | `reachy2_mujoco/description/mjcf/test_scene.xml` (include paths changed only) |

Note: with mujoco 3.14, load scenes with an ABSOLUTE path. A relative path makes
mesh lookup fail ("Error opening file ... .obj"), for the original repo too.
