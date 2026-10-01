from gymnasium.envs.registration import register

register(id="ReachyReach-v0", entry_point="reachy_sim.envs.reach_env:ReachEnv")