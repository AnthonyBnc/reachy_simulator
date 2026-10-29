# Trained policies

Ready-to-use policies for **Task 1 (Reach)**: Reachy 2's right hand moves to a 3D goal and holds it. All are 4-joint (`--no-wrist`), 500,000 steps, seed 0. To train your own, see [docs/TASK1_REACH.md](../docs/TASK1_REACH.md).

| Folder | Goals | Use it for |
|---|---|---|
| [reach_random_goal/](reach_random_goal/) | any point in the workspace | **Most uses**: any goal, the follow-the-ball demo, the real robot |
| [reach_fixed_goal/](reach_fixed_goal/) | only (0.40, −0.20, 0.00) | That one point; always pass `--goal 0.40 -0.20 0.00` |

## Results ("reached and held", 100 unseen test episodes)

| Algorithm | Random goals | Fixed goal |
|---|---|---|
| SAC | **100%**, 0.9 cm | 100%, 0.3 cm |
| TD3 | 98%, 1.5 cm | 100%, 0.8 cm |
| PPO | 96%, 1.8 cm | 100%, 0.3 cm |
| DDPG | 91%, 2.0 cm | 100%, 1.5 cm |

The fixed goal is solved by every algorithm; random goals separate them. **SAC is the recommended policy**: the best result, and it learns fastest (~90% after ~70k steps on random goals).

```bash
python scripts/evaluate.py --model policies/reach_random_goal/sac/model.zip --no-wrist
```
