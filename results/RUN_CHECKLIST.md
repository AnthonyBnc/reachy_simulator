# Random-goal policies on new goals: run checklist

Tick each box when the run is done. Dry run first (nothing moves), then live with the E-stop in hand.
Every run is logged to `results/runs.csv` automatically.

| # | Goal | Policy | Dry run | Live | Result / notes |
|---|---|---|---|---|---|
| 1 | A | SAC | [ ] | [ ] | |
| 2 | A | PPO | [ ] | [ ] | |
| 3 | A | TD3 | [ ] | [ ] | |
| 4 | A | DDPG | [ ] | [ ] | |
| 5 | B | SAC | [ ] | [ ] | |
| 6 | B | PPO | [ ] | [ ] | |
| 7 | B | TD3 | [ ] | [ ] | |
| 8 | B | DDPG | [ ] | [ ] | |
| 9 | C | SAC | [ ] | [ ] | |
| 10 | C | PPO | [ ] | [ ] | |
| 11 | C | TD3 | [ ] | [ ] | |
| 12 | C | DDPG | [ ] | [ ] | |
| 13 | D | SAC | [ ] | [ ] | |
| 14 | D | PPO | [ ] | [ ] | |
| 15 | D | TD3 | [ ] | [ ] | |
| 16 | D | DDPG | [ ] | [ ] | |

---

## Goal A: `0.35 -0.30 -0.25` (low, close to the arm)

### 1. Goal A / SAC

- [ ] Dry run
```bash
python scripts/run_real.py --model policies/reach_random_goal/sac/model.zip --no-wrist --goal 0.35 -0.30 -0.25 --speed 0.3
```
- [ ] Live
```bash
python scripts/run_real.py --model policies/reach_random_goal/sac/model.zip --no-wrist --goal 0.35 -0.30 -0.25 --live --speed 0.3
```

### 2. Goal A / PPO

- [ ] Dry run
```bash
python scripts/run_real.py --model policies/reach_random_goal/ppo/model.zip --no-wrist --goal 0.35 -0.30 -0.25 --speed 0.3
```
- [ ] Live
```bash
python scripts/run_real.py --model policies/reach_random_goal/ppo/model.zip --no-wrist --goal 0.35 -0.30 -0.25 --live --speed 0.3
```

### 3. Goal A / TD3

- [ ] Dry run
```bash
python scripts/run_real.py --model policies/reach_random_goal/td3/model.zip --no-wrist --goal 0.35 -0.30 -0.25 --speed 0.3
```
- [ ] Live
```bash
python scripts/run_real.py --model policies/reach_random_goal/td3/model.zip --no-wrist --goal 0.35 -0.30 -0.25 --live --speed 0.3
```

### 4. Goal A / DDPG

- [ ] Dry run
```bash
python scripts/run_real.py --model policies/reach_random_goal/ddpg/model.zip --no-wrist --goal 0.35 -0.30 -0.25 --speed 0.3
```
- [ ] Live
```bash
python scripts/run_real.py --model policies/reach_random_goal/ddpg/model.zip --no-wrist --goal 0.35 -0.30 -0.25 --live --speed 0.3
```

- [ ] Plot goal A
```bash
python scripts/plot_runs.py --goal "0.35 -0.30 -0.25"
```

---

## Goal B: `0.30 -0.45 0.10` (high, out to the side)

### 5. Goal B / SAC

- [ ] Dry run
```bash
python scripts/run_real.py --model policies/reach_random_goal/sac/model.zip --no-wrist --goal 0.30 -0.45 0.10 --speed 0.3
```
- [ ] Live
```bash
python scripts/run_real.py --model policies/reach_random_goal/sac/model.zip --no-wrist --goal 0.30 -0.45 0.10 --live --speed 0.3
```

### 6. Goal B / PPO

- [ ] Dry run
```bash
python scripts/run_real.py --model policies/reach_random_goal/ppo/model.zip --no-wrist --goal 0.30 -0.45 0.10 --speed 0.3
```
- [ ] Live
```bash
python scripts/run_real.py --model policies/reach_random_goal/ppo/model.zip --no-wrist --goal 0.30 -0.45 0.10 --live --speed 0.3
```

### 7. Goal B / TD3

- [ ] Dry run
```bash
python scripts/run_real.py --model policies/reach_random_goal/td3/model.zip --no-wrist --goal 0.30 -0.45 0.10 --speed 0.3
```
- [ ] Live
```bash
python scripts/run_real.py --model policies/reach_random_goal/td3/model.zip --no-wrist --goal 0.30 -0.45 0.10 --live --speed 0.3
```

### 8. Goal B / DDPG

- [ ] Dry run
```bash
python scripts/run_real.py --model policies/reach_random_goal/ddpg/model.zip --no-wrist --goal 0.30 -0.45 0.10 --speed 0.3
```
- [ ] Live
```bash
python scripts/run_real.py --model policies/reach_random_goal/ddpg/model.zip --no-wrist --goal 0.30 -0.45 0.10 --live --speed 0.3
```

- [ ] Plot goal B
```bash
python scripts/plot_runs.py --goal "0.30 -0.45 0.10"
```

---

## Goal C: `0.55 -0.20 -0.15` (far forward)

### 9. Goal C / SAC

- [ ] Dry run
```bash
python scripts/run_real.py --model policies/reach_random_goal/sac/model.zip --no-wrist --goal 0.55 -0.20 -0.15 --speed 0.3
```
- [ ] Live
```bash
python scripts/run_real.py --model policies/reach_random_goal/sac/model.zip --no-wrist --goal 0.55 -0.20 -0.15 --live --speed 0.3
```

### 10. Goal C / PPO

- [ ] Dry run
```bash
python scripts/run_real.py --model policies/reach_random_goal/ppo/model.zip --no-wrist --goal 0.55 -0.20 -0.15 --speed 0.3
```
- [ ] Live
```bash
python scripts/run_real.py --model policies/reach_random_goal/ppo/model.zip --no-wrist --goal 0.55 -0.20 -0.15 --live --speed 0.3
```

### 11. Goal C / TD3

- [ ] Dry run
```bash
python scripts/run_real.py --model policies/reach_random_goal/td3/model.zip --no-wrist --goal 0.55 -0.20 -0.15 --speed 0.3
```
- [ ] Live
```bash
python scripts/run_real.py --model policies/reach_random_goal/td3/model.zip --no-wrist --goal 0.55 -0.20 -0.15 --live --speed 0.3
```

### 12. Goal C / DDPG

- [ ] Dry run
```bash
python scripts/run_real.py --model policies/reach_random_goal/ddpg/model.zip --no-wrist --goal 0.55 -0.20 -0.15 --speed 0.3
```
- [ ] Live
```bash
python scripts/run_real.py --model policies/reach_random_goal/ddpg/model.zip --no-wrist --goal 0.55 -0.20 -0.15 --live --speed 0.3
```

- [ ] Plot goal C
```bash
python scripts/plot_runs.py --goal "0.55 -0.20 -0.15"
```

---

## Goal D: `0.50 -0.40 0.15` (far forward, high, to the side)

### 13. Goal D / SAC

- [ ] Dry run
```bash
python scripts/run_real.py --model policies/reach_random_goal/sac/model.zip --no-wrist --goal 0.50 -0.40 0.15 --speed 0.3
```
- [ ] Live
```bash
python scripts/run_real.py --model policies/reach_random_goal/sac/model.zip --no-wrist --goal 0.50 -0.40 0.15 --live --speed 0.3
```

### 14. Goal D / PPO

- [ ] Dry run
```bash
python scripts/run_real.py --model policies/reach_random_goal/ppo/model.zip --no-wrist --goal 0.50 -0.40 0.15 --speed 0.3
```
- [ ] Live
```bash
python scripts/run_real.py --model policies/reach_random_goal/ppo/model.zip --no-wrist --goal 0.50 -0.40 0.15 --live --speed 0.3
```

### 15. Goal D / TD3

- [ ] Dry run
```bash
python scripts/run_real.py --model policies/reach_random_goal/td3/model.zip --no-wrist --goal 0.50 -0.40 0.15 --speed 0.3
```
- [ ] Live
```bash
python scripts/run_real.py --model policies/reach_random_goal/td3/model.zip --no-wrist --goal 0.50 -0.40 0.15 --live --speed 0.3
```

### 16. Goal D / DDPG

- [ ] Dry run
```bash
python scripts/run_real.py --model policies/reach_random_goal/ddpg/model.zip --no-wrist --goal 0.50 -0.40 0.15 --speed 0.3
```
- [ ] Live
```bash
python scripts/run_real.py --model policies/reach_random_goal/ddpg/model.zip --no-wrist --goal 0.50 -0.40 0.15 --live --speed 0.3
```

- [ ] Plot goal D
```bash
python scripts/plot_runs.py --goal "0.50 -0.40 0.15"
```

---

## When all goals are done

```bash
python scripts/compare_runs.py
```
