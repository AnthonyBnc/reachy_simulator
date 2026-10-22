import json 
from pathlib import Path 

from stable_baselines3 import DDPG, PPO, SAC, TD3

ALGOS = {
    "sac": SAC,
    "td3": TD3,
    "ppo": PPO,
    "ddpg": DDPG,
}

def load_model(path):

    """
    Load a saved policy with the algorithm it was trained with.
    """

    path = Path(path)
    for folder in (path.parent, path.parent.parent):
        args_file = folder / "args.json"
        if args_file.exists():
            return ALGOS[json.loads(args_file.read_text()).get("algo", "sac")].load(path)
    return SAC.load(path)