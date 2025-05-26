# evaluate_model.py
# Evaluate a trained RL model in the F1Tenth Gym environment and log metrics.

import os
import gym
import argparse
import numpy as np
from train_models import F110_Wrapper, evaluate_agent, log_metrics
from stable_baselines3 import PPO, SAC, DDPG

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Evaluate a trained RL model and log metrics")
parser.add_argument("--model", required=True, help="Path to the trained model (.zip)")
parser.add_argument("--algo", required=True, choices=["ppo", "sac", "ddpg"], help="Algorithm used")
parser.add_argument("--episodes", type=int, default=5, help="Number of evaluation episodes")
args = parser.parse_args()

# Load arguments
MODEL_PATH = os.path.join("models", args.model + ".zip")
ALGO = args.algo.lower()
EVAL_EPISODES = args.episodes
MAX_STEPS = 50000
TRACK_MAP = "/sim_ws/src/f1tenth_gym_ros/maps/Spielberg_map"
TRACK_EXT = ".png"

# Build and wrap the evaluation environment
raw_env = gym.make(
    "f110_gym:f110-v0",
    map=TRACK_MAP,
    map_ext=TRACK_EXT,
    num_agents=1
)
env = F110_Wrapper(raw_env)

# Load the trained model
ALGO_MAP = {"ppo": PPO, "sac": SAC, "ddpg": DDPG}
ModelCls = ALGO_MAP[ALGO]
model = ModelCls.load(MODEL_PATH, env=env)

# Run evaluation and collect metrics
metrics = evaluate_agent(model, env, EVAL_EPISODES, MAX_STEPS)

# Log evaluation results to CSV
csv_path = os.path.join("models", "metrics.csv")
log_metrics(csv_path, MODEL_PATH, ALGO.upper(), metrics)

# Display evaluation summary
print(f"\nEvaluation over {EVAL_EPISODES} episodes:")
for k, v in metrics.items():
    print(f"  {k:20s}: {v}")
