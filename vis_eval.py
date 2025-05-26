# visualize_with_progress.py
# Evaluate and visualise a trained RL model in the F1Tenth Gym environment.

import os
import gym
import numpy as np
import argparse
from tqdm import tqdm
from train_models import F110_Wrapper
from stable_baselines3 import PPO, SAC, DDPG

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Visualise a trained RL model in F1Tenth Gym")
parser.add_argument("--model", required=True, help="Path to the trained model file (.zip)")
parser.add_argument("--algo", required=True, choices=["ppo", "sac", "ddpg"], help="RL algorithm used")
parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run")
parser.add_argument("--render", type=int, default=20, help="Render every N steps")
args = parser.parse_args()

# Load arguments
MODEL_PATH = os.path.join("models", args.model + ".zip")
ALGO = args.algo.lower()
EPISODES = args.episodes
RENDER_SKIP = args.render
MAX_STEPS = 50000
TRACK_MAP = "/sim_ws/src/f1tenth_gym_ros/maps/Spielberg_map"
TRACK_EXT = ".png"

# Create and wrap the simulation environment
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
print(f"\n Visualising {EPISODES} episodes (rendering every {RENDER_SKIP} steps)\n")

# Loop over multiple test episodes
for ep in range(1, EPISODES + 1):
    print(f"=== Episode {ep}/{EPISODES} ===")
    obs = env.reset(seed=ep)  # Reset environment with seed
    done = False

    # Initialise statistics for this episode
    total_reward = 0.0
    collisions = 0
    laps = 0
    lap_times = []
    step = 0
    pbar = tqdm(total=MAX_STEPS, desc=f"Ep{ep}", ncols=60)  # Progress bar
    
    # Step through the episode until done or step limit reached
    while not done and step < MAX_STEPS:
        action, _ = model.predict(obs, deterministic=True)  
        obs, reward, done, info = env.step(action)          

        # Update statistics
        total_reward += reward
        if info.get("collision", False):
            collisions += 1
        if "lap_time" in info:
            lap_times.append(info["lap_time"])
        laps = info.get("laps", laps)  # Always update laps if available

        step += 1
        pbar.update(1)

        # Render frame every RENDER_SKIP steps
        if step % RENDER_SKIP == 0:
            env.render()

    pbar.close()

    # Calculate average lap time
    # Make sure to get final lap count and lap time
    final_laps = info.get("laps", laps)
    if "lap_time" in info and info["lap_time"] not in lap_times:
        lap_times.append(info["lap_time"])
    if "lap_time" in info:
        print(f"Lap {info['laps']} completed at step {step}, lap time: {info['lap_time']}")
    if info.get("collision", False):
        print(f"Crash at step {step}")


    mean_lap_time = float(np.mean(lap_times)) if lap_times else None
    print(f" Episode {ep} finished: Reward={total_reward:.1f}, "
        f"Steps={step}, Collisions={collisions}, "
        f"Laps={final_laps}, Mean Lap Time={mean_lap_time}\n")


env.close()

# example: python vis_eval.py --model models/ppo_model_031.zip --algo ppo --episodes 3 --render_skip 10
