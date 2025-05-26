import gym
import argparse
import numpy as np
from stable_baselines3 import PPO, SAC, DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback
from train_models import F110_Wrapper

# Get command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--algo", type=str, required=True, help="ppo | sac | ddpg")
parser.add_argument("--model_path", type=str, required=True, help="Path to saved model (.zip)")
parser.add_argument("--timesteps", type=int, default=500000, help="Number of steps to continue training")
args = parser.parse_args()

# Create the simulation environment
TRACK_MAP = "/sim_ws/src/f1tenth_gym_ros/maps/Spielberg_map"
TRACK_EXT = ".png"

env = F110_Wrapper(gym.make(
    "f110_gym:f110-v0",
    map=TRACK_MAP,
    map_ext=TRACK_EXT,
    num_agents=1
), algo=args.algo)

# Load the existing model based on selected algorithm
if args.algo == "ppo":
    model = PPO.load(args.model_path, env=env)
elif args.algo == "sac":
    model = SAC.load(args.model_path, env=env)
elif args.algo == "ddpg":
    model = DDPG.load(args.model_path, env=env)
else:
    raise ValueError("Unsupported algorithm. Use: ppo, sac, or ddpg")

# Save a new checkpoint every 200,000 steps
checkpoint_callback = CheckpointCallback(
    save_freq=600000,
    save_path="./models/",
    name_prefix=f"{args.algo}_7",
    save_replay_buffer=True,
    save_vecnormalize=True
)

# Continue training the model
model.learn(total_timesteps=args.timesteps, callback=checkpoint_callback)

# Save the final version of the model
final_model_path = args.model_path.replace(".zip", "_slow.zip")
model.save(final_model_path)
print(f"Final model saved to: {final_model_path}")
