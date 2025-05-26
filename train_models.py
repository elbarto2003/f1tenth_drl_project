import gym
import numpy as np
import argparse
import os
import re
import csv
import pandas as pd
from stable_baselines3 import PPO, SAC, DDPG
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor
from gym import spaces
import math
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import CheckpointCallback

# train_models.py
# ----------------
# This script trains a reinforcement learning model (PPO, SAC, or DDPG)
# using the F1TENTH Gym environment. It includes:
# - A custom environment wrapper with a reward function
# - Model setup and training logic
# - Evaluation and metric logging
# - Model saving and best-model tracking

def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate PPO, SAC, or DDPG models.")
    parser.add_argument("--algo", type=str, choices=["ppo", "sac", "ddpg"], required=True,
                        help="Algorithm to train")
    parser.add_argument("--timesteps", type=int, default=1000000,
                        help="Total timesteps to train")
    return parser.parse_args()

# F110_Wrapper:
# Extends the F1TENTH Gym environment with:
# - 1D laser scan observation (1080 beams)
# - Custom reward function encouraging speed, centering, and lap completion
# - Lap tracking logic based on crossing a start line
class F110_Wrapper(gym.Wrapper):
    def __init__(self, env, algo='ppo'):
        super(F110_Wrapper, self).__init__(env)
        # Observation: 1080 laser beams
        self.algo = algo.lower()
        self.observation_space = spaces.Box(low=0.0, high=30.0, shape=(1080,), dtype=np.float32)
        # Action: [steering angle, speed]
        self.action_space = spaces.Box(low=np.array([-1.0, 0.0]), high=np.array([1.0, 2.0]), dtype=np.float32)
        self.step_count = 0
        self.lap_start_step = 0
        # Step counters
        self.step_count = 0
        self.lap_start_step = 0
        
        # Finish-line and lap tracking
        self._start_pt = None              # Start position (x, y)
        self._start_normal = None          # Unit vector of heading at start
        self._prev_forward_pos = None      # How far the car was from the start line in the last step
        self._lap_ready = False            # True after driving away from start
        self._lap_warmup_dist = 1.0        # Min distance before lap tracking starts
        self._line_half_width = 1.0        # Crossing must be within 1.0 m of line

    def reset_counters(self):
        self.step_count = 0
        self.lap_count = 0
        self.lap_start_step = 0
        self.collisions = 0

    def reset(self, **kwargs):
        # Randomise initial pose unless provided
        seed = kwargs.pop('seed', None)
        if 'poses' not in kwargs:
            rng = np.random.default_rng(seed)
            x = rng.uniform(-0.5, 0.5)
            y = rng.uniform(-0.5, 0.5)
            theta = rng.uniform(-0.1, 0.1)
            kwargs['poses'] = np.array([[x, y, theta]])

        obs, _, _, _ = self.env.reset(**kwargs)

        # Define start line and direction
        x0, y0 = obs['poses_x'][0], obs['poses_y'][0]
        theta0 = obs['poses_theta'][0]
        self._start_pt = (x0, y0)
        self._start_normal = (math.cos(theta0), math.sin(theta0))

        # Compute signed distance from car to finish line
        dx = x0 - self._start_pt[0]
        dy = y0 - self._start_pt[1]
        self._prev_forward_pos = dx * self._start_normal[0] + dy * self._start_normal[1]

        self._lap_ready = False
        self.reset_counters()
        return obs['scans'][0]

    # Reward function
    def step(self, action):
        self.step_count += 1
        steer, speed = action
        obs, _, done, info = self.env.step(np.array([[steer, speed]]))
        scan = obs['scans'][0]
        done = False
        info['collision'] = False

        # Get position and directional projection
        x, y = obs['poses_x'][0], obs['poses_y'][0]
        dx, dy = x - self._start_pt[0], y - self._start_pt[1]

        # Check if the car has moved far enough from the start to begin counting laps
        if not self._lap_ready and math.hypot(dx, dy) > self._lap_warmup_dist:
            self._lap_ready = True

        # How far the car is from the start line along its initial direction
        forward_pos = dx * self._start_normal[0] + dy * self._start_normal[1]
        prev_forward_pos = self._prev_forward_pos
        self._prev_forward_pos = forward_pos  # Save for next step

        # How far the car is from the finish line (sideways distance)
        tx, ty = -self._start_normal[1], self._start_normal[0]
        side_distance = abs(dx * tx + dy * ty)

        # How much the car moved forward since the last step
        progress = forward_pos - prev_forward_pos if prev_forward_pos is not None else 0.0
        progress_reward = 5.0 * progress
        
        collision = np.min(scan) < 0.15
        collision_penalty = -5.0 if collision else 0.0
        forward_reward = speed * 2.5  
        steer_penalty = -1.0 * abs(steer)   # discourage extreme steering

        # Keep LiDAR-based centering simple
        left_scan = scan[:360]
        right_scan = scan[720:]
        left_mean = np.mean(left_scan)
        right_mean = np.mean(right_scan)
        symmetry_bonus = max(0.0, 0.5 - abs(left_mean - right_mean))  # re-wards being centered

        # Keep wall proximity penalty linear
        min_left = np.min(left_scan)
        min_right = np.min(right_scan)
        wall_penalty = -0.5 if min_left < 0.3 or min_right < 0.3 else 0.0

        # Remove or adjust reward value for experimentation
        reward = (
            forward_reward +         # prefer faster
            progress_reward +        # reward for progress
            collision_penalty +      # penalise crashes
            symmetry_bonus +         # prefer centred
            steer_penalty +          # penalise harsh steering
            wall_penalty             # penalise being close to wall
        )
        reward -= 0.01

        # Lap counting
        if self._lap_ready and prev_forward_pos is not None:
            if prev_forward_pos <= 0 < forward_pos and side_distance < self._line_half_width:
                # Car has crossed the line and was close enough — count a lap
                self.lap_count += 1
                lap_steps = self.step_count - self.lap_start_step
                lap_time_sec = lap_steps / 100.0
                info['lap_time'] = round(lap_time_sec, 2)

                self.lap_start_step = self.step_count
                reward += 50.0  # lap bonus
        info['laps'] = self.lap_count

        if collision:
            self.collisions += 1
            info['collision'] = self.collisions
            done = True

        MAX_STEPS = 15000
        if self.step_count >= MAX_STEPS:
            done = True

        return scan, reward, done, info


def make_model(env, algo):
    # Select device: GPU for SAC/DDPG, CPU for PPO
    device = "cuda" if algo in ["sac", "ddpg"] else "cpu"

    # Shared settings
    common_params = {
        "policy": "MlpPolicy",
        "env": env,
        "verbose": 1,
        "device": device
    }

    # Algorithm-specific settings
    # Experiment with these parameters to improve performance
    algo_params = {
        "ppo": dict(batch_size=2048, n_steps=8192, learning_rate=3e-5,
                    ent_coef=0.01, n_epochs=6, vf_coef=0.6,
                    policy_kwargs=dict(net_arch=[512, 512])),

        "sac": dict(batch_size=256, buffer_size=1000000, learning_rate=3e-4,
                    learning_starts=2500, ent_coef='auto', 
                    policy_kwargs=dict(net_arch=[256, 256])),

        "ddpg": dict(batch_size=128, buffer_size=1000000, learning_rate=1e-4,
                     learning_starts=2500, 
                    action_noise=NormalActionNoise(
                    mean=np.zeros(env.action_space.shape[0]),
                    sigma=0.3 * np.ones(env.action_space.shape[0])),
                    policy_kwargs=dict(net_arch=[256, 256]))
    }
    
    if algo == "ppo":
        return PPO(**common_params, **algo_params["ppo"])
    elif algo == "sac":
        return SAC(**common_params, **algo_params["sac"])
    elif algo == "ddpg":
        return DDPG(**common_params, **algo_params["ddpg"])
    else:
        raise ValueError("Unsupported algorithm")

    
def evaluate_agent(model, env, episodes, max_steps):
    ep_rewards, ep_lengths, ep_collisions, lap_counts, lap_times = [], [], [], [], []
    crash_count = 0
    for ep in range(episodes):
        obs = env.reset(seed=ep)
        done, steps, total_reward = False, 0, 0.0
        while not done and steps < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

            total_reward += reward
            steps += 1
            
            if 'lap_time' in info:
                lap_times.append(info['lap_time'])

        # Only count 1 collision per episode, if it caused the episode to end
        crashed = int(info.get("collision", 0) > 0)
        ep_collisions.append(crashed)
        ep_rewards.append(total_reward)
        ep_lengths.append(steps)
        lap_counts.append(info.get('laps', 0))
 

    return {
        "mean_reward"   : float(np.mean(ep_rewards)),
        "avg_length"    : float(np.mean(ep_lengths)),
        "eps_with_crash": float(np.sum(ep_collisions)),
        "avg_lap_count": float(np.mean(lap_counts)),
        "mean_lap_time": float(np.mean(lap_times)) if lap_times else 0.0,
        "total_laps": sum(lap_counts)
    }

def create_next_path(folder, prefix="model_", ext=".zip"):
    os.makedirs(folder, exist_ok=True)
    pat = re.compile(rf"{prefix}(\d+){re.escape(ext)}")
    files = [f for f in os.listdir(folder) if pat.match(f)]
    idxs = [int(pat.match(f).group(1)) for f in files]
    next_idx = max(idxs, default=0) + 1
    return os.path.join(folder, f"{prefix}{next_idx:03d}{ext}"), idxs


def save_model(model, metrics, algo, folder="models", limit=25):
    reward = metrics['mean_reward']
    prefix = f"{algo}_model_"

    # Save current model
    path, idxs = create_next_path(folder, prefix=prefix)
    model.save(path)
    print(f"Model saved to: {path}")

    # Maintain only the most recent `limit` models
    model_regex = re.compile(rf"{prefix}(\d+)\.zip")
    matching_files = sorted(
        [(int(model_regex.match(f).group(1)), f) for f in os.listdir(folder) if model_regex.match(f)],
        key=lambda x: x[0]
    )
    if len(matching_files) > limit:
        to_delete = matching_files[0][1]
        os.remove(os.path.join(folder, to_delete))
        print(f"Oldest model removed: {to_delete}")

    # Save best model based on reward
    score_file = os.path.join(folder, f"best_{algo}_reward.txt")
    best = -np.inf
    try:
        with open(score_file, 'r') as f:
            best = float(f.read())
    except:
        pass

    if reward > best:
        best_model_path = os.path.join(folder, f"best_{algo}_model.zip")
        model.save(best_model_path)
        with open(score_file, 'w') as f:
            f.write(str(reward))
        print(f"Best {algo.upper()} model updated.")

    # Log evaluation metrics
    csv_path = os.path.join(folder, 'metrics.csv')
    log_metrics(csv_path, path, algo, metrics)

    return path

def log_metrics(csv_path, model_path, algo, metrics):
    header = ['model', 'algo', 'mean_reward', 'avg_length', 'eps_with_crash', 'avg_lap_count', 'mean_lap_time']

    first = not os.path.exists(csv_path)
    
    with open(csv_path, 'a', newline='') as f:
        w = csv.writer(f)
        if first:
            w.writerow(header)  # Write headers if the file does not exist
        w.writerow([
            model_path,
            algo.upper(),
            metrics['mean_reward'],
            metrics['avg_length'],
            metrics['eps_with_crash'],
            metrics['avg_lap_count'],
            metrics['mean_lap_time']
        ])


def plot_metrics(folder="models"):
    # Load evaluation metrics from CSV file
    df = pd.read_csv(os.path.join(folder, 'metrics.csv'))

    # Define which metrics to plot and how to evaluate them
    # Each entry: (column name, y-axis label, plot title, filename, aggregation method)
    metrics = [
        ("mean_reward", "Mean Reward", "Best Mean Reward by Algo", "mean_reward.png", "max"),
        ("eps_with_crash", "Amount of episodes with crash", "Lowest Mean Collisions by Algo", "eps_with_crash.png", "min"),
        ("avg_length", "Avg Length (steps)", "Longest Average Episode Length", "avg_length.png", "max"),
        ("avg_lap_count", "Lap Completion Rate (%)", "Average Lap Count", "avg_lap_count.png", "max"),
        ("mean_lap_time", "Mean Lap Time (steps)", "Fastest Mean Lap Time", "mean_lap_time.png", "min"),
    ]

    # Generate a bar plot for each selected metric
    for metric, ylabel, title, filename, agg in metrics:
        # Select the best-performing model for each algorithm (grouped by 'algo')
        if agg == "max":
            best = df.loc[df.groupby('algo')[metric].idxmax()]
        else:
            best = df.loc[df.groupby('algo')[metric].idxmin()]

        # Create and save the plot
        plt.figure(figsize=(6, 4))
        plt.bar(best['algo'], best[metric])
        plt.title(title)
        plt.ylabel(ylabel)
        plt.savefig(os.path.join(folder, filename))
        plt.close()



if __name__ == '__main__':
    args = parse_args()
    # setup
    folder = 'models'
    os.makedirs(folder, exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=200000,
        save_path="models",
        name_prefix=args.algo + "_ver"
    )

    # environment
    env = gym.make(
        'f110_gym:f110-v0',
        map='/sim_ws/src/f1tenth_gym_ros/maps/Spielberg_map',
        map_ext='.png',
        num_agents=1
    )
    env = F110_Wrapper(env)  

    # train
    model = make_model(env, args.algo)
    model.learn(total_timesteps=args.timesteps, callback=checkpoint_callback)

    # evaluate
    TEST_EPISODES = 5
    metrics = evaluate_agent(model, env, TEST_EPISODES, 5000)
    print(f"Evaluation over {TEST_EPISODES} test episodes: reward={metrics['mean_reward']:.2f}, "
        f"avg_length={metrics['avg_length']:.1f}, "
        f"mean_collisions={metrics['mean_collisions']:.1f}")

    # save & log
    save_model(model, metrics, args.algo, folder)

    # plot results
    plot_metrics(folder)
