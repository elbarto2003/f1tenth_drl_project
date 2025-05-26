#!/bin/bash

# python3 train_models.py --algo ppo --timesteps 3000000
python3 train_models.py --algo sac --timesteps 1000000
python3 train_models.py --algo ddpg --timesteps 1000000
