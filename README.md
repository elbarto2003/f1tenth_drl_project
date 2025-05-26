# F1TENTH RL Racing Project

This project builds on the F1TENTH Gym ROS simulator to train and evaluate autonomous racing agents using reinforcement learning (PPO, SAC, DDPG).

---

## Getting Started (Docker Setup)

### With NVIDIA GPU (recommended)

1. Clone the F1TENTH simulator and build the container:
```bash
git clone https://github.com/f1tenth/f1tenth_gym_ros.git
cd f1tenth_gym_ros
docker build -t f1tenth_gym_ros -f Dockerfile .
```

2. Run the container:
```bash
rocker --nvidia --x11 --volume .:/sim_ws/src/f1tenth_gym_ros -- f1tenth_gym_ros
```

3. Inside the container, launch the simulator:
```bash
source /opt/ros/foxy/setup.bash
source install/local_setup.bash
ros2 launch f1tenth_gym_ros gym_bridge_launch.py
```

---

### Without an NVIDIA GPU (noVNC + docker-compose)

If your system doesn't support `nvidia-docker2`, you can use **noVNC** to forward the display from the container to your browser.

1. Make sure you have **Docker** and **Docker Compose** installed:
```bash
sudo apt install docker docker-compose
```

2. Clone the F1TENTH simulator and launch the containers:
```bash
git clone https://github.com/f1tenth/f1tenth_gym_ros.git
cd f1tenth_gym_ros
docker-compose up
```

3. In another terminal, open a shell inside the simulation container:
```bash
docker exec -it f1tenth_gym_ros-sim-1 /bin/bash
```

4. Inside the container, launch the simulator:
```bash
source /opt/ros/foxy/setup.bash
source install/local_setup.bash
ros2 launch f1tenth_gym_ros gym_bridge_launch.py
```

5. In your web browser, go to:
```
http://localhost:8080/vnc.html
```
Click the **Connect** button to open the simulation GUI via noVNC.

---

## My Contributions

| File | Purpose |
|------|---------|
| `train_models.py` | Train PPO, SAC, or DDPG models with a custom reward |
| `continue_training.py` | Resume training from a saved model checkpoint |
| `evaluate_model.py` | Evaluate a model and log metrics to CSV |
| `vis_eval.py` | Run and visualise episodes with stats and rendering |
| `plot_metrics.py` | Generate performance plots from evaluation data |
| `my_controller.py` | Deploy a trained model as a ROS2 control node |
| `dual_controller_launch.py` | Launch two agents for adversarial testing |
| `train_all_models.sh` | Train multiple models from shell script |

Models, metrics, and plots are saved in the `models/` folder.

Feel free to modify any part of the code from my contributions to suit your project needs.

If changes to the simulation environment itself are required (e.g. sensor inputs, launch behaviour, topic remapping), these can typically be made in the `gym_bridge.py` file within the F1TENTH Gym ROS package.

---

## Reward Function and Training

The reward function is defined in the `step()` method of the `F110_Wrapper` class in `train_models.py`. You can adjust:

- **Speed incentives**: how much faster movement is rewarded
- **Crash penalties**: how strongly collisions are penalised
- **Lap bonuses**: how many points are awarded per completed lap
- **Wall proximity and centering**: small incentives/penalties for position on track

This lets you shape behaviour, e.g. favouring speed, safety, or lap count.

### Example: Train a SAC agent
```bash
python3 train_models.py --algo sac --timesteps 1000000
```

You can also edit `train_all_models.sh` to train multiple agents in sequence.

---

## Testing Trained Models in the Simulation

To deploy a trained RL model inside the simulation:

### Single Agent

Use `my_controller.py` to control one vehicle with your chosen model:
```bash
ros2 run my_controller.py --model sac_model_004.zip --algo sac
```

Make sure:
- The model file (e.g. `sac_model_004.zip`) is located in the `models/` folder.
- The simulator has been launched using `gym_bridge_launch.py`.

### Dual Agent

If you have `num_agent` set to 2 in your `sim.yaml`, you can run two agents against each other using:
```bash
ros2 launch f1tenth_gym_ros dual_controller_launch.py
```

The file will launch both ego and opponent agents with their respective models and topic names preconfigured.

You can modify `dual_controller_launch.py` to select different models or adjust topic routing if needed.

---

## Visualising a Trained Agent

To visualise a model in the simulator with live rendering and stats:
```bash
python3 vis_eval.py --model sac_model_004 --algo sac --episodes 3 --render 20
```

---

## Notes

- Three maps are included, but the **Spielberg map** has the simplest layout for testing performance and handling.
- You can configure the simulation to use **one or two agents** by editing the `num_agent` parameter in the `sim.yaml` configuration file.
- The **starting position** of the ego and opponent cars can also be manually set in the same file.

---

## License

Based on [F1TENTH Gym ROS](https://github.com/f1tenth/f1tenth_gym_ros). Refer to their license for usage terms.
