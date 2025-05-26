from launch import LaunchDescription
from launch_ros.actions import Node
import os

# ROS 2 launch file to run two reinforcement learning agents (ego and opponent)
def generate_launch_description():

    # Absolute path to the controller script (mounted into /sim_ws/src in Docker)
    ctrl_script = '/sim_ws/src/f1tenth_gym_ros/my_controller.py'

    # Define models and algorithms for ego and opponent cars
    MODEL    = 'sac_model_004.zip' # ego model
    OP_MODEL = 'ppo_slow.zip' # opponent model
    ALGO     = 'sac'
    OP_ALGO  = 'ppo'

    ego = Node(
        executable='python3',
        name='ego_controller',
        output='screen',
        arguments=[
            ctrl_script,
            '--model', MODEL,
            '--algo',  ALGO,
            '--scan_topic',  '/scan',
            '--drive_topic', '/drive'
        ],
    )

    opp = Node(
        executable='python3',
        name='opp_controller',
        output='screen',
        arguments=[
            ctrl_script,
            '--model', OP_MODEL,
            '--algo',  OP_ALGO,
            '--scan_topic',  '/opp_scan',
            '--drive_topic', '/opp_drive'
        ],
    )

    return LaunchDescription([ego, opp])
