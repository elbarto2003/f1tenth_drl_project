from launch import LaunchDescription
from launch_ros.actions import Node
import os

def generate_launch_description():
    # <-- point directly to the file in your workspace src folder, not install/share
    ctrl_script = '/sim_ws/src/f1tenth_gym_ros/my_controller.py'
    MODEL       = 'sac_model_004.zip'
    OP_MODEL   = 'ppo_slow.zip'
    ALGO        = 'sac'
    OP_ALGO    = 'ppo'

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
