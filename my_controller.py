import rclpy, os, argparse, numpy as np
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from stable_baselines3 import PPO, SAC, DDPG

class RLAgent(Node):
    def __init__(self, model_file, algo, scan_topic, drive_topic):
        super().__init__('rl_controller')

        # Subscribe to LiDAR scans and publish drive commands 
        self.subscriber_scan = self.create_subscription(
            LaserScan, scan_topic, self.scan_callback, 10)
        self.publisher_drive = self.create_publisher(
            AckermannDriveStamped, drive_topic, 10)

        # Load the trained RL model
        path = os.path.join("/sim_ws/src/f1tenth_gym_ros/models", model_file)
        algo = algo.lower()
        if   algo == 'ppo':  self.model = PPO.load(path,  weights_only=True)
        elif algo == 'sac':  self.model = SAC.load(path,  weights_only=True)
        elif algo == 'ddpg': self.model = DDPG.load(path, weights_only=True)
        else: raise ValueError("algo must be ppo/sac/ddpg")
        self.get_logger().info(f"Loaded model: {path}")

        self.scan = None # Holds latest LiDAR scan data
        self.timer = self.create_timer(0.01, self.control_loop)  # Control loop at 100 Hz

    # Callback to process incoming LiDAR data
    def scan_callback(self, msg):
        self.scan = np.array(msg.ranges, dtype=np.float32)

    # Use model to decide speed and steering from scan data
    def decide(self):
        action, _ = self.model.predict(self.scan.reshape(1, -1),
                                       deterministic=True)
        action = action[0]
        if action.size == 2:
            steer, speed = map(float, action)
        elif action.size == 1:
            speed = float(action[0])
            steer = 0.0
        else:
            raise RuntimeError(f"Unexpected action length {action.size}")
        return speed, steer

    # Main control loop: send driving command based on model output
    def control_loop(self):
        if self.scan is None:
            self.get_logger().info_once("Waiting for scan data…")
            return
        speed, steer = self.decide()

        msg = AckermannDriveStamped()
        msg.drive.speed, msg.drive.steering_angle = speed, steer
        self.publisher_drive.publish(msg)
        self.get_logger().info(
            f"Cmd published: speed={speed:.2f}  steer={steer:.2f}")

# ROS 2 entry point with argument parsing
def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--algo",  required=True, choices=['ppo','sac','ddpg'])
    p.add_argument("--scan_topic",  default="/scan")
    p.add_argument("--drive_topic", default="/drive")
    cfg, ros_args = p.parse_known_args(argv)

    rclpy.init(args=ros_args)
    node = RLAgent(cfg.model, cfg.algo, cfg.scan_topic, cfg.drive_topic)
    rclpy.spin(node)
    node.destroy_node(); rclpy.shutdown()

if __name__ == "__main__":
    main()
