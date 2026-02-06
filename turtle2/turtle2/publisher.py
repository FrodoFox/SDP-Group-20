import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import LaserScan
from rclpy.qos import QoSProfile, ReliabilityPolicy
import random

# Parameters
FORWARD_SPEED = 0.2      # m/s
SPIN_SPEED = 1.0         # rad/s
SAFE_DISTANCE = 1.0    # meters
FRONT_ANGLE = 0.52       # radians (~30°)
LIDAR_MIN = 0.05         # minimum valid LIDAR reading

class Wanderer(Node):
    def __init__(self):
        super().__init__('wanderer_robot')

        # Publisher to cmd_vel
        self.cmd_pub = self.create_publisher(TwistStamped, 'cmd_vel', 10)

        # QoS profile for LIDAR
        qos_profile = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT
        )

        # Subscriber to LIDAR scan
        self.scan_sub = self.create_subscription(
            LaserScan,
            'scan',
            self.scan_callback,
            qos_profile
        )

        # Timer for updating motion
        self.timer = self.create_timer(0.1, self.update_motion)

        # State variables
        self.obstacle_detected = False
        self.spinning = False
        self.spin_direction = random.choice([-1, 1])

    def scan_callback(self, msg: LaserScan):
        # Collect front LIDAR readings ±FRONT_ANGLE
        front_ranges = [
            r for i, r in enumerate(msg.ranges)
            if -FRONT_ANGLE < (msg.angle_min + i * msg.angle_increment) < FRONT_ANGLE
        ]

        # Remove invalid readings
        valid_ranges = [r for r in front_ranges if r == r and r != float('inf') and r > LIDAR_MIN]

        # Detect obstacle if any reading < SAFE_DISTANCE
        obstacle_now = bool(valid_ranges and min(valid_ranges) < SAFE_DISTANCE)

        if obstacle_now and not self.obstacle_detected:
            self.get_logger().info(f"Obstacle detected at {min(valid_ranges):.2f} m! Starting to spin.")
            # Randomize spin direction each time an obstacle is detected
            self.spin_direction = random.choice([-1, 1])

        self.obstacle_detected = obstacle_now

    def update_motion(self):
        msg = TwistStamped()

        if self.obstacle_detected:
            # Spin in place until path is clear
            msg.twist.linear.x = 0.0
            msg.twist.angular.z = self.spin_direction * SPIN_SPEED
            if not self.spinning:
                self.spinning = True
                self.get_logger().info("Spinning to avoid obstacle...")
        else:
            # Path is clear, move forward
            msg.twist.linear.x = FORWARD_SPEED
            msg.twist.angular.z = 0.0
            if self.spinning:
                self.spinning = False
                self.get_logger().info("Path clear. Moving forward...")

        self.cmd_pub.publish(msg)


def main():
    rclpy.init()
    node = Wanderer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
