#!/usr/bin/env python3

import time
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool


BALL_LOCATIONS = [
    (1.0, 0.5),
    (1.2, -0.4),
    (0.8, 0.3),
    (1.5, 0.0),
    (1.0, -0.6)
]

TEE_LOCATION = (0.0, 0.0)
MAX_BALLS = 5
TIME_LIMIT = 30  # seconds


class BallMission(Node):

    def __init__(self):
        super().__init__('ball_mission')

        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.pickup_pub = self.create_publisher(Bool, '/pickup_ball', 10)

        self.start_time = time.time()
        self.collected = 0

        self.get_logger().info('Waiting for Nav2...')
        self.nav_client.wait_for_server()
        self.get_logger().info('Nav2 ready.')

        self.run_mission()

    def send_goal(self, x, y):
        goal = NavigateToPose.Goal()

        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.orientation.w = 1.0

        goal.pose = pose

        self.get_logger().info(f'Navigating to ({x:.2f}, {y:.2f})')
        future = self.nav_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)

        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('Goal rejected')
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)

        return True

    def pick_up_ball(self):
        self.get_logger().info('Picking up ball')
        msg = Bool()
        msg.data = True
        self.pickup_pub.publish(msg)
        time.sleep(1)

    def run_mission(self):
        for x, y in BALL_LOCATIONS:
            if self.collected >= MAX_BALLS:
                break
            if time.time() - self.start_time > TIME_LIMIT:
                break

            if self.send_goal(x, y):
                self.pick_up_ball()
                self.collected += 1
                self.get_logger().info(f'Collected {self.collected} balls')

        self.get_logger().info('Returning to tee')
        self.send_goal(*TEE_LOCATION)
        self.get_logger().info('Mission complete')


def main():
    rclpy.init()
    node = BallMission()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
