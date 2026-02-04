import rclpy
import threading
import time
import sys

from turtlebot_controller.ball import BallMission


def _mission_runner(node_holder, finished_event):
    """Constructs and runs BallMission. BallMission runs its mission in its constructor
    (run_mission is called inside __init__), so this function will block until the
    mission completes or an exception occurs. The created node is stored in
    node_holder['node'] so the main thread can try to destroy it on shutdown.
    """
    try:
        node = BallMission()
        node_holder['node'] = node
    except Exception as exc:
        print(f"BallMission failed: {exc}", file=sys.stderr)
    finally:
        finished_event.set()


def main():
    """Launcher entrypoint for running the BallMission node.

    Behavior:
    - Calls rclpy.init() once.
    - Starts the mission in a dedicated thread so the main thread can handle
      signals and graceful shutdown.
    - Waits for the mission to finish or for a KeyboardInterrupt, then performs
      cleanup (destroying the node and shutting down rclpy).
    """
    rclpy.init()
    """Launcher entrypoint for running the BallMission node.

    Behavior:
    - Calls rclpy.init() once.
    - Starts the mission in a dedicated thread so the main thread can handle
      signals and graceful shutdown.
    - Waits for the mission to finish or for a KeyboardInterrupt, then performs
      cleanup (destroying the node and shutting down rclpy).
    """
    node_holder = {}
    finished = threading.Event()

    thread = threading.Thread(target=_mission_runner, args=(node_holder, finished), daemon=False)
    thread.start()

    try:
        # Wait until the mission finishes; check periodically so we remain responsive
        while not finished.wait(timeout=0.5):
            # Could add status/logging here if desired
            pass
    except KeyboardInterrupt:
        # Attempt to stop the mission node if it has been created
        print('KeyboardInterrupt received — attempting to stop BallMission...')
        node = node_holder.get('node')
        if node is not None:
            try:
                node.get_logger().info('Shutdown requested — destroying node')
                node.destroy_node()
            except Exception:
                pass
    finally:
        # Ensure the thread finishes (it should if the mission completed or node destroyed)
        thread.join(timeout=2.0)
        rclpy.shutdown()


if __name__ == '__main__':
    main()
