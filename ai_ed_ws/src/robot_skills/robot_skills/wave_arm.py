import rclpy
from rclpy.node import Node
from dofbot_pro_interface.msg import ArmJoint
import time

class ArmWaver(Node):
    def __init__(self):
        super().__init__('arm_waver_node')
        self.publisher = self.create_publisher(ArmJoint, '/TargetAngle', 10)
        self.get_logger().info('Final Safe Arm Waver node started.')
        time.sleep(1) # Allow time for publisher to connect
        self.wave()

    def wave(self):
        # Define the safe poses for the sequence
        home_pose = [90.0, 90.0, 90.0, 90.0, 90.0, 135.0]
        wave_left = [90.0, 90.0, 90.0, 45.0, 90.0, 135.0]
        # NEW: Reduced angle for wave_right to prevent collision
        wave_right = [90.0, 90.0, 90.0, 110.0, 90.0, 135.0]

        # The full, safe sequence
        poses = [home_pose, wave_left, wave_right, wave_left, wave_right, home_pose]
        run_times = [2000, 1500, 1500, 1500, 1500, 2000] # Slower movements (in ms)

        self.get_logger().info('Starting final safe wave sequence...')
        for i, pose in enumerate(poses):
            msg = ArmJoint()
            msg.joints = pose
            msg.run_time = run_times[i]
            self.publisher.publish(msg)
            self.get_logger().info(f'Moving to pose {i+1}')
            time.sleep(run_times[i] / 1000.0 + 0.5)

        self.get_logger().info('Waving sequence complete.')
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = ArmWaver()
    node.destroy_node()

if __name__ == '__main__':
    main()