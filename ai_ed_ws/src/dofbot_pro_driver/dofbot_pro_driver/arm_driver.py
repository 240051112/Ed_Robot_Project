#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from dofbot_pro_interface.msg import ArmJoint
from sensor_msgs.msg import JointState

try:
    from Arm_Lib import Arm_Device
    HAS_HW = True
except ImportError:
    HAS_HW = False

class ArmDriver(Node):
    def __init__(self):
        super().__init__('arm_driver')
        self.hw = Arm_Device() if HAS_HW else None
        self.current_positions = [90.0] * 6
        self.create_subscription(
            ArmJoint, 'TargetAngle', self._on_command_received, 10)
        self.pub_state = self.create_publisher(
            JointState, '/ArmAngleUpdate', 10)
        self.get_logger().info(f'Arm driver ready (Hardware present: {HAS_HW})')

    def _on_command_received(self, msg: ArmJoint):
        if msg.joints:
            self._move_all_servos(msg.joints, msg.run_time)
            self.current_positions = list(msg.joints)
        elif 1 <= msg.id <= 6:
            self._move_one_servo(msg.id, msg.angle, msg.run_time)
            self.current_positions[msg.id - 1] = msg.angle
        else:
            self.get_logger().warning(f'Received command with illegal servo ID: {msg.id}')
            return
        self._publish_current_state()

    def _move_all_servos(self, angles: list[float], run_ms: int):
        if self.hw:
            self.hw.Arm_serial_servo_write6(*angles, time=run_ms)
        else:
            self.get_logger().info(f'[SIM] Move6: {angles} over {run_ms}ms')

    def _move_one_servo(self, servo_id: int, angle: float, run_ms: int):
        if self.hw:
            self.hw.Arm_serial_servo_write(servo_id, angle, run_ms)
        else:
            self.get_logger().info(f'[SIM] Move1: id={servo_id}, angle={angle}, time={run_ms}ms')

    def _publish_current_state(self):
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = [f'joint{i+1}' for i in range(6)]
        js.position = [float(p) for p in self.current_positions]
        self.pub_state.publish(js)
        self.get_logger().info(f'Published state: {js.position}')

def main(args=None):
    rclpy.init(args=args)
    node = ArmDriver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()