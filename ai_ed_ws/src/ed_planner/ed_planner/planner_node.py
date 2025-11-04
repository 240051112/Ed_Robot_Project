#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np

# This node subscribes to our custom 3D detection message
from dofbot_pro_interface.msg import DofbotDetection3DArray, ArmJoint

class PlannerNodeV2(Node):
    def __init__(self):
        super().__init__('planner_node')
        
        # --- Parameters for easy tuning ---
        # The arm will track objects within +/- this distance (in meters) from the camera's center
        self.declare_parameter('tracking_horizontal_range_m', 0.2) 
        # The arm's joint angle limits for tracking
        self.declare_parameter('min_angle_deg', 45.0)
        self.declare_parameter('max_angle_deg', 135.0)
        
        self.tracking_range = self.get_parameter('tracking_horizontal_range_m').get_parameter_value().double_value
        self.min_angle = self.get_parameter('min_angle_deg').get_parameter_value().double_value
        self.max_angle = self.get_parameter('max_angle_deg').get_parameter_value().double_value
        
        # Publisher to send commands to the arm
        self.arm_publisher_ = self.create_publisher(ArmJoint, '/TargetAngle', 10)
        
        # Subscriber to receive 3D detections from the vision node
        self.object_subscription_ = self.create_subscription(
            DofbotDetection3DArray,
            '/ed/detections_3d',
            self.object_callback,
            10)
        
        self.last_sent_angle_ = 90.0
        self.get_logger().info('Planner node V2 is running and ready.')

    def object_callback(self, msg: DofbotDetection3DArray):
        # If no objects are detected, do nothing
        if not msg.detections:
            return
        
        # --- Decision Logic: Find the closest valid object ---
        closest_object = None
        min_distance = float('inf')
        for detection in msg.detections:
            # Check if the detection has a valid 3D position (Z > 0)
            if detection.position.z > 0 and detection.position.z < min_distance:
                min_distance = detection.position.z
                closest_object = detection
        
        if closest_object is None:
            return

        # Get the real-world X coordinate (left/right) of the closest object
        object_x_position = closest_object.position.x
        
        # --- Calculation Logic: Map the object's position to an arm angle ---
        # A positive X is to the camera's left, a negative X is to the right
        target_angle = np.interp(object_x_position, 
                                 [-self.tracking_range, self.tracking_range], 
                                 [self.max_angle, self.min_angle])
        
        # Clamp the angle to the physical limits of the robot (0-180 degrees)
        target_angle = np.clip(target_angle, 0.0, 180.0)

        # Only send a new command if the angle has changed significantly to avoid jitter
        if abs(target_angle - self.last_sent_angle_) > 2.0:
            self.get_logger().info(
                f"Tracking '{closest_object.class_name}' at X={object_x_position:.2f}m -> Commanding base to {target_angle:.1f} degrees.")
            self.last_sent_angle_ = target_angle
            
            # --- Action: Publish the command to the arm driver ---
            cmd_msg = ArmJoint()
            cmd_msg.id = 1 # Joint 1 is the base rotation
            cmd_msg.angle = target_angle
            cmd_msg.run_time = 500 # Time in milliseconds for the move
            self.arm_publisher_.publish(cmd_msg)

def main(args=None):
    rclpy.init(args=args)
    node = PlannerNodeV2()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()