#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from sensor_msgs.msg import Image
from message_filters import ApproximateTimeSynchronizer, Subscriber
from cv_bridge import CvBridge
from dt_apriltags import Detector
from dofbot_pro_driver.vutils import draw_tags
from dofbot_pro_interface.msg import AprilTagInfo, ArmJoint

class AprilTagDetectNode(Node):
    def __init__(self):
        super().__init__('apriltag_detect')
        self.get_logger().info("AprilTag Detector Node is running (RGB + Depth windows).")

        # Move to a known pose at startup (optional)
        self.init_joints = [90.0, 120.0, 0.0, 0.0, 90.0, 30.0]

        # Sync RGB + depth
        self.rgb_sub   = Subscriber(self, Image, '/camera/color/image_raw')
        self.depth_sub = Subscriber(self, Image, '/camera/depth/image_raw')
        self.ts = ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub], queue_size=10, slop=0.5)
        self.ts.registerCallback(self._cb)

        # Publishers
        self.pos_pub = self.create_publisher(AprilTagInfo, 'PosInfo', 10)
        self.arm_pub = self.create_publisher(ArmJoint, 'TargetAngle', 1)

        self.rgb_bridge   = CvBridge()
        self.depth_bridge = CvBridge()

        self.det = Detector(
            searchpath=['apriltags'],
            families='tag36h11',
            nthreads=8, quad_decimate=2.0, quad_sigma=0.0,
            refine_edges=1, decode_sharpening=0.25, debug=0
        )

        # Windows you can resize freely
        cv2.namedWindow('result_image', cv2.WINDOW_NORMAL)
        cv2.namedWindow('depth_image',  cv2.WINDOW_NORMAL)
        cv2.resizeWindow('result_image', 900, 650)
        cv2.resizeWindow('depth_image',  900, 650)

    # Optional “go to observe”
    def _publish_arm(self, joints, runtime=1500):
        m = ArmJoint(); m.run_time = int(runtime); m.joints = [float(j) for j in joints]
        self.arm_pub.publish(m)

    def _cb(self, rgb_msg: Image, depth_msg: Image):
        # --- RGB ---
        rgb = self.rgb_bridge.imgmsg_to_cv2(rgb_msg, 'rgb8')
        vis = np.copy(rgb)

        # --- Depth (robust to 16UC1 mm or 32FC1 m) ---
        depth = self.depth_bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        if depth.dtype == np.uint16:                     # 16UC1, millimeters
            depth_m = depth.astype(np.float32) / 1000.0  # -> meters
        else:                                            # e.g. float32 meters already
            depth_m = depth.astype(np.float32)

        # If resolutions differ, resize depth to RGB size
        if depth_m.shape[:2] != rgb.shape[:2]:
            depth_m = cv2.resize(depth_m, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Colorize depth for viewing (clip 0.10–0.80 m; adjust to your scene)
        d_show = depth_m.copy()
        d_show[~np.isfinite(d_show)] = 0.0
        vmin, vmax = 0.10, 0.80
        d_norm = np.clip((d_show - vmin) / (vmax - vmin), 0.0, 1.0)
        depth_color = cv2.applyColorMap((d_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)

        # --- Detect & annotate ---
        tags = self.det.detect(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), False, None, 0.025)
        draw_tags(vis, tags, corners_color=(50, 240, 255), center_color=(80, 255, 80))

        for t in tags:
            cx, cy = t.center
            u, v = int(round(cx)), int(round(cy))
            z = 0.0
            if 0 <= v < depth_m.shape[0] and 0 <= u < depth_m.shape[1]:
                z = float(depth_m[v, u])
                if not np.isfinite(z) or z <= 0.0:
                    z = 0.0

            # Publish PosInfo (z in meters)
            msg = AprilTagInfo()
            msg.id = int(t.tag_id); msg.x = float(cx); msg.y = float(cy); msg.z = float(z)
            if msg.z > 0.0:
                self.pos_pub.publish(msg)

            # Readable outline text
            label = f"ID:{msg.id}  px=({u},{v})  z={msg.z:.3f}m"
            org = (u + 10, v - 10)
            cv2.putText(vis, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(vis, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0),   1, cv2.LINE_AA)

        # Show
        cv2.imshow('result_image', cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        cv2.imshow('depth_image',  depth_color)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = AprilTagDetectNode()
    node._publish_arm(node.init_joints, runtime=1500)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
