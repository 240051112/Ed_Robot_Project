#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from vision_msgs.msg import Detection3DArray
from geometry_msgs.msg import Point
from vision_msgs.msg import BoundingBox2D, Pose2D, Point2D
from dofbot_pro_interface.msg import DofbotDetection3DArray, DofbotDetection3D

class Bridge(Node):
    def __init__(self):
        super().__init__('ed_detections_bridge')
        self.sub = self.create_subscription(
            Detection3DArray, '/ed/detections_3d', self.cb, 10
        )
        self.pub = self.create_publisher(
            DofbotDetection3DArray, '/ed/detections_3d_custom', 10
        )
        self.msg_count = 0
        self.get_logger().info("Bridge: /ed/detections_3d (vision_msgs) -> /ed/detections_3d_custom (dofbot_pro_interface)")

    def cb(self, msg: Detection3DArray):
        out = DofbotDetection3DArray()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = msg.header.frame_id

        for det in msg.detections:
            dd = DofbotDetection3D()

            # class + score
            cls, score = 'object', 0.0
            try:
                if det.results:
                    best = max(det.results, key=lambda h: getattr(getattr(h, 'hypothesis', h), 'score', 0.0))
                    hyp = getattr(best, 'hypothesis', best)
                    cls = str(getattr(hyp, 'class_id', getattr(hyp, 'id', 'object')))
                    score = float(getattr(hyp, 'score', 0.0))
            except Exception:
                pass
            dd.class_name = cls
            dd.score = score

            # 3D position from bbox center
            try:
                p = det.bbox.center.position
                dd.position = Point(x=float(p.x), y=float(p.y), z=float(p.z))
            except Exception:
                dd.position = Point()

            # filler 2D bbox
            bb = BoundingBox2D()
            bb.center = Pose2D(position=Point2D(x=0.0, y=0.0), theta=0.0)
            bb.size_x = 0.0
            bb.size_y = 0.0
            dd.bbox = bb

            out.detections.append(dd)

        self.pub.publish(out)
        self.msg_count += 1
        if self.msg_count % 10 == 0:
            self.get_logger().info(f"Published {self.msg_count} msgs to /ed/detections_3d_custom (frame={out.header.frame_id})")

def main():
    rclpy.init()
    node = Bridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
