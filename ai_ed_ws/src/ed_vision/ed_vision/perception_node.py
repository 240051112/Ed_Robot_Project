#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.duration import Duration
from rclpy.time import Time
from rcl_interfaces.msg import SetParametersResult, ParameterDescriptor, ParameterType

from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
from message_filters import Subscriber, ApproximateTimeSynchronizer

from ultralytics import YOLO
import numpy as np
import cv2

from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesis, ObjectHypothesisWithPose, BoundingBox2D, Pose2D, Point2D
from geometry_msgs.msg import PointStamped
import tf2_ros
from tf2_geometry_msgs import do_transform_point

# Custom interface
from dofbot_pro_interface.msg import DofbotDetection3DArray, DofbotDetection3D


def median_depth(cv_depth, u, v, w=5):
    h, W = cv_depth.shape[:2]
    half = w // 2
    u0, u1 = max(0, u - half), min(W, u + half + 1)
    v0, v1 = max(0, v - half), min(h, v + half + 1)
    patch = cv_depth[v0:v1, u0:u1].astype(np.float32)
    if patch.size == 0:
        return 0.0
    vals = patch[patch > 0.0]
    if vals.size == 0:
        return 0.0
    return float(np.median(vals))


class PerceptionNodeV2(Node):
    def __init__(self):
        super().__init__('ed_perception_node')

        # --- Core detector params ---
        self.declare_parameter('model_path', '/home/jetson/ultralytics/ultralytics/yolo11n.engine')
        self.declare_parameter('confidence_threshold', 0.5)   # base conf
        self.declare_parameter('iou_threshold', 0.6)
        self.declare_parameter('imgsz', 640)
        self.declare_parameter('show_video', True)
        self.declare_parameter('depth_scale', 0.001)          # mm -> m
        self.declare_parameter('depth_window', 5)

        # --- Force STRING_ARRAY (fixes BYTE_ARRAY issue); ['__all__'] means no filtering ---
        self.declare_parameter(
            'target_classes',
            ['__all__'],
            ParameterDescriptor(type=ParameterType.PARAMETER_STRING_ARRAY)
        )

        # --- Smoothing (3D position EMA) ---
        self.declare_parameter('smooth_alpha_pos', 0.5)   # 0..1, higher = snappier
        self.declare_parameter('smooth_alpha_z', 0.4)
        self.declare_parameter('publish_smoothed', True)

        # --- Stabilization (hysteresis + persistence) ---
        self.declare_parameter('conf_on', 0.60)           # promote above this
        self.declare_parameter('conf_off', 0.48)          # demote below this
        self.declare_parameter('persist_frames', 4)       # N hits to lock stable
        self.declare_parameter('max_miss_frames', 6)      # allowed brief misses
        self.declare_parameter('min_box_area', 1200)      # ignore tiny boxes
        self.declare_parameter('publish_stable_only', True)
        self.declare_parameter('pub_stable_topic', '/ed/detections_3d_stable')

        # --- Topics / frames ---
        self.declare_parameter('output_frame', '')  # set to 'base_link' if you run a camera→base_link TF
        self.declare_parameter('color_topic', '/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/depth/image_raw')
        self.declare_parameter('info_topic',  '/camera/color/camera_info')
        self.declare_parameter('pub_custom_topic', '/ed/detections_3d_custom')
        self.declare_parameter('pub_vision_topic', '/ed/detections_3d')
        self.declare_parameter('pub_overlay_topic', '/ed/overlay/image')

        # --- Read params ---
        model_path         = self.get_parameter('model_path').value
        self.base_conf     = float(self.get_parameter('confidence_threshold').value)
        self.iou_thr       = float(self.get_parameter('iou_threshold').value)
        self.imgsz         = int(self.get_parameter('imgsz').value)
        self.show_video    = bool(self.get_parameter('show_video').value)
        self.depth_scale   = float(self.get_parameter('depth_scale').value)
        self.depth_window  = int(self.get_parameter('depth_window').value)

        tc_val = self.get_parameter('target_classes').value
        tc_list = [str(x) for x in (tc_val if isinstance(tc_val, (list, tuple)) else [])]
        # [] means filter disabled (ALL)
        self.target_classes = [] if tc_list == ['__all__'] else tc_list

        self.alpha_pos     = float(self.get_parameter('smooth_alpha_pos').value)
        self.alpha_z       = float(self.get_parameter('smooth_alpha_z').value)
        self.publish_smoothed = bool(self.get_parameter('publish_smoothed').value)

        self.conf_on       = float(self.get_parameter('conf_on').value)
        self.conf_off      = float(self.get_parameter('conf_off').value)
        self.persist_frames = int(self.get_parameter('persist_frames').value)
        self.max_miss      = int(self.get_parameter('max_miss_frames').value)
        self.min_box_area  = int(self.get_parameter('min_box_area').value)
        self.publish_stable_only = bool(self.get_parameter('publish_stable_only').value)
        self.pub_stable_topic = str(self.get_parameter('pub_stable_topic').value)

        self.output_frame  = str(self.get_parameter('output_frame').value)
        color_topic        = self.get_parameter('color_topic').value
        depth_topic        = self.get_parameter('depth_topic').value
        info_topic         = self.get_parameter('info_topic').value
        self.pub_custom_topic = self.get_parameter('pub_custom_topic').value
        self.pub_vision_topic = self.get_parameter('pub_vision_topic').value
        overlay_topic      = self.get_parameter('pub_overlay_topic').value

        # --- Model ---
        self.model = YOLO(model_path, task='detect')
        self.get_logger().info(f'Loaded YOLO model: {model_path}')

        # --- Utils ---
        self.bridge = CvBridge()
        self.camera_intrinsics = None
        self.camera_frame = None

        # --- State: 3D EMA smoothing + per-class tracker ---
        # tracks[name] = { 'seen': int, 'miss': int, 'active': bool,
        #                  'px': float, 'py': float, 'pz': float,
        #                  'bbox': (x1,y1,x2,y2), 'score': float }
        self.smooth = {}
        self.tracks = {}

        # --- TF ---
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # --- Subscriptions ---
        self.sub_color = Subscriber(self, Image, color_topic, qos_profile=qos_profile_sensor_data)
        self.sub_depth = Subscriber(self, Image, depth_topic, qos_profile=qos_profile_sensor_data)
        self.sub_info  = Subscriber(self, CameraInfo, info_topic, qos_profile=qos_profile_sensor_data)
        self.sync = ApproximateTimeSynchronizer([self.sub_color, self.sub_depth, self.sub_info], queue_size=10, slop=0.20)
        self.sync.registerCallback(self.image_callback)

        # --- Publishers ---
        self.pub_detections_vision = self.create_publisher(Detection3DArray, self.pub_vision_topic, 10)
        self.pub_detections_custom = self.create_publisher(DofbotDetection3DArray, self.pub_custom_topic, 10)
        self.pub_detections_stable = self.create_publisher(DofbotDetection3DArray, self.pub_stable_topic, 10)
        self.pub_overlay = self.create_publisher(Image, overlay_topic, 10)

        # --- Dynamic params ---
        self.add_on_set_parameters_callback(self._on_set_params)

        self.get_logger().info(f"Target filter: {self.target_classes or 'ALL'}")
        self.get_logger().info("Ed's perception node V2 (stabilized) is running...")

    # ------------------------ Dynamic params ------------------------
    def _on_set_params(self, params):
        for p in params:
            if p.name == 'confidence_threshold':
                self.base_conf = float(p.value)
            elif p.name == 'iou_threshold':
                self.iou_thr = float(p.value)
            elif p.name == 'imgsz':
                self.imgsz = int(p.value)
            elif p.name == 'show_video':
                self.show_video = bool(p.value)
            elif p.name == 'target_classes':
                arr = [str(x) for x in list(p.value)]
                self.target_classes = [] if arr == ['__all__'] else arr
                self.get_logger().info(f"Updated target_classes -> {self.target_classes or 'ALL'}")
            elif p.name == 'smooth_alpha_pos':
                self.alpha_pos = float(p.value)
            elif p.name == 'smooth_alpha_z':
                self.alpha_z = float(p.value)
            elif p.name == 'publish_smoothed':
                self.publish_smoothed = bool(p.value)
            elif p.name == 'conf_on':
                self.conf_on = float(p.value)
            elif p.name == 'conf_off':
                self.conf_off = float(p.value)
            elif p.name == 'persist_frames':
                self.persist_frames = int(p.value)
            elif p.name == 'max_miss_frames':
                self.max_miss = int(p.value)
            elif p.name == 'min_box_area':
                self.min_box_area = int(p.value)
            elif p.name == 'publish_stable_only':
                self.publish_stable_only = bool(p.value)
        return SetParametersResult(successful=True)

    # ------------------------ EMA helper ------------------------
    def _ema(self, key, px, py, pz):
        s = self.smooth.get(key)
        if s is None:
            s = {'px': px, 'py': py, 'pz': pz}
        else:
            s['px'] = self.alpha_pos * px + (1.0 - self.alpha_pos) * s['px']
            s['py'] = self.alpha_pos * py + (1.0 - self.alpha_pos) * s['py']
            s['pz'] = self.alpha_z   * pz + (1.0 - self.alpha_z)   * s['pz']
        self.smooth[key] = s
        return s['px'], s['py'], s['pz']

    # ------------------------ Main callback ------------------------
    def image_callback(self, color_msg, depth_msg, info_msg):
        try:
            cv_color = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
            cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        except CvBridgeError as e:
            self.get_logger().error(f'Image conversion error: {e}')
            return

        if self.camera_intrinsics is None:
            self.camera_intrinsics = {'fx': info_msg.k[0], 'fy': info_msg.k[4], 'cx': info_msg.k[2], 'cy': info_msg.k[5]}
            self.camera_frame = info_msg.header.frame_id or color_msg.header.frame_id
            self.get_logger().info(f"Camera frame: {self.camera_frame}; intrinsics set.")

        # Inference
        results = self.model(cv_color, conf=self.base_conf, iou=self.iou_thr, imgsz=self.imgsz, verbose=False)
        boxes = results[0].boxes if results and results[0].boxes is not None else []

        # Gather best candidate per class (highest score), ignore tiny boxes
        candidates = {}  # name -> dict(score, xyxy, uv)
        names = results[0].names if results else {}
        if boxes:
            for b in boxes:
                score = float(b.conf[0])
                cls_id = int(b.cls[0])
                name = names.get(cls_id, str(cls_id))

                if self.target_classes and (name not in self.target_classes):
                    continue

                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int)
                area = max(1, (x2 - x1)) * max(1, (y2 - y1))
                if area < self.min_box_area:
                    continue

                u, v = int((x1 + x2) / 2), int((y1 + y2) / 2)
                cur = candidates.get(name)
                if (cur is None) or (score > cur['score']):
                    candidates[name] = {'score': score, 'xyxy': (x1, y1, x2, y2), 'uv': (u, v)}

        # Depth & 3D; update tracks
        K = self.camera_intrinsics
        for name, c in list(candidates.items()):
            u, v = c['uv']
            d_raw = median_depth(cv_depth, u, v, w=self.depth_window)
            z_m = d_raw * self.depth_scale
            if z_m <= 0.0:
                continue

            x_m = (u - K['cx']) * z_m / K['fx']
            y_m = (v - K['cy']) * z_m / K['fy']

            px, py, pz = x_m, y_m, z_m
            if self.output_frame.strip():
                try:
                    ps = PointStamped()
                    ps.header = color_msg.header
                    ps.point.x, ps.point.y, ps.point.z = x_m, y_m, z_m
                    tf = self.tf_buffer.lookup_transform(self.output_frame, ps.header.frame_id, Time())
                    ps_out = do_transform_point(ps, tf)
                    px, py, pz = ps_out.point.x, ps_out.point.y, ps_out.point.z
                except Exception as ex:
                    self.get_logger().warn(f"TF transform to '{self.output_frame}' failed: {ex}")

            if self.publish_smoothed:
                px, py, pz = self._ema(name, px, py, pz)

            # Hysteresis + persistence
            t = self.tracks.get(name, {'seen': 0, 'miss': 0, 'active': False, 'px': px, 'py': py, 'pz': pz, 'bbox': c['xyxy'], 'score': c['score']})
            score = c['score']

            if score >= self.conf_on:
                t['seen'] += 1
                t['miss'] = 0
                t['px'], t['py'], t['pz'] = px, py, pz
                t['bbox'] = c['xyxy']
                t['score'] = score
                if t['seen'] >= self.persist_frames:
                    t['active'] = True
            elif score <= self.conf_off:
                t['miss'] += 1
                t['seen'] = max(0, t['seen'] - 1)
                if t['miss'] > self.max_miss:
                    self.tracks.pop(name, None)
                    continue
            # else keep state

            self.tracks[name] = t

        # Build outputs
        use_stable = self.publish_stable_only and any(t.get('active', False) for t in self.tracks.values())

        vision_msg = Detection3DArray(); vision_msg.header = color_msg.header
        custom_msg = DofbotDetection3DArray(); custom_msg.header = color_msg.header
        stable_msg = DofbotDetection3DArray(); stable_msg.header = color_msg.header

        def add_det(out_header, name, score, px, py, pz, xyxy):
            det3d = Detection3D(); det3d.header = out_header
            hyp = ObjectHypothesisWithPose(hypothesis=ObjectHypothesis(class_id=name, score=score))
            hyp.pose.pose.position.x = px; hyp.pose.pose.position.y = py; hyp.pose.pose.position.z = pz
            det3d.results.append(hyp)
            det3d.bbox.center.position.x = px
            det3d.bbox.center.position.y = py
            det3d.bbox.center.position.z = pz
            vision_msg.detections.append(det3d)

            x1, y1, x2, y2 = xyxy
            u, v = int((x1 + x2) / 2), int((y1 + y2) / 2)
            cdet = DofbotDetection3D()
            cdet.class_name = name
            cdet.score = float(score)
            cdet.position.x, cdet.position.y, cdet.position.z = float(px), float(py), float(pz)
            bb = BoundingBox2D()
            bb.center = Pose2D(); bb.center.position = Point2D(x=float(u), y=float(v)); bb.center.theta = 0.0
            bb.size_x = float(max(1, x2 - x1)); bb.size_y = float(max(1, y2 - y1))
            cdet.bbox = bb
            custom_msg.detections.append(cdet)
            return cdet

        if use_stable:
            for name, t in self.tracks.items():
                if not t.get('active', False):
                    continue
                cdet = add_det(color_msg.header, name, t['score'], t['px'], t['py'], t['pz'], t['bbox'])
                stable_msg.detections.append(cdet)
        else:
            for name, t in self.tracks.items():
                add_det(color_msg.header, name, t['score'], t['px'], t['py'], t['pz'], t['bbox'])

        # Overlay
        try:
            for name, t in self.tracks.items():
                x1, y1, x2, y2 = t['bbox']
                color = (0, 255, 0) if t.get('active', False) else (0, 180, 0)
                thickness = 3 if t.get('active', False) else 1
                cv2.rectangle(cv_color, (x1, y1), (x2, y2), color, thickness)
                label = f"{name} {t['score']:.2f} {'STABLE' if t.get('active', False) else 'CAND'}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(cv_color, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
                cv2.putText(cv_color, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            self.pub_overlay.publish(self.bridge.cv2_to_imgmsg(cv_color, encoding='bgr8'))
        except Exception:
            pass

        # Publish
        if vision_msg.detections:
            self.pub_detections_vision.publish(vision_msg)
        if custom_msg.detections:
            self.pub_detections_custom.publish(custom_msg)
        if stable_msg.detections:
            self.pub_detections_stable.publish(stable_msg)

        if self.show_video:
            cv2.imshow("Ed's Vision V2 (stabilized)", cv_color)
            cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNodeV2()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == '__main__':
    main()
