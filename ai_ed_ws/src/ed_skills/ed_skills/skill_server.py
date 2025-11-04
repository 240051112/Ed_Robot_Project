#!/usr/bin/env python3
import time, math
from typing import Dict, Optional, Tuple, List

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, GoalResponse

from sensor_msgs.msg import JointState
from dofbot_pro_interface.msg import ArmJoint, AprilTagInfo
from dofbot_pro_interface.action import SkillExecution

def clamp(v, lo, hi): return max(lo, min(hi, v))

class SkillServer(Node):
    def __init__(self):
        super().__init__('ed_skill_server')

        # ---------- Safe poses (deg)
        # observe is higher to avoid sweeping into the base while yawing
        self.declare_parameter('home_angles',    [90.0, 145.0, 30.0, 0.0, 90.0, 45.0])
        self.declare_parameter('observe_angles', [90.0, 145.0, 30.0, 0.0, 90.0, 90.0])

        # ---------- Camera center (pixels)
        self.declare_parameter('cx', 319.5)    # if your color stream is 640x480
        self.declare_parameter('cy', 239.5)

        # ---------- Scan limits (joint1 only)
        # keep base sweep moderate
        self.declare_parameter('scan_j1_min_deg',  60.0)
        self.declare_parameter('scan_j1_max_deg', 120.0)
        self.declare_parameter('scan_step_deg',     8.0)
        self.declare_parameter('scan_dwell_sec',    0.45)
        self.declare_parameter('detection_hits_required', 3)
        self.declare_parameter('detection_fresh_sec',     2.0)

        # ---------- Visual servo parameters (joint1 only)
        # AUTO sign (0) will test the first correction; set to +1 / -1 to force
        self.declare_parameter('j1_pixel_gain_deg_per_px', 0.04)   # 100 px -> 4 deg
        self.declare_parameter('j1_pixel_sign', 0.0)               # 0 = auto decide on first move
        self.declare_parameter('j1_rate_limit_deg', 6.0)           # max change per command
        self.declare_parameter('center_tol_px', 12)
        self.declare_parameter('center_consec_ok', 6)
        self.declare_parameter('center_timeout_sec', 25.0)

        # Slow & gentle motion
        self.declare_parameter('runtime_ms', 3000)

        # ---------- State
        self.home_angles    : List[float] = [float(x) for x in self.get_parameter('home_angles').value]
        self.observe_angles : List[float] = [float(x) for x in self.get_parameter('observe_angles').value]
        self.current_joints : List[Optional[float]] = [None]*6
        self.cx = float(self.get_parameter('cx').value)
        self.apriltag_detections: Dict[int, AprilTagInfo] = {}
        self.apriltag_last_seen : Dict[int, float] = {}

        # ---------- ROS I/O
        self.arm_pub = self.create_publisher(ArmJoint, '/TargetAngle', 10)
        self.create_subscription(JointState,   '/ArmAngleUpdate', self._joint_state_cb, 10)
        self.create_subscription(AprilTagInfo, '/PosInfo',        self._tag_cb,         10)

        self._as = ActionServer(
            self, SkillExecution, 'ed_skill_server',
            goal_callback=self.goal_cb, execute_callback=self.execute_cb
        )
        self.get_logger().info("Skill server ready (center_tag, yaw-only).")

    # ----- Callbacks
    def _joint_state_cb(self, msg: JointState):
        idx = {n:i for i,n in enumerate(msg.name)}
        for i in range(6):
            j = idx.get(f'joint{i+1}')
            if j is not None and j < len(msg.position):
                self.current_joints[i] = float(msg.position[j])

    def _tag_cb(self, msg: AprilTagInfo):
        self.apriltag_detections[msg.id] = msg
        self.apriltag_last_seen[msg.id]  = self._now()

    def goal_cb(self, goal_request):
        self.get_logger().info(f"Goal: {goal_request.skill_name}")
        return GoalResponse.ACCEPT

    def _now(self) -> float:
        s, ns = self.get_clock().now().seconds_nanoseconds()
        return s + ns*1e-9

    # ----- Helpers
    def _joints(self, fb: List[float]) -> List[float]:
        return [self.current_joints[i] if self.current_joints[i] is not None else fb[i] for i in range(6)]

    def _publish(self, joints: List[float], rt_ms: Optional[int] = None):
        msg = ArmJoint()
        msg.run_time = int(rt_ms if rt_ms is not None else int(self.get_parameter('runtime_ms').value))
        # send exactly what we intend, but clamp to servo bounds
        msg.joints = [float(clamp(j, 0.0, 180.0)) for j in joints]
        self.arm_pub.publish(msg)
        self.get_logger().info(f"CMD { [round(x,1) for x in msg.joints] }  rt={msg.run_time}ms")

    def _fresh(self, tag_id: int) -> Optional[AprilTagInfo]:
        t = self.apriltag_last_seen.get(tag_id)
        if t is None: return None
        if self._now() - t > float(self.get_parameter('detection_fresh_sec').value):
            return None
        return self.apriltag_detections.get(tag_id)

    def _observe_pose(self):
        self._publish(self.observe_angles, 3000)
        time.sleep(3.2)  # wait for motion + first JointState updates

    # ----- Scan (joint1 only)
    def _scan_for_tag(self, tag_id: int) -> bool:
        jmin = float(self.get_parameter('scan_j1_min_deg').value)
        jmax = float(self.get_parameter('scan_j1_max_deg').value)
        step = float(self.get_parameter('scan_step_deg').value)
        dwell= float(self.get_parameter('scan_dwell_sec').value)
        hits_req = int(self.get_parameter('detection_hits_required').value)

        def detect_here() -> bool:
            hits = 0; t0 = self._now()
            while self._now() - t0 < dwell:
                if self._fresh(tag_id) is not None:
                    hits += 1
                    if hits >= hits_req: return True
                time.sleep(0.05)
            return False

        def sweep(a, b, s):
            j1 = a
            while (s>0 and j1<=b) or (s<0 and j1>=b):
                js = self._joints(self.observe_angles)
                js[0] = clamp(j1, jmin, jmax)             # ONLY joint1 changes
                self._publish(js, 800)
                if detect_here():
                    self.get_logger().info(f"Scan hit at j1={js[0]:.1f}°")
                    return True
                j1 += s
            return False

        if sweep(jmin, jmax, +step): return True
        if sweep(jmax, jmin, -step): return True
        return False

    # ----- Centering (joint1 only, with auto-sign & rate limit)
    def _center_tag(self, tag_id: int) -> Tuple[bool, str]:
        self._observe_pose()

        if self._fresh(tag_id) is None:
            self.get_logger().info(f"Scanning for tag {tag_id}…")
            if not self._scan_for_tag(tag_id):
                return False, f"Tag {tag_id} not found after scan."

        gain   = float(self.get_parameter('j1_pixel_gain_deg_per_px').value)
        sign   = float(self.get_parameter('j1_pixel_sign').value)   # 0.0 means auto
        tol    = float(self.get_parameter('center_tol_px').value)
        needok = int(self.get_parameter('center_consec_ok').value)
        timeout= float(self.get_parameter('center_timeout_sec').value)
        rate   = float(self.get_parameter('j1_rate_limit_deg').value)
        jmin   = float(self.get_parameter('scan_j1_min_deg').value)
        jmax   = float(self.get_parameter('scan_j1_max_deg').value)

        cx = self.cx
        t0 = self._now()
        ok_count = 0
        auto_sign_done = (sign != 0.0)

        while self._now() - t0 < timeout:
            tag = self._fresh(tag_id)
            if tag is None:
                ok_count = 0
                time.sleep(0.06)
                continue

            ex = float(tag.x) - cx      # +ve if tag is to the right
            self.get_logger().info(f"Center loop: px error = {ex:.1f}")

            if abs(ex) <= tol:
                ok_count += 1
                if ok_count >= needok:
                    return True, "Centered."
                time.sleep(0.10)
                continue

            js = self._joints(self.observe_angles)
            j1_now = float(js[0])

            # propose a step (limited)
            step = gain * ex
            step = clamp(step, -rate, +rate)

            # sign auto-discovery once, using a tiny test nudge
            if not auto_sign_done:
                test = clamp(j1_now - step, jmin, jmax)
                js_test = js[:]; js_test[0] = test
                self._publish(js_test, 700)
                time.sleep(0.45)
                tag2 = self._fresh(tag_id)
                if tag2 is not None:
                    ex2 = float(tag2.x) - cx
                    # If error got worse in magnitude, flip sign
                    if abs(ex2) > abs(ex):
                        sign = -1.0
                        self.get_logger().warn("Auto-set j1_pixel_sign = -1 (inverted).")
                    else:
                        sign = +1.0
                        self.get_logger().info("Auto-set j1_pixel_sign = +1.")
                else:
                    # couldn’t check, default to +1
                    sign = +1.0
                auto_sign_done = True
                continue

            # apply correction with decided sign, within yaw window
            j1_cmd = clamp(j1_now - sign * step, jmin, jmax)  # minus: turning right moves image left
            js[0] = j1_cmd
            self._publish(js, 900)
            time.sleep(0.35)

        return False, "Timeout centering tag."

    # ----- Action entry point
    def execute_cb(self, goal_handle):
        req = goal_handle.request
        skill_in = (req.skill_name or "").strip()
        name, arg = (skill_in.split(':', 1) + [None])[:2]
        name = (name or "").lower().strip()
        arg  = arg.strip() if arg else None

        result = SkillExecution.Result()
        try:
            if name == 'center_tag':
                tag_id = int(arg) if arg else 1
                ok, msg = self._center_tag(tag_id)
                if ok: goal_handle.succeed()
                else:  goal_handle.abort()
                result.success = bool(ok)
                result.message = msg
                return result

            # All other skills disabled on purpose for safety
            goal_handle.abort()
            result.success = False
            result.message = f"Unknown skill '{name}'. Try center_tag:ID"
            return result

        except Exception as e:
            self.get_logger().error(f"Skill '{skill_in}' crashed: {e}")
            goal_handle.abort()
            result.success = False
            result.message = f"Skill crashed: {e}"
            return result

def main(args=None):
    rclpy.init(args=args)
    node = SkillServer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()