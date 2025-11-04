#!/usr/bin/env bash
set -e

# 0 — prerequisites ----------------------------------------------------------
source /opt/ros/humble/setup.bash   # make sure Humble is in the PATH

# 1 — workspace skeleton ----------------------------------------------------
WS=~/ai_ed_ws
echo "🔄  Re-creating workspace at $WS"
rm -rf "$WS"
mkdir -p "$WS/src"
cd "$WS"

# 2 — create five empty Python packages -------------------------------------
for pkg in bringup skills action_orchestrator rag_service voice_interface; do
  ros2 pkg create --build-type ament_python "$pkg" \
        --license Apache-2.0 \
        --dependencies rclpy std_msgs \
        --destination-directory src
done

# 3 — skills: add one micro-skill ------------------------------------------
SK=src/skills/skills
mkdir -p "$SK"
echo '"""Micro-skills package."""' > "$SK/__init__.py"

cat > "$SK/move_joint.py" << 'PY'
#!/usr/bin/env python3
"""
A heartbeat-only move_joint micro-skill.
Replace the timer callback with real trajectory code whenever you're ready.
"""
import rclpy
from rclpy.node import Node

class MoveJoint(Node):
    def __init__(self):
        super().__init__("move_joint_skill")
        self.create_timer(2.0, self._beat)

    def _beat(self):
        self.get_logger().info("move_joint_skill alive")

def main():
    rclpy.init()
    node = MoveJoint()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
PY
chmod +x "$SK/move_joint.py"

# 4 — action_orchestrator: dispatcher node -----------------------------------
AO=src/action_orchestrator/action_orchestrator
mkdir -p "$AO"
echo '"""Dispatcher package."""' > "$AO/__init__.py"

cat > "$AO/dispatcher_node.py" << 'PY'
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class Dispatcher(Node):
    def __init__(self):
        super().__init__("dispatcher")
        self.create_subscription(String, "robot_command", self._cb, 10)

    def _cb(self, msg):
        self.get_logger().info(f'[Dispatcher] received: "{msg.data}"')
        # TODO: turn the message into a ROS 2 action/service call to a micro-skill.

def main():
    rclpy.init()
    node = Dispatcher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
PY
chmod +x "$AO/dispatcher_node.py"

# 5 — rag_service: FastAPI stub ----------------------------------------------
RS=src/rag_service/rag_service
mkdir -p "$RS"
echo '"""RAG FastAPI stub."""' > "$RS/__init__.py"

cat > "$RS/api.py" << 'PY'
from fastapi import FastAPI
app = FastAPI()
@app.get("/ping")
def ping():
    return {"pong": True}
PY

# 6 — voice_interface: stub node ---------------------------------------------
VI=src/voice_interface/voice_interface
mkdir -p "$VI"
echo '"""Voice interface stub."""' > "$VI/__init__.py"
printf '#!/usr/bin/env python3\nprint("voice_node stub")\n' > "$VI/voice_node.py"
chmod +x "$VI/voice_node.py"

# 7 — bringup: master launch file --------------------------------------------
BL=src/bringup/bringup/launch
mkdir -p "$BL"

cat > "$BL/ed_agent.launch.py" << 'PY'
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(package="skills", executable="move_joint.py",   name="move_joint_skill"),
        Node(package="action_orchestrator", executable="dispatcher_node.py", name="dispatcher"),
        # Add rag_service & voice_interface when they are ready
    ])
PY

# 8 — helper test publisher ---------------------------------------------------
mkdir -p scripts
cat > scripts/test_command_publisher.py << 'PY'
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class CmdPub(Node):
    def __init__(self):
        super().__init__("test_pub")
        self.pub = self.create_publisher(String, "robot_command", 10)
        self.create_timer(2.0, self.fire)

    def fire(self):
        self.pub.publish(String(data="move_joint"))
        self.get_logger().info('sent "move_joint"')

rclpy.init()
rclpy.spin(CmdPub())
rclpy.shutdown()
PY
chmod +x scripts/test_command_publisher.py

# 9 — build -------------------------------------------------------------------
echo "🔧  colcon build --symlink-install"
colcon build --symlink-install

echo -e "\n✅  Workspace ready!\nNext steps:\n"
echo "1) source ~/ai_ed_ws/install/setup.bash"
echo "2) ros2 launch bringup ed_agent.launch.py            # Terminal 1"
echo "3) python3 ~/ai_ed_ws/scripts/test_command_publisher.py  # Terminal 2"
echo -e "\nOpen ~/ai_ed_ws in VS Code to hack on each package.\n"
