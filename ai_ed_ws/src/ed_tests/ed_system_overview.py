#!/usr/bin/env python3
# Creates a one-page overview diagram including the external ED Brain (FastAPI LLM+RAG).
from pathlib import Path
import subprocess, shutil, os

OUT = os.path.expanduser("~/ai_ed_ws/src/ed_tests/result/ros_snapshot_fast/graphs")
Path(OUT).mkdir(parents=True, exist_ok=True)
dot = f"""
digraph ED_OVERVIEW {{
  rankdir=LR;
  fontname="Helvetica";
  node [fontname="Helvetica"];
  edge [fontname="Helvetica", fontsize=10];

  // clusters for clarity
  subgraph cluster_voice {{
    label="Sense";
    color="#dddddd"; style="rounded";
    voice[label="/ed_voice_client\\n(voice_interface)", shape=ellipse, fillcolor="#eef6ff", style=filled];
    mic[label="Mic / Wakeword", shape=box, style="rounded"];
    mic -> voice [label="audio → text"];
  }}

  subgraph cluster_brain {{
    label="Think (ED Brain)"; color="#dddddd"; style="rounded";
    brain[label="ed_core_api\\n(FastAPI • LLM+RAG)", shape=box, style="rounded,filled", fillcolor="#fff7e6"];
  }}

  subgraph cluster_ros {{
    label="ROS 2 Bus"; color="#dddddd"; style="rounded";
    orch[label="/ed_orchestrator_node\\n(action_orchestrator)", shape=ellipse, style="filled", fillcolor="#f0f8ff"];
    asrv[label="/ed_skill_server\\n(action server)", shape=ellipse, style="filled", fillcolor="#f0f8ff"];
    arm[label="/arm_driver", shape=ellipse, style="filled", fillcolor="#f0f8ff"];
    april[label="/apriltag_detect", shape=ellipse, style="filled", fillcolor="#f0f8ff"];

    speech[label="/ed/speech_command\\n(dofbot_pro_interface/msg/SpeechCommand)", shape=oval, style="filled", fillcolor="#eef6ff"];
    target[label="/TargetAngle\\n(dofbot_pro_interface/msg/ArmJoint)", shape=oval, style="filled", fillcolor="#eef6ff"];
    armupd[label="/ArmAngleUpdate\\n(sensor_msgs/msg/JointState)", shape=oval, style="filled", fillcolor="#eef6ff"];
    det3d[label="/ed/detections_3d\\n(dofbot_pro_interface/msg/DofbotDetection3DArray)", shape=oval, style="filled", fillcolor="#eef6ff"];

    // topic wiring (pub blue, sub orange)
    voice -> speech [color="#4e79a7", label="pub"];
    speech -> orch   [color="#f28e2b", label="sub"];

    orch -> target   [color="#4e79a7", label="pub"];
    target -> arm    [color="#f28e2b", label="sub"];

    arm -> armupd    [color="#4e79a7", label="pub"];

    april -> det3d   [color="#4e79a7", label="pub"];
    det3d -> orch    [color="#f28e2b", label="sub"];

    // action lines (server teal, feedback gold)
    orch -> asrv [color="#76b7b2", label="/ed_skill_server (goal)"];
    asrv -> orch [color="#edc948", label="status/feedback/result"];
  }}

  // show the non-ROS HTTP hop into the brain
  orch -> brain [style=dashed, color="#666666", label="HTTP POST /query"];
  brain -> orch [style=dashed, color="#666666", label="JSON (cmd/answer)"];

  // legend
  legend [shape=none, margin=0, label=<
    <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">
      <TR><TD COLSPAN="2"><B>Legend</B></TD></TR>
      <TR><TD>Publish (node→topic)</TD><TD BGCOLOR="#4e79a7"></TD></TR>
      <TR><TD>Subscribe (topic→node)</TD><TD BGCOLOR="#f28e2b"></TD></TR>
      <TR><TD>Action goal/server</TD><TD BGCOLOR="#76b7b2"></TD></TR>
      <TR><TD>Action feedback/status</TD><TD BGCOLOR="#edc948"></TD></TR>
      <TR><TD>HTTP (non-ROS)</TD><TD>╌╌ dashed ╌╌</TD></TR>
    </TABLE>
  >];

}}
"""
dot_path = Path(OUT) / "system_overview.dot"
png_path = Path(OUT) / "system_overview.png"
dot_path.write_text(dot)
if shutil.which("dot"):
    subprocess.run(["dot", "-Tpng", str(dot_path), "-o", str(png_path)], check=False)
print(f"Wrote: {png_path}")
