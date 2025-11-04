#!/bin/bash
echo "🚀 Activating Ed Environment...";
source /opt/ros/humble/setup.bash;
source ~/jetson_phi3_env/bin/activate;
source ~/drivers_ws/install/setup.bash;
source ~/ai_ed_ws/install/setup.bash;
echo "✅ Ed Environment is ready.";