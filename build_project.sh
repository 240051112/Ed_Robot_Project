#!/bin/bash
set -e
echo "🚀 Starting a clean, robust build for Ed..."
source ~/jetson_phi3_env/bin/activate

echo "▶️ Building drivers_ws..."
cd ~/drivers_ws
rm -rf build install log
colcon build

echo "▶️ Building ai_ed_ws..."
cd ~/ai_ed_ws
rm -rf build install log
colcon build

echo "✅ Build complete!"