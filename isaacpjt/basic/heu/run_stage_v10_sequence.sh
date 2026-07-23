set -euo pipefail

# Change this only when Isaac Sim is installed elsewhere.
ISAAC_SIM_DIR="${ISAAC_SIM_DIR:-$HOME/dev_ws/isaac_sim/isaacsim/_build/linux-x86_64/release}"
SCRIPT_PATH="${SCRIPT_PATH:-/mnt/data/run_stage_v10_sequence.py}"

# Required for communication with ROS 2 Humble nodes in another terminal.
source /opt/ros/humble/setup.bash

export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export VISION_SERVICE_NAME="${VISION_SERVICE_NAME:-/vision/inspect}"
export USD_PATH="${USD_PATH:-/mnt/data/stage_v10(1).usd}"

exec "$ISAAC_SIM_DIR/python.sh" "$SCRIPT_PATH"