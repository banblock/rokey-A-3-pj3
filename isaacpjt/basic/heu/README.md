설치 필요

NVIDIA의 Isaac Sim ROS 워크스페이스 저장소
home 위치에서 git clone https://github.com/isaac-sim/IsaacSim-ros_workspaces.git
클론 완료 후 colcon build --symlink-install
빌드 실패 시 의존성 설치 필요
cd ~/IsaacSim-ros_workspaces/humble_ws
rosdep install --from-paths src --ignore-src -r -y

AMR(Nova Carter) 동작 테스트

1번 터미널
isaac 실행
상단 Window -> Examples -> Robotics Examples 실행
하단 Robotics Examples 에서 ROS2 -> NAVIGATION -> Nova Cater 로드

2번 터미널
humble
source ~/IsaacSim-ros_workspaces/humble_ws/install/setup.bash
2개 source 후 Nav2 실행
ros2 launch carter_navigation carter_navigation_isaacsim.launch.py

3번 터미널
humble
source ~/IsaacSim-ros_workspaces/humble_ws/install/setup.bash
source ~/cobot3_ws/install/setup.bash
3개 source 후 이동 코드 실행
ros2 run amr_transport destination_controller

4번 터미널
humble
source ~/IsaacSim-ros_workspaces/humble_ws/install/setup.bash
source ~/cobot3_ws/install/setup.bash
3개 source 후 이동 신호 발행

ros2 topic pub --once /amr/destination std_msgs/msg/String "{data: 'STORAGE'}"
ros2 topic pub --once /amr/destination std_msgs/msg/String "{data: 'DISCARD'}"
ros2 topic pub --once /amr/destination std_msgs/msg/String "{data: 'HOME'}"
