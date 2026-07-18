설치 필요

NVIDIA의 Isaac Sim ROS 워크스페이스 저장소
home 위치에서 git clone https://github.com/isaac-sim/IsaacSim-ros_workspaces.git
클론 완료 후 colcon build --symlink-install
빌드 실패 시 의존성 설치 필요
cd ~/IsaacSim-ros_workspaces/humble_ws
rosdep install --from-paths src --ignore-src -r -y



carter_navigation.launch.py 파일에
test_warehouse_navigation.yaml 적용해서 테스트 진행


7/18
상태 토픽 추가 ("IDLE", "MOVING", "ARRIVED", "FAILED", "CANCELED")
"CANCELED"은 구현 안되어있음
이동 거리 출력 1초 -> 2초로 증가


------------------------------


AMR(Nova Carter) 동작 테스트 /navigate_to_pose만 사용

1번 터미널
isaac 실행
상단 FIle -> open -> cobot3_ws/isaacpjt/nova-carter/test_warehouse.usd

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
ros2 run commander destination_controller

4번 터미널
humble
source ~/IsaacSim-ros_workspaces/humble_ws/install/setup.bash
source ~/cobot3_ws/install/setup.bash
3개 source 후 이동 신호 발행

ros2 topic pub --once /amr/destination std_msgs/msg/String "{data: 'STORAGE'}"
ros2 topic pub --once /amr/destination std_msgs/msg/String "{data: 'DISCARD'}"
ros2 topic pub --once /amr/destination std_msgs/msg/String "{data: 'HOME'}"


----------------


