"""FMS + Fleet Driver (+ 임시 비전 스텁)를 한 번에 띄우는 launch 파일.

FMS를 먼저 띄우고, 나머지는 약간 지연시켜서 시작한다 — 동시에 띄우면 DDS
디스커버리가 끝나기 전에 Fleet Driver/비전 스텁의 최초 상태 메시지가 FMS
구독 시작 전에 나가서 유실될 수 있기 때문이다(실제로 겪었던 문제).

Isaac Sim(1_conveyor_sorter_env.py)은 여기 포함되지 않는다 — 별도로
Isaac Sim 전용 파이썬(python.sh)으로 실행해야 한다.

사용법:
  ros2 launch sorting_line_fms sorting_line.launch.py
  ros2 launch sorting_line_fms sorting_line.launch.py use_vision_stub:=false  # 실제 비전 노드 생기면
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    use_vision_stub = LaunchConfiguration("use_vision_stub")

    fms_node = Node(
        package="sorting_line_fms",
        executable="fms_node",
        name="fleet_management_system",
        output="screen",
    )

    fleet_driver = Node(
        package="sorting_line_fms",
        executable="fleet_driver",
        name="fleet_driver",
        output="screen",
    )

    vision_stub = Node(
        package="sorting_line_fms",
        executable="vision_stub",
        name="vision_stub",
        output="screen",
        condition=IfCondition(use_vision_stub),
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "use_vision_stub",
            default_value="true",
            description="임시 비전 스텁을 같이 띄울지 여부 — 실제 비전 노드가 생기면 false로",
        ),
        fms_node,
        # FMS가 구독을 다 마칠 시간을 준 뒤 나머지 시작 (초기 등록 메시지 유실 방지)
        TimerAction(period=2.0, actions=[fleet_driver, vision_stub]),
    ])
