"""edge_conflict(구간 충돌) 회피 로직만 단독으로 확인하기 위한 launch 파일.

운영용 sorting_line.launch.py와 달리 fms_node/main_control_stub은 전혀 안
띄우고, crossing_test_fms(X자 교차 그래프에서 로봇 2대를 영원히 왕복시키는
테스트 전용 FMS)와 fleet_driver(config_module 파라미터로 crossing_test_config
그래프를 쓰도록 지정)만 띄운다.

Isaac Sim 쪽은 4_crossing_test_env.py를 별도로 python.sh로 실행해야 한다
(1_conveyor_sorter_env.py와 마찬가지로 이 launch 파일에는 포함되지 않음).

사용법:
  ros2 launch sorting_line_fms crossing_test.launch.py
"""

from launch import LaunchDescription
from launch.actions import TimerAction
from launch_ros.actions import Node


def generate_launch_description():
    crossing_test_fms = Node(
        package="sorting_line_fms",
        executable="crossing_test_fms",
        name="crossing_test_fms",
        output="screen",
    )

    fleet_driver = Node(
        package="sorting_line_fms",
        executable="fleet_driver",
        name="fleet_driver",
        output="screen",
        parameters=[{"config_module": "crossing_test_config"}],
    )

    return LaunchDescription([
        crossing_test_fms,
        # FMS가 구독을 다 마칠 시간을 준 뒤 fleet_driver 시작 (초기 상태 메시지 유실 방지)
        TimerAction(period=2.0, actions=[fleet_driver]),
    ])
