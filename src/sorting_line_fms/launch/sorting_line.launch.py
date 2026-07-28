"""FMS + Fleet Driver (+ 임시 메인 컨트롤 스텁)를 한 번에 띄우는 launch 파일.

FMS를 먼저 띄우고, 나머지는 약간 지연시켜서 시작한다 — 동시에 띄우면 DDS
디스커버리가 끝나기 전에 Fleet Driver의 최초 상태 메시지가 FMS 구독 시작 전에
나가서 유실될 수 있기 때문이다(실제로 겪었던 문제).

작업 지시는 메인 컨트롤 노드가 FMS의 /control/pickup(PickupList.msg) 토픽으로 발행해서 준다. 실제
메인 컨트롤 노드가 아직 없으니, 그동안은 무작위 배치를 주기적으로 던져주는
임시 스텁(main_control_stub)을 같이 띄운다 — 실제 노드가 생기면
use_main_control_stub:=false로 끄거나 이 파일 자체를 launch에서 빼면 된다.

Isaac Sim(1_conveyor_sorter_env.py)은 여기 포함되지 않는다 — 별도로
Isaac Sim 전용 파이썬(python.sh)으로 실행해야 한다.

사용법:
  ros2 launch sorting_line_fms sorting_line.launch.py
  ros2 launch sorting_line_fms sorting_line.launch.py use_main_control_stub:=false  # 실제 메인 컨트롤 노드가 생기면
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    use_main_control_stub = LaunchConfiguration("use_main_control_stub")
    
    fms_node = Node(
        package="sorting_line_fms",
        executable="fms_node",
        name="fleet_management_system",
        output="screen",
    )

    # fleet_driver.py는 config_module 파라미터로 어떤 그래프를 쓸지 정한다
    # (안 주면 fleet_driver.py 기본값인 "fleet_config"). fms_node.py는 지금
    # fleet_config_test1을 고정 import하고 있어서, 여기서 안 맞춰주면
    # fleet_driver만 원래 운영용 fleet_config를 봐서 스폰 위치/자세가 서로
    # 어긋나 로봇이 엉뚱한 곳으로 가는 문제가 있었다 — fms_node.py가 쓰는
    # 것과 반드시 같은 config_module을 줘야 한다.
    fleet_driver = Node(
        package="sorting_line_fms",
        executable="fleet_driver",
        name="fleet_driver",
        output="screen",
        parameters=[{"config_module": "fleet_config_test1"}],
    )

    main_control_stub = Node(
        package="sorting_line_fms",
        executable="main_control_stub",
        name="main_control_stub",
        output="screen",
        condition=IfCondition(use_main_control_stub),
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "use_main_control_stub",
            default_value="false",
            description="무작위 배치를 주기적으로 던져주는 임시 메인 컨트롤 스텁을 같이 띄울지 여부 — 실제 메인 컨트롤 노드가 생기면 false로",
        ),
        fms_node,
        # FMS가 구독을 다 마칠 시간을 준 뒤 나머지 시작 (초기 등록 메시지 유실 방지)
        TimerAction(period=2.0, actions=[fleet_driver, main_control_stub]),
    ])
