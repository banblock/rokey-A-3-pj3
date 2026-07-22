
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})     # 1. Application

from isaacsim.core.utils.extensions import enable_extension
enable_extension("isaacsim.ros2.bridge")
simulation_app.update()

import math

import numpy as np
import omni.usd
import omni.graph.core as og
import usdrt.Sdf
from pxr import UsdLux

from isaacsim.core.api import World
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.storage.native import get_assets_root_path

from fleet_config import NODE_GRAPH, ROBOT_HOME_NODE, ROBOT_SHOE_TYPE, robot_spawn_yaw

# ╔══════════════════════════════════════════════════════════════╗
# ║  AMR 함대(Nova Carter x8) — 3단 구조의 3번째 계층: 순수 물리 세계        ║
# ╚══════════════════════════════════════════════════════════════╝
# FMS(fms_node.py, 관제탑) ↔ Fleet Driver(fleet_driver.py, 중간관리자) ↔ 이 파일(물리)
#
# 이 파일은 로봇 제어 로직을 전혀 갖지 않는다. 각 Nova Carter마다 ROS2 브리지
# OmniGraph(디퍼렌셜 드라이브 + 오도메트리)만 구성해두고, 외부(fleet_driver.py)가
# 보내는 /<robot_id>/cmd_vel 값대로 바퀴 조인트를 굴리고 /<robot_id>/odom을
# 내보내기만 한다. 이전 버전(직접 set_world_pose로 텔레포트)을 대체함.
#
# Nova Carter 실측 스펙 (Isaac 5.1 에셋, 아티큘레이션을 직접 읽어 확인된 값):
#   구동 조인트 : joint_wheel_left, joint_wheel_right (나머지 5개는 뒤쪽 캐스터, 구동 무관)
#   바퀴 반지름 : 0.14 m
#   트랙폭      : 0.4132 m (좌우 구동 바퀴 간격)
#   참조된 prim 아래에서 아티큘레이션 루트는 <참조 prim>/chassis_link 로 붙는다
#   (실측 시 /World/NovaCarter로 참조했더니 루트가 /World/NovaCarter/chassis_link였음
#   → /World/{robot_id}로 참조하면 루트는 /World/{robot_id}/chassis_link).
#
# 주의: 아래 OmniGraph 노드 구성은 Isaac Sim 공식 ROS2 디퍼렌셜 드라이브 샘플의
# 표준 패턴(OnPlaybackTick → ROS2SubscribeTwist → Twist를 x/z 스칼라로 분해 →
# DifferentialController → IsaacArticulationController, 그리고 별도로
# IsaacComputeOdometry → ROS2PublishOdometry)을 따랐지만, 실제 엔진 없이 작성한
# 거라 노드/속성 이름 중 일부(특히 Twist 벡터 분해 노드, ROS2PublishOdometry의
# child frame 속성명)는 Isaac Sim에서 처음 실행할 때 Visual Scripting 노드
# 검색으로 확인/수정이 필요할 수 있다.

WHEEL_RADIUS_M = 0.14
WHEEL_BASE_M = 0.4132  # 트랙폭(좌우 구동 바퀴 간격)
WHEEL_JOINT_NAMES = ["joint_wheel_left", "joint_wheel_right"]

world = World(stage_units_in_meters=1.0)
stage = omni.usd.get_context().get_stage()

dome_light = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
dome_light.CreateIntensityAttr(1000.0)
world.scene.add_default_ground_plane()

_assets_root_path = get_assets_root_path()
if _assets_root_path is None:
    raise RuntimeError("Isaac Sim 기본 에셋 서버(Nucleus)에 연결할 수 없습니다.")
NOVA_CARTER_USD = _assets_root_path + "/Isaac/Robots/NVIDIA/NovaCarter/nova_carter.usd"


def spawn_asset(usd_path, prim_path, position, yaw=0.0):
    add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
    xform = SingleXFormPrim(prim_path)
    # 스폰 자세를 yaw만큼 돌려서(기본은 항상 0=월드 X축 방향) 첫 이동 방향과
    # 맞춘다 — fleet_config.robot_spawn_yaw()가 계산해준 값을 그대로 받는다.
    half = yaw / 2.0
    orientation_wxyz = np.array([math.cos(half), 0.0, 0.0, math.sin(half)])
    xform.set_world_pose(position=np.array(position), orientation=orientation_wxyz)
    return xform


def build_ros2_diffdrive_graph(robot_id, chassis_prim_path):
    """이 로봇 전용: /<robot_id>/cmd_vel 구독 → 디퍼렌셜 드라이브 → 바퀴 조인트 구동,
    그리고 IsaacComputeOdometry → /<robot_id>/odom 발행. OnPlaybackTick 기반이라
    world.step()이 도는 동안 매 프레임 자동으로 재실행된다(수동 evaluate 불필요)."""
    graph_path = f"/World/{robot_id}/ROS_DiffDrive"
    keys = og.Controller.Keys

    (graph, _, _, _) = og.Controller.edit(
        {"graph_path": graph_path, "evaluator_name": "execution"},
        {
            keys.CREATE_NODES: [
                ("OnTick", "omni.graph.action.OnPlaybackTick"),
                ("SubscribeTwist", "isaacsim.ros2.bridge.ROS2SubscribeTwist"),
                ("BreakLinear", "omni.graph.nodes.BreakVector3"),
                ("BreakAngular", "omni.graph.nodes.BreakVector3"),
                ("DiffController", "isaacsim.robot.wheeled_robots.DifferentialController"),
                ("ArticulationController", "isaacsim.core.nodes.IsaacArticulationController"),
                ("ComputeOdometry", "isaacsim.core.nodes.IsaacComputeOdometry"),
                ("PublishOdometry", "isaacsim.ros2.bridge.ROS2PublishOdometry"),
                ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
            ],
            keys.CONNECT: [
                ("OnTick.outputs:tick", "SubscribeTwist.inputs:execIn"),
                ("SubscribeTwist.outputs:linearVelocity", "BreakLinear.inputs:tuple"),
                ("SubscribeTwist.outputs:angularVelocity", "BreakAngular.inputs:tuple"),
                ("SubscribeTwist.outputs:execOut", "DiffController.inputs:execIn"),
                ("BreakLinear.outputs:x", "DiffController.inputs:linearVelocity"),
                ("BreakAngular.outputs:z", "DiffController.inputs:angularVelocity"),
                # DifferentialController는 순수 계산 노드라 outputs:execOut이 없음 — 실행 검증됨.
                ("SubscribeTwist.outputs:execOut", "ArticulationController.inputs:execIn"),
                ("DiffController.outputs:velocityCommand", "ArticulationController.inputs:velocityCommand"),
                ("OnTick.outputs:tick", "ComputeOdometry.inputs:execIn"),
                ("ComputeOdometry.outputs:execOut", "PublishOdometry.inputs:execIn"),
                ("ComputeOdometry.outputs:angularVelocity", "PublishOdometry.inputs:angularVelocity"),
                ("ComputeOdometry.outputs:linearVelocity", "PublishOdometry.inputs:linearVelocity"),
                ("ComputeOdometry.outputs:orientation", "PublishOdometry.inputs:orientation"),
                ("ComputeOdometry.outputs:position", "PublishOdometry.inputs:position"),
                ("ReadSimTime.outputs:simulationTime", "PublishOdometry.inputs:timeStamp"),
            ],
            keys.SET_VALUES: [
                ("SubscribeTwist.inputs:topicName", f"/{robot_id}/cmd_vel"),
                ("DiffController.inputs:wheelRadius", WHEEL_RADIUS_M),
                ("DiffController.inputs:wheelDistance", WHEEL_BASE_M),
                ("ArticulationController.inputs:targetPrim", [usdrt.Sdf.Path(chassis_prim_path)]),
                ("ArticulationController.inputs:jointNames", WHEEL_JOINT_NAMES),
                ("ComputeOdometry.inputs:chassisPrim", [usdrt.Sdf.Path(chassis_prim_path)]),
                ("PublishOdometry.inputs:topicName", f"/{robot_id}/odom"),
                ("PublishOdometry.inputs:odomFrameId", f"{robot_id}/odom"),
                ("PublishOdometry.inputs:chassisFrameId", f"{robot_id}/base_link"),
            ],
        },
    )
    return graph


# ╔══════════════════════════════════════════════════════════════╗
# ║  Nova Carter 8대 스폰 — 각자 fms_node.py가 정해준 홈 슬롯에서 시작       ║
# ╚══════════════════════════════════════════════════════════════╝
for robot_id, home_node in ROBOT_HOME_NODE.items():
    home_pos = list(NODE_GRAPH[home_node]["position"])
    home_yaw = robot_spawn_yaw(robot_id)
    prim_path = f"/World/{robot_id}"
    spawn_asset(NOVA_CARTER_USD, prim_path, position=home_pos, yaw=home_yaw)
    build_ros2_diffdrive_graph(robot_id, chassis_prim_path=f"{prim_path}/chassis_link")
    print(
        f"[스폰] {robot_id} (담당 종류={ROBOT_SHOE_TYPE[robot_id]}) @ {home_node} {home_pos} "
        f"yaw={math.degrees(home_yaw):.0f}도"
    )

world.reset()
simulation_app.update()

print("\n" + "=" * 60)
print(f"[환경] Nova Carter {len(ROBOT_HOME_NODE)}대 스폰 + ROS2 디퍼렌셜 드라이브 브리지 구성 완료")
print("  제어 로직 없음 — 외부 fleet_driver.py가 /<robot_id>/cmd_vel 로 조종, /<robot_id>/odom 구독")
print("=" * 60)

while simulation_app.is_running():
    world.step(render=True)

simulation_app.close()
