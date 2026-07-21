
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})     # 1. Application

from isaacsim.core.utils.extensions import enable_extension
enable_extension("isaacsim.ros2.bridge")
simulation_app.update()

import numpy as np
import omni.usd
import omni.graph.core as og
import usdrt.Sdf
from pxr import UsdLux

from isaacsim.core.api import World
from isaacsim.core.api.objects import VisualCuboid
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.storage.native import get_assets_root_path

from fleet_config import NODE_GRAPH, ROBOT_HOME_NODE, ROBOT_SHOE_TYPE

world = World(stage_units_in_meters=1.0)
stage = omni.usd.get_context().get_stage()

dome_light = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
dome_light.CreateIntensityAttr(1000.0)

# ╔══════════════════════════════════════════════════════════════╗
# ║  A. 파라미터                                                   ║
# ╚══════════════════════════════════════════════════════════════╝
# 이 스크립트는 Nova Carter만 스폰한다 — 컨베이어/랙/신발/폐기·임시저장소는
# 없음. 목적지 배정/이동은 AMR + FMS가 담당한다.

# ── AMR 함대 (Nova Carter x8, 실측 스펙) ──
# 제어 로직은 여기 없음 — 외부 fleet_driver.py가 /<robot_id>/cmd_vel 로 조종하고,
# 이 스크립트는 ROS2 디퍼렌셜 드라이브 브리지만 각 로봇에 붙여둔다.
WHEEL_RADIUS_M = 0.14
WHEEL_BASE_M = 0.4132  # 트랙폭(좌우 구동 바퀴 간격)
WHEEL_JOINT_NAMES = ["joint_wheel_left", "joint_wheel_right"]

_assets_root_path = get_assets_root_path()
if _assets_root_path is None:
    raise RuntimeError("Isaac Sim 기본 에셋 서버(Nucleus)에 연결할 수 없습니다.")
NOVA_CARTER_USD = _assets_root_path + "/Isaac/Robots/NVIDIA/NovaCarter/nova_carter.usd"

def spawn_asset(usd_path, prim_path, position):
    add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
    xform = SingleXFormPrim(prim_path)
    xform.set_world_pose(position=np.array(position))
    return xform


# ╔══════════════════════════════════════════════════════════════╗
# ║  C. AMR 함대 (Nova Carter x8) — 순수 물리 + ROS2 브리지               ║
# ╚══════════════════════════════════════════════════════════════╝
# fms_node.py가 정해준 로봇별 홈 슬롯(WAIT_1~8) 위치에 스폰하고, 로봇마다
# 독립된 ROS2 디퍼렌셜 드라이브 그래프를 붙인다. 이동 제어는 전부 외부
# fleet_driver.py가 /<robot_id>/cmd_vel 로 담당 — 이 스크립트는 물리 세계만 제공.
#
# 주의: 그래프 노드 구성(Twist 벡터 분해, ROS2PublishOdometry 속성명)은 Isaac Sim
# 표준 디퍼렌셜 드라이브 샘플 패턴을 따랐지만 엔진 없이 작성한 부분이 있어
# 처음 실행 시 노드/속성 이름을 확인해야 할 수 있다.


def build_ros2_diffdrive_graph(robot_id, chassis_prim_path):
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
                # ArticulationController의 트리거는 SubscribeTwist에서 바로 받는다
                # (DiffController의 데이터 출력은 같은 틱 안에서 데이터 의존성으로 먼저 계산됨).
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


# 로봇 자체(섀시)에 색을 입히지 않고, 위에 종류별 색깔 비콘을 따로 띄운다 —
# Nova Carter는 이미 자체 머티리얼이 있는 복잡한 참조 에셋이라 내부 메시의
# 재질을 직접 바꾸는 건 실패 위험이 크다. 비콘은 매 시뮬레이션 스텝마다
# 섀시의 월드 위치를 읽어 같은 위치(+ 높이 오프셋)로 따라가게 한다
# (spawn_asset()과 동일한 set_world_pose 패턴 — 부모-자식 로컬 좌표 가정 없이
# 항상 안전하게 동작).
_ROBOT_TYPE_COLORS = {
    "A": np.array([0.80, 0.20, 0.20]),  # 빨강
    "B": np.array([0.20, 0.40, 0.80]),  # 파랑
    "C": np.array([0.85, 0.65, 0.13]),  # 노랑
    "D": np.array([0.20, 0.70, 0.30]),  # 초록
}
_BEACON_HEIGHT_OFFSET_M = 0.5
_robot_chassis_prims = {}
_robot_beacons = {}

for _robot_id, _home_node in ROBOT_HOME_NODE.items():
    _home_pos = list(NODE_GRAPH[_home_node]["position"])
    _prim_path = f"/World/{_robot_id}"
    spawn_asset(NOVA_CARTER_USD, _prim_path, position=_home_pos)
    build_ros2_diffdrive_graph(_robot_id, chassis_prim_path=f"{_prim_path}/chassis_link")
    print(f"[스폰] {_robot_id} (담당 종류={ROBOT_SHOE_TYPE[_robot_id]}) @ {_home_node} {_home_pos}")

    _robot_chassis_prims[_robot_id] = SingleXFormPrim(f"{_prim_path}/chassis_link")

    _beacon_pos = list(_home_pos)
    _beacon_pos[2] += _BEACON_HEIGHT_OFFSET_M
    _beacon = VisualCuboid(
        prim_path=f"/World/{_robot_id}/ColorBeacon",
        name=f"beacon_{_robot_id}",
        position=np.array(_beacon_pos),
        scale=np.array([0.2, 0.2, 0.2]),
        color=_ROBOT_TYPE_COLORS[ROBOT_SHOE_TYPE[_robot_id]],
    )
    world.scene.add(_beacon)
    _robot_beacons[_robot_id] = _beacon


# ╔══════════════════════════════════════════════════════════════╗
# ║  D. 그래프 포인트 시각화 — NODE_GRAPH의 각 지점을 바닥에 색칠된      ║
# ║     정사각형으로 표시 (물리/충돌 없음, 순수 디버그용)                ║
# ╚══════════════════════════════════════════════════════════════╝
# fleet_config.py를 계속 손보는 중이라, 실제로 로봇이 어디로 가는지 눈으로
# 바로 확인할 수 있게 노드 종류별로 색을 다르게 칠한다. 세분화(_subdivide_edge)
# 로 자동 생성된 중간 칸("__"이 이름에 들어간 노드)은 개수가 많아서 작게 그린다.
_NODE_MARKER_COLORS = {
    "pickup": np.array([0.12, 0.54, 0.54]),        # PICKUP_X — 청록
    "pickup_wait": np.array([0.72, 0.55, 0.31]),   # PICKUP_WAIT_X — 황토
    "hub": np.array([0.76, 0.47, 0.18]),           # HUB_X — 주황
    "rack_detour": np.array([0.49, 0.36, 0.75]),   # 우회 통로 — 보라
    "rack_near": np.array([0.23, 0.43, 0.65]),     # 근접 통로 — 파랑
    "rack_out": np.array([0.42, 0.48, 0.36]),      # RackX_OUT — 올리브
    "wait": np.array([0.60, 0.59, 0.54]),          # WAIT_N(홈 슬롯) — 회색
    "segment": np.array([0.72, 0.70, 0.63]),       # 본선 세분화 칸 — 연한 회색
}


def _categorize_node(name):
    if name.startswith("PICKUP_WAIT_"):
        return "pickup_wait"
    if name.startswith("PICKUP_") and "__" not in name:
        return "pickup"
    if name.startswith("HUB_"):
        return "hub"
    if "__" in name:
        return "segment"
    if name.startswith("WAIT_"):
        return "wait"
    if "_우회" in name and not name.endswith("_OUT"):
        return "rack_detour"
    if name.startswith("Rack") and name.endswith("_OUT"):
        return "rack_out"
    return "rack_near"


# USD 프림 경로/이름은 아스키 식별자만 허용한다 — 노드 이름에 들어간 한글
# (소/중/대/우회)이 그대로 들어가면 프림 생성이 실패하므로 아스키로 치환한다.
_KOREAN_TO_ASCII = {"소": "S", "중": "M", "대": "L", "우회": "Detour"}


def _sanitize_prim_name(name):
    safe = name
    for korean, ascii_ in _KOREAN_TO_ASCII.items():
        safe = safe.replace(korean, ascii_)
    return safe


for _node_name, _node_data in NODE_GRAPH.items():
    _category = _categorize_node(_node_name)
    _color = _NODE_MARKER_COLORS[_category]
    _safe_name = _sanitize_prim_name(_node_name)
    _size = 0.15 if _category == "segment" else 0.35
    _pos = list(_node_data["position"])
    _pos[2] = 0.01  # 바닥 위로 살짝 띄워서 그라운드 플레인과 Z-fighting 방지
    marker = VisualCuboid(
        prim_path=f"/World/GraphMarkers/{_safe_name}",
        name=f"marker_{_safe_name}",
        position=np.array(_pos),
        scale=np.array([_size, _size, 0.02]),
        color=_color,
    )
    world.scene.add(marker)


world.scene.add_default_ground_plane()
world.reset()

frame = 0
was_playing = False

print("\n" + "=" * 60)
print(f"[환경] Nova Carter {len(ROBOT_HOME_NODE)}대(ROS2 브리지) 생성 완료")
print("  AMR 제어: 외부 fleet_driver.py가 /<robot_id>/cmd_vel 로 담당")
print("=" * 60)

while simulation_app.is_running():
    world.step(render=True)
    is_playing = world.is_playing()

    if is_playing and not was_playing:
        print("[재생] 시작")

    # 색깔 비콘이 로봇을 따라가게 매 스텝 섀시의 현재 월드 위치를 읽어
    # 비콘 위치를 갱신한다. 부모-자식 프림 계층에 기대지 않고 항상 절대
    # 위치를 다시 계산해서 넣기 때문에, 어떤 링크가 실제로 움직이는지와
    # 무관하게 항상 정확하다.
    if is_playing:
        for _robot_id, _chassis in _robot_chassis_prims.items():
            _chassis_pos, _ = _chassis.get_world_pose()
            _beacon_pos = np.array(_chassis_pos)
            _beacon_pos[2] += _BEACON_HEIGHT_OFFSET_M
            _robot_beacons[_robot_id].set_world_pose(position=_beacon_pos)

    was_playing = is_playing

simulation_app.close()
