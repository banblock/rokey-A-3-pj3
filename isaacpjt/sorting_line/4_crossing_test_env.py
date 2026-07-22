
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
from isaacsim.core.api.objects import VisualCuboid, VisualCylinder
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.storage.native import get_assets_root_path

from crossing_test_config import NODE_GRAPH, ROBOT_HOME_NODE, ROBOT_LABEL, CROSSING_POINT, robot_spawn_yaw

# ╔══════════════════════════════════════════════════════════════╗
# ║  edge_conflict(구간 충돌) 회피 로직 단독 테스트용 씬                  ║
# ║  — crossing_test_fms.py + fleet_driver.py(config_module=       ║
# ║    crossing_test_config)와 같이 띄운다.                          ║
# ║                                                                ║
# ║  로봇 2대가 서로 노드를 공유하지 않는 X자 대각선을 영원히 왕복한다.       ║
# ║  두 대각선이 겹치는 교차점(CROSSING_POINT)은 그래프 노드가 아니라       ║
# ║  순수 시각화용 마커만 그려둔다 — 실제로 저기서 두 로봇이 부딪히지        ║
# ║  않고 서로 양보하는지 눈으로 확인하는 게 이 씬의 목적이다.              ║
# ╚══════════════════════════════════════════════════════════════╝

world = World(stage_units_in_meters=1.0)
stage = omni.usd.get_context().get_stage()

dome_light = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
dome_light.CreateIntensityAttr(1000.0)

WHEEL_RADIUS_M = 0.14
WHEEL_BASE_M = 0.4132  # 트랙폭(좌우 구동 바퀴 간격)
WHEEL_JOINT_NAMES = ["joint_wheel_left", "joint_wheel_right"]

_assets_root_path = get_assets_root_path()
if _assets_root_path is None:
    raise RuntimeError("Isaac Sim 기본 에셋 서버(Nucleus)에 연결할 수 없습니다.")
NOVA_CARTER_USD = _assets_root_path + "/Isaac/Robots/NVIDIA/NovaCarter/nova_carter.usd"


def spawn_asset(usd_path, prim_path, position, yaw=0.0):
    add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
    xform = SingleXFormPrim(prim_path)
    half = yaw / 2.0
    orientation_wxyz = np.array([math.cos(half), 0.0, 0.0, math.sin(half)])
    xform.set_world_pose(position=np.array(position), orientation=orientation_wxyz)
    return xform


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


# 로봇마다 다른 색 비콘을 위에 띄워 어느 로봇이 어느 대각선(R1: 좌하↔우상,
# R2: 좌상↔우하)을 타는지 한눈에 구분한다.
_ROBOT_COLORS = {
    "amr_1": np.array([0.80, 0.20, 0.20]),  # 빨강 — R1
    "amr_2": np.array([0.20, 0.40, 0.80]),  # 파랑 — R2
}
_BEACON_HEIGHT_OFFSET_M = 0.5
_robot_chassis_prims = {}
_robot_beacons = {}

for _robot_id, _home_node in ROBOT_HOME_NODE.items():
    _home_pos = list(NODE_GRAPH[_home_node]["position"])
    _home_yaw = robot_spawn_yaw(_robot_id)
    _prim_path = f"/World/{_robot_id}"
    spawn_asset(NOVA_CARTER_USD, _prim_path, position=_home_pos, yaw=_home_yaw)
    build_ros2_diffdrive_graph(_robot_id, chassis_prim_path=f"{_prim_path}/chassis_link")
    print(
        f"[스폰] {_robot_id}({ROBOT_LABEL[_robot_id]}) @ {_home_node} {_home_pos} "
        f"yaw={math.degrees(_home_yaw):.0f}도"
    )

    _robot_chassis_prims[_robot_id] = SingleXFormPrim(f"{_prim_path}/chassis_link")

    _beacon_pos = list(_home_pos)
    _beacon_pos[2] += _BEACON_HEIGHT_OFFSET_M
    _beacon = VisualCuboid(
        prim_path=f"/World/{_robot_id}/ColorBeacon",
        name=f"beacon_{_robot_id}",
        position=np.array(_beacon_pos),
        scale=np.array([0.2, 0.2, 0.2]),
        color=_ROBOT_COLORS[_robot_id],
    )
    world.scene.add(_beacon)
    _robot_beacons[_robot_id] = _beacon


# 그래프 4개 지점(HOME/FAR) + 교차점(그래프 노드 아님, 시각화 전용)을 바닥에 표시.
for _node_name, _node_data in NODE_GRAPH.items():
    # 노드 이름 접두사(R1_/R2_)로 로봇별 색을 그대로 맞춘다.
    _color = _ROBOT_COLORS["amr_1"] if _node_name.startswith("R1_") else _ROBOT_COLORS["amr_2"]
    _pos = list(_node_data["position"])
    _pos[2] = 0.01  # 바닥 위로 살짝 띄워서 그라운드 플레인과 Z-fighting 방지
    marker = VisualCuboid(
        prim_path=f"/World/GraphMarkers/{_node_name}",
        name=f"marker_{_node_name}",
        position=np.array(_pos),
        scale=np.array([0.35, 0.35, 0.02]),
        color=_color,
    )
    world.scene.add(marker)

_crossing_pos = list(CROSSING_POINT)
_crossing_pos[2] = 0.01
crossing_marker = VisualCylinder(
    prim_path="/World/GraphMarkers/CROSSING_POINT",
    name="marker_crossing_point",
    position=np.array(_crossing_pos),
    radius=0.25,
    height=0.02,
    color=np.array([0.9, 0.9, 0.1]),  # 노랑 — 두 로봇의 경로가 겹치는 지점, 그래프 노드 아님
)
world.scene.add(crossing_marker)


world.scene.add_default_ground_plane()
world.reset()

was_playing = False

print("\n" + "=" * 60)
print(f"[환경] edge_conflict 테스트용 Nova Carter {len(ROBOT_HOME_NODE)}대(ROS2 브리지) 생성 완료")
print("  AMR 제어: crossing_test_fms.py + fleet_driver.py(config_module=crossing_test_config)")
print("  노란 원(교차점)에서 두 로봇이 서로 충돌 없이 양보하는지 확인할 것")
print("=" * 60)

while simulation_app.is_running():
    world.step(render=True)
    is_playing = world.is_playing()

    if is_playing and not was_playing:
        print("[재생] 시작")

    if is_playing:
        for _robot_id, _chassis in _robot_chassis_prims.items():
            _chassis_pos, _ = _chassis.get_world_pose()
            _beacon_pos = np.array(_chassis_pos)
            _beacon_pos[2] += _BEACON_HEIGHT_OFFSET_M
            _robot_beacons[_robot_id].set_world_pose(position=_beacon_pos)

    was_playing = is_playing

simulation_app.close()
