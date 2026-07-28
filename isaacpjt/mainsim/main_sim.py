from isaacsim import SimulationApp

CONFIG = {
    "headless": False,
    "width": 1280,
    "height": 720,
}

simulation_app = SimulationApp(CONFIG)

import os
import traceback
import math
import numpy as np

from isaacsim.core.utils.extensions import enable_extension

for extension_name in (
    "omni.graph.core",
    "omni.graph.action_nodes",
    "omni.graph.action_nodes_core",
    "omni.graph.scriptnode",
    "isaacsim.ros2.bridge",
    "omni.physx.graph",
    "isaacsim.asset.gen.conveyor",
):
    enable_extension(extension_name)

# Extension 초기화
simulation_app.update()

import omni.graph.core as og
import usdrt.Sdf
import omni.usd
import omni.timeline

from pxr import Usd

from isaacsim.core.api import World
from isaacsim.core.api.objects import VisualCuboid, VisualCylinder
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.prims import SingleXFormPrim

from fleet_config_test1 import (
    NODE_GRAPH,
    ROBOT_HOME_NODE,
    ROBOT_SHOE_TYPE,
    SHOE_TYPES,
    robot_spawn_yaw,
)

from simulation_node import (
    SimulationNode,
    create_simulation_node,
    prepare_stage,
)
from amr_pick_place import AmrArmController, prepare_cone_triggers

# amr_1의 로봇팔 pick&place 자동화만 우선 구현한다(다른 AMR로 확장하려면 이
# 세트를 늘리면 된다) - amr_ready/amr_carrying_complete의 amr_id(int)와
# AmrArmController를 매핑하는 데 쓴다.
PICK_PLACE_ROBOT_IDS = ("amr_1", "amr_2")
# pick/place 목표(demo0725.usd의 pallet/pallet_01/02/03)는 amr_pick_place.py가
# 직접 읽으므로 여기서는 박스 크기만 필요하다.
PICK_BOX_SIZE = np.array([0.28, 0.20, 0.11])

WHEEL_RADIUS_M = 0.14
WHEEL_BASE_M = 0.4132  # 트랙폭(좌우 구동 바퀴 간격)
WHEEL_JOINT_NAMES = ["joint_wheel_left", "joint_wheel_right"]

NOVA_CARTER_USD = os.path.expanduser(
    "~/cobot3_ws/isaacpjt/nova-carter/nova_carter_ur5e_surface_gripper.usd"
)


USD_PATH = os.path.expanduser(
    "~/cobot3_ws/isaacpjt/basic/heu/demo0728_v3.usd"
)

# NODE_GRAPH 지점 시각화용 색상 — 노드 종류가 아니라 담당 신발 종류(A/B/C/D)로
# 칠한다("이 노드를 이 로봇이 담당한다"는 게 색으로 바로 보이게).
_ROBOT_TYPE_COLORS = {
    "A": np.array([0.80, 0.20, 0.20]),  # 빨강
    "B": np.array([0.85, 0.65, 0.13]),  # 노랑
    "C": np.array([0.20, 0.70, 0.30]),  # 초록
    "D": np.array([0.20, 0.40, 0.80]),  # 파랑
}


def _node_shoe_type(name):
    for _t in SHOE_TYPES:
        if name.startswith(f"PICKUP_WAIT_{_t}"):
            return _t
    for _t in SHOE_TYPES:
        if name.startswith(f"PICKUP_{_t}"):
            return _t
    for _t in SHOE_TYPES:
        if name.startswith(f"HUB_{_t}"):
            return _t
    for _t in SHOE_TYPES:
        if name.startswith(f"Rack{_t}"):
            return _t
    return None


def add_node_graph_markers(world):
    """NODE_GRAPH의 각 지점을 바닥에 색칠된 정사각형(원기둥은 PICKUP_WAIT만)으로
    표시한다. 물리/충돌 없음, 순수 디버그용 — 1_conveyor_sorter_env.py와 동일 로직."""
    for _node_name, _node_data in NODE_GRAPH.items():
        _node_type = _node_shoe_type(_node_name)
        _color = _ROBOT_TYPE_COLORS.get(_node_type, np.array([0.6, 0.6, 0.6]))
        _is_segment = "__" in _node_name
        _is_pickup_wait = any(_node_name.startswith(f"PICKUP_WAIT_{_t}") for _t in SHOE_TYPES)
        _size = 0.15 if _is_segment else 0.35
        _pos = list(_node_data["position"])
        _pos[2] = 1.01  # 바닥 위로 살짝 띄워서 그라운드 플레인과 Z-fighting 방지
        if _is_pickup_wait:
            marker = VisualCylinder(
                prim_path=f"/World/GraphMarkers/{_node_name}",
                name=f"marker_{_node_name}",
                position=np.array(_pos),
                radius=_size / 2.0,
                height=0.02,
                color=_color,
            )
        else:
            marker = VisualCuboid(
                prim_path=f"/World/GraphMarkers/{_node_name}",
                name=f"marker_{_node_name}",
                position=np.array(_pos),
                scale=np.array([_size, _size, 0.02]),
                color=_color,
            )
        world.scene.add(marker)


def load_usd_stage(usd_path: str) -> Usd.Stage:
    absolute_path = os.path.abspath(os.path.expanduser(usd_path))

    if not os.path.isfile(absolute_path):
        raise FileNotFoundError(
            f"USD 파일이 없습니다: {absolute_path}"
        )

    usd_context = omni.usd.get_context()
    timeline = omni.timeline.get_timeline_interface()
    timeline.stop()

    for _ in range(3):
        simulation_app.update()

    print(f"[Stage] 열기: {absolute_path}")
    usd_context.open_stage(absolute_path)

    for _ in range(50):
        simulation_app.update()

    stage = usd_context.get_stage()

    if stage is None:
        raise RuntimeError("USD Stage 열기 실패")

    for _ in range(20):
        simulation_app.update()

    stage.SetTimeCodesPerSecond(180.0)

    timeline = omni.timeline.get_timeline_interface()
    timeline.set_target_framerate(180.0)

    print(f"[Stage] 로딩 완료: {stage.GetRootLayer().identifier}")

    return stage

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



def main() -> None:
    world = None
    simulation_node: SimulationNode | None = None

    try:
        stage = load_usd_stage(USD_PATH)

        # DomeLight 텍스처 확인
        from pxr import UsdLux

        dome_light_found = False

        for prim in stage.TraverseAll():
            if prim.IsA(UsdLux.DomeLight):
                dome_light_found = True
                dome_light = UsdLux.DomeLight(prim)

                print(f"[DomeLight] 경로: {prim.GetPath()}")
                print(
                    f"[DomeLight] 텍스처: "
                    f"{dome_light.GetTextureFileAttr().Get()}"
                )

        if not dome_light_found:
            print("[DomeLight] 현재 Stage에 DomeLight가 없습니다.")

        # World.reset() 전에 오류가 나는 구형 그래프와 충돌 노드를 끕니다.
        prepare_stage()

        for _ in range(5):
            simulation_app.update()

        for _robot_id, _home_node in ROBOT_HOME_NODE.items():
            _home_pos = list(NODE_GRAPH[_home_node]["position"])
            _home_yaw = robot_spawn_yaw(_robot_id)
            _prim_path = f"/World/{_robot_id}"
            spawn_asset(NOVA_CARTER_USD, _prim_path, position=_home_pos, yaw=_home_yaw)
            build_ros2_diffdrive_graph(_robot_id, chassis_prim_path=f"{_prim_path}/chassis_link")

            if _robot_id in PICK_PLACE_ROBOT_IDS:
                # world.reset()(첫 physics step) 전에 반드시 호출해야 한다 -
                # 그 뒤에 붙이면 접촉 판정이 영원히 감지되지 않는다(헤드리스로
                # 확인된 회귀, amr_pick_place.prepare_cone_triggers 참고).
                prepare_cone_triggers(_prim_path)

        for _ in range(5):
            simulation_app.update()

        world = World(stage_units_in_meters=1.0)

        add_node_graph_markers(world)

        print("[World] reset")
        world.reset()

        amr_arm_controllers: dict[int, AmrArmController] = {}
        for _robot_id in PICK_PLACE_ROBOT_IDS:
            _amr_num = int(_robot_id.split("_")[1])
            amr_arm_controllers[_amr_num] = AmrArmController(
                robot_id=_robot_id,
                robot_prim_path=f"/World/{_robot_id}",
                box_size=PICK_BOX_SIZE,
                physics_sim_view=world.physics_sim_view,
            )
            print(f"[amr_pick_place] {_robot_id} 로봇팔 pick&place 컨트롤러 준비 완료", flush=True)

        # 이후 토픽/서비스/Stage 제어 기능은 simulation_node.py에 추가합니다.
        simulation_node = create_simulation_node(amr_arm_controllers=amr_arm_controllers)

        print("[World] play")

        while simulation_app.is_running():
            world.step(render=True)
            simulation_node.update()

    except Exception as error:
        print(f"[Main 오류] {error}")
        traceback.print_exc()

    finally:
        if simulation_node is not None:
            simulation_node.shutdown()

        if world is not None:
            world.stop()

        simulation_app.close()


if __name__ == "__main__":
    main()