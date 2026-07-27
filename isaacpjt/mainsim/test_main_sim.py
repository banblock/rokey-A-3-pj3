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
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.storage.native import get_assets_root_path

from fleet_config_test1 import (
    NODE_GRAPH,
    ROBOT_HOME_NODE,
    ROBOT_SHOE_TYPE,
    SHOE_TYPES,
    robot_spawn_yaw,
)

from simulation_node_14 import (
    SimulationNode,
    create_simulation_node,
    prepare_stage,
)

WHEEL_RADIUS_M = 0.14
WHEEL_BASE_M = 0.4132  # 트랙폭(좌우 구동 바퀴 간격)
WHEEL_JOINT_NAMES = ["joint_wheel_left", "joint_wheel_right"]

_assets_root_path = get_assets_root_path()
if _assets_root_path is None:
    raise RuntimeError("Isaac Sim 기본 에셋 서버(Nucleus)에 연결할 수 없습니다.")
NOVA_CARTER_USD = _assets_root_path + "/Isaac/Robots/NVIDIA/NovaCarter/nova_carter.usd"


USD_PATH = os.path.expanduser(
    "~/cobot3_ws/isaacpjt/basic/heu/demo0726_v2.usd"
)


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

        for _ in range(5):
            simulation_app.update()

        world = World(stage_units_in_meters=1.0)

        print("[World] reset")
        world.reset()

        # 이후 토픽/서비스/Stage 제어 기능은 simulation_node.py에 추가합니다.
        simulation_node = create_simulation_node()

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