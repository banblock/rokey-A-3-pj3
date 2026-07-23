from isaacsim import SimulationApp

CONFIG = {
    "headless": False,
    "width": 1280,
    "height": 720,
}

simulation_app = SimulationApp(CONFIG)

import os
import traceback

import omni.usd
from pxr import Usd

from isaacsim.core.api import World
from isaacsim.core.utils.extensions import enable_extension


for extension_name in (
    "omni.graph.core",
    "omni.graph.action_nodes",
    "omni.graph.action_nodes_core",
    "isaacsim.ros2.bridge",
    "omni.physx.graph",
    "isaacsim.asset.gen.conveyor",
):
    enable_extension(extension_name)

simulation_app.update()


USD_PATH = os.path.expanduser(
    "~/cobot3_ws/isaacpjt/basic/heu/stage_v10.usd"
)

CONVEYOR_GRAPH_PATH = (
    "/World/ReturnCell/Conveyors/Input/ConveyorTrack/ConveyorBeltGraph"
)

CONVEYOR_VELOCITY_ATTRIBUTE = "graph:variable:Velocity"


def load_usd_stage(usd_path: str) -> Usd.Stage:
    absolute_path = os.path.abspath(
        os.path.expanduser(usd_path)
    )

    if not os.path.isfile(absolute_path):
        raise FileNotFoundError(
            f"USD 파일이 없습니다: {absolute_path}"
        )

    usd_context = omni.usd.get_context()

    print(f"[Stage] 열기: {absolute_path}")
    usd_context.open_stage(absolute_path)

    for _ in range(50):
        simulation_app.update()

    stage = usd_context.get_stage()

    if stage is None:
        raise RuntimeError("USD Stage 열기 실패")

    print(
        f"[Stage] 로딩 완료: "
        f"{stage.GetRootLayer().identifier}"
    )

    return stage


def get_conveyor_velocity_attribute(
    stage: Usd.Stage,
) -> Usd.Attribute:
    graph_prim = stage.GetPrimAtPath(
        CONVEYOR_GRAPH_PATH
    )

    if not graph_prim.IsValid():
        raise RuntimeError(
            f"Action Graph Prim을 찾을 수 없습니다: "
            f"{CONVEYOR_GRAPH_PATH}"
        )

    velocity_attribute = graph_prim.GetAttribute(
        CONVEYOR_VELOCITY_ATTRIBUTE
    )

    if not velocity_attribute.IsValid():
        raise RuntimeError(
            f"Velocity Attribute를 찾을 수 없습니다: "
            f"{CONVEYOR_GRAPH_PATH}."
            f"{CONVEYOR_VELOCITY_ATTRIBUTE}"
        )

    print(
        f"[Conveyor] Attribute 확인: "
        f"{velocity_attribute.GetName()} = "
        f"{velocity_attribute.Get()}"
    )

    return velocity_attribute


def main() -> None:
    world = None

    try:
        stage = load_usd_stage(USD_PATH)

        for _ in range(5):
            simulation_app.update()

        world = World(stage_units_in_meters=1.0)

        print("[World] reset")
        world.reset()

        for _ in range(5):
            simulation_app.update()

        velocity_attribute = (
            get_conveyor_velocity_attribute(stage)
        )

        elapsed_time = 0.0
        conveyor_stopped = False
        conveyor_restarted = False

        print("[World] play")

        while simulation_app.is_running():
            world.step(render=True)

            elapsed_time += world.get_physics_dt()

            if (
                elapsed_time >= 1.55
                and not conveyor_stopped
            ):
                velocity_attribute.Set(0.0)
                conveyor_stopped = True
                simulation_app.update()
                print(
                    f"[Conveyor] 정지: "
                    f"{elapsed_time:.2f}초, "
                    f"Velocity={velocity_attribute.Get()}"
                )

            if (
                elapsed_time >= 6.55
                and not conveyor_restarted
            ):
                velocity_attribute.Set(1.0)
                conveyor_restarted = True
                simulation_app.update()
                print(
                    f"[Conveyor] 재시작: "
                    f"{elapsed_time:.2f}초, "
                    f"Velocity={velocity_attribute.Get()}"
                )

    except Exception as error:
        print(f"[Main 오류] {error}")
        traceback.print_exc()

    finally:
        if world is not None:
            world.stop()

        simulation_app.close()


if __name__ == "__main__":
    main()