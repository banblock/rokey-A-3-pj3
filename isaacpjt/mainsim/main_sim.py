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


def load_usd_stage(usd_path: str) -> Usd.Stage:
    absolute_path = os.path.abspath(os.path.expanduser(usd_path))

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

    print(f"[Stage] 로딩 완료: {stage.GetRootLayer().identifier}")

    return stage


def main() -> None:
    world = None

    try:
        load_usd_stage(USD_PATH)

        for _ in range(5):
            simulation_app.update()

        world = World(stage_units_in_meters=1.0)

        print("[World] reset")
        world.reset()

        for _ in range(5):
            simulation_app.update()

        print("[World] play")

        while simulation_app.is_running():
            world.step(render=True)

    except Exception as error:
        print(f"[Main 오류] {error}")
        traceback.print_exc()

    finally:
        if world is not None:
            world.stop()

        simulation_app.close()


if __name__ == "__main__":
    main()
