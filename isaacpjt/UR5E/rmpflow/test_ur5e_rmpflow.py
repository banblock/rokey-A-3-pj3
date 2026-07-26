"""UR5e RMPflow 동작 확인용 standalone 스크립트.

순수 ur5e.usd(Isaac Sim 기본 에셋)에 RMPFlowController를 붙여서 목표 pose로
end-effector를 이동시켜본다. RMPflow 설정 파일 자체는 Isaac Sim 에셋에 내장된
것을 그대로 사용한다 (rmpflow/ur5e_rmpflow_controller.py 참고).
"""

import sys
from pathlib import Path

import numpy as np

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import omni.usd
from pxr import UsdGeom

from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from ur5e_rmpflow_controller import RMPFlowController

UR5E_USD = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
    "Assets/Isaac/5.1/Isaac/Robots/UniversalRobots/ur5e/ur5e.usd"
)
ROBOT_PRIM_PATH = "/World/ur5e"

TARGET_POSITION = np.array([0.4, 0.2, 0.4])
TARGET_ORIENTATION = np.array([0.0, 1.0, 0.0, 0.0])  # w,x,y,z - flange가 아래를 보는 자세


def main():
    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()

    world_prim = stage.GetPrimAtPath("/World")
    if not world_prim.IsValid():
        world_prim = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    robot_prim = UsdGeom.Xform.Define(stage, ROBOT_PRIM_PATH).GetPrim()
    robot_prim.GetReferences().AddReference(UR5E_USD)
    for _ in range(20):
        simulation_app.update()

    robot = world.scene.add(SingleArticulation(prim_path=ROBOT_PRIM_PATH, name="ur5e"))
    world.reset()
    robot.initialize()

    controller = RMPFlowController(name="ur5e_rmpflow_controller", robot_articulation=robot)
    controller.reset()

    print(f"[TARGET] position={TARGET_POSITION} orientation(wxyz)={TARGET_ORIENTATION}", flush=True)

    for i in range(300):
        actions = controller.forward(
            target_end_effector_position=TARGET_POSITION,
            target_end_effector_orientation=TARGET_ORIENTATION,
        )
        robot.apply_action(actions)
        world.step(render=False)
        if i % 50 == 0:
            print(f"  step={i} joint_positions={robot.get_joint_positions()}", flush=True)

    print("[OK] RMPflow 300 step 실행 완료 (예외 없이 종료)", flush=True)
    simulation_app.close()


if __name__ == "__main__":
    main()
