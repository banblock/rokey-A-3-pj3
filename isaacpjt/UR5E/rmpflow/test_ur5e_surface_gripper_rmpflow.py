"""nova_carter_ur5e_surface_gripper.usd에 붙여서 gripper_tip_link 기준 RMPflow가
실제로 동작하는지 확인하는 standalone 스크립트.

이 조립체는 RobotAssembler로 합쳐져 있어서 전체가 하나의 articulation이고,
articulation root는 chassis_link다 (UR5e 자체의 articulation root는 assembler가
비활성화함). 따라서 RMPflow의 URDF 좌표계 원점(UR5e의 base_link)이 chassis와
다른 위치에 있으므로, robot_base_position/orientation을 직접 넘겨준다.
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

from ur5e_surface_gripper_rmpflow_controller import RMPFlowController

INPUT_USD = "/home/rokey/cobot3_ws/isaacpjt/nova-carter/nova_carter_ur5e_surface_gripper.usd"
ROBOT_ROOT = "/World/nova_carter_ur5e"
ARTICULATION_PATH = f"{ROBOT_ROOT}/chassis_link"
UR5E_BASE_LINK_PATH = f"{ROBOT_ROOT}/arm_mount/ur5e/base_link"

TARGET_POSITION = np.array([0.5, 0.3, 0.4])
TARGET_ORIENTATION = np.array([0.0, 1.0, 0.0, 0.0])  # w,x,y,z


def main():
    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()

    world_prim = stage.GetPrimAtPath("/World")
    if not world_prim.IsValid():
        world_prim = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    world_prim.GetReferences().AddReference(INPUT_USD)
    for _ in range(20):
        simulation_app.update()

    robot = world.scene.add(SingleArticulation(prim_path=ARTICULATION_PATH, name="nova_carter_ur5e"))
    world.reset()
    robot.initialize()

    ur5e_base_prim = stage.GetPrimAtPath(UR5E_BASE_LINK_PATH)
    xcache = UsdGeom.XformCache()
    base_world = xcache.GetLocalToWorldTransform(ur5e_base_prim)
    base_position = np.array(base_world.ExtractTranslation())
    base_quat_gf = base_world.ExtractRotationQuat()
    base_orientation = np.array(
        [base_quat_gf.GetReal(), *base_quat_gf.GetImaginary()]
    )
    print(f"[INFO] UR5e base_link world pose: pos={base_position} quat(wxyz)={base_orientation}", flush=True)

    controller = RMPFlowController(
        name="ur5e_surface_gripper_rmpflow_controller",
        robot_articulation=robot,
        end_effector_frame_name="gripper_tip_link",
        robot_base_position=base_position,
        robot_base_orientation=base_orientation,
    )
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

    print("[OK] RMPflow(surface gripper) 300 step 실행 완료 (예외 없이 종료)", flush=True)
    simulation_app.close()


if __name__ == "__main__":
    main()
