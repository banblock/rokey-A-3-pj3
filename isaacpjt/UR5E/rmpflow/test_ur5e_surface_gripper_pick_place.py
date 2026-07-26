"""nova_carter_ur5e_surface_gripper.usd + isaacsim.robot.manipulators.controllers.
PickPlaceController(우리 surface gripper 버전)가 실제로 동작하는지 확인.

gripper 인자로 isaacsim.robot.manipulators.grippers.surface_gripper.SurfaceGripper
(Isaac Sim 내장 래퍼)를 그대로 사용한다 - ParallelGripper 전용이 아니라
Gripper 베이스 클래스(forward(action=...)만 구현하면 됨)를 받으므로 호환된다.
"""

import sys
from pathlib import Path

import numpy as np

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import omni.usd
from pxr import UsdGeom

from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.prims import SingleArticulation
from isaacsim.robot.manipulators.grippers.surface_gripper import SurfaceGripper

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from ur5e_surface_gripper_pick_place_controller import PickPlaceController

INPUT_USD = "/home/rokey/cobot3_ws/isaacpjt/nova-carter/nova_carter_ur5e_surface_gripper.usd"
ROBOT_ROOT = "/World/nova_carter_ur5e"
ARTICULATION_PATH = f"{ROBOT_ROOT}/chassis_link"
UR5E_BASE_LINK_PATH = f"{ROBOT_ROOT}/arm_mount/ur5e/base_link"
GRIPPER_BASE_PATH = f"{ROBOT_ROOT}/arm_mount/ur5e/GripperBase"
SURFACE_GRIPPER_PATH = f"{ROBOT_ROOT}/arm_mount/ur5e/SurfaceGripper"

CUBE_INIT_POS = np.array([0.5, 0.2, 0.03])
GOAL_POS = np.array([0.3, -0.3, 0.03])

EVENTS_DT = [0.008, 0.005, 0.02, 0.1, 0.0025, 0.01, 0.0025, 0.05, 0.008, 0.08]


def main():
    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()

    world_prim = stage.GetPrimAtPath("/World")
    if not world_prim.IsValid():
        world_prim = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    world_prim.GetReferences().AddReference(INPUT_USD)
    for _ in range(20):
        simulation_app.update()

    cube = world.scene.add(
        DynamicCuboid(
            prim_path="/World/target_cube",
            name="target_cube",
            position=CUBE_INIT_POS,
            scale=np.array([0.04, 0.04, 0.04]),
            color=np.array([0.0, 0.0, 1.0]),
            mass=0.02,
        )
    )

    robot = world.scene.add(SingleArticulation(prim_path=ARTICULATION_PATH, name="nova_carter_ur5e"))
    world.reset()
    robot.initialize()

    ur5e_base_prim = stage.GetPrimAtPath(UR5E_BASE_LINK_PATH)
    xcache = UsdGeom.XformCache()
    base_world = xcache.GetLocalToWorldTransform(ur5e_base_prim)
    base_position = np.array(base_world.ExtractTranslation())
    base_quat_gf = base_world.ExtractRotationQuat()
    base_orientation = np.array([base_quat_gf.GetReal(), *base_quat_gf.GetImaginary()])

    gripper = SurfaceGripper(
        end_effector_prim_path=GRIPPER_BASE_PATH,
        surface_gripper_path=SURFACE_GRIPPER_PATH,
    )
    gripper.initialize(physics_sim_view=world.physics_sim_view, articulation_num_dofs=robot.num_dof)
    gripper.set_default_state(opened=True)
    print(f"[INFO] gripper initial: open={gripper.is_open()} closed={gripper.is_closed()}", flush=True)

    controller = PickPlaceController(
        name="ur5e_surface_gripper_pick_place_controller",
        gripper=gripper,
        robot_articulation=robot,
        end_effector_initial_height=0.35,
        events_dt=EVENTS_DT,
        robot_base_position=base_position,
        robot_base_orientation=base_orientation,
    )
    controller.reset()

    print(f"[INFO] cube={CUBE_INIT_POS} goal={GOAL_POS}", flush=True)

    step = 0
    while not controller.is_done() and step < 2000:
        cube_pos, _ = cube.get_world_pose()
        current_joints = robot.get_joint_positions()
        actions = controller.forward(
            picking_position=cube_pos,
            placing_position=GOAL_POS,
            current_joint_positions=current_joints,
            end_effector_offset=np.array([0.0, 0.0, 0.02]),
        )
        robot.apply_action(actions)
        world.step(render=False)
        if step % 100 == 0:
            print(
                f"  step={step} event={controller.get_current_event()} "
                f"cube_z={cube_pos[2]:.4f} gripper_closed={gripper.is_closed()}",
                flush=True,
            )
        step += 1

    print(f"[OK] pick&place 종료: is_done={controller.is_done()} total_steps={step}", flush=True)
    simulation_app.close()


if __name__ == "__main__":
    main()
