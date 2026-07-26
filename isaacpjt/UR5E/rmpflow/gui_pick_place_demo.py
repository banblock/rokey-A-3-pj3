"""nova_carter_ur5e_surface_gripper.usd + PickPlaceController(surface gripper)를
Isaac Sim GUI(headless=False)로 띄워서 눈으로 확인하는 데모.

한 사이클(집기->옮기기->놓기)이 끝나면 큐브를 원위치로 되돌리고 컨트롤러를
reset해서 계속 반복 재생한다.
"""

import sys
from pathlib import Path

import numpy as np

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import omni.usd
from pxr import PhysxSchema, UsdGeom

from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.prims import SingleArticulation

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from ur5e_surface_gripper_pick_place_controller import PickPlaceController
from trigger_surface_gripper import TriggerSurfaceGripper

INPUT_USD = "/home/rokey/cobot3_ws/isaacpjt/nova-carter/nova_carter_ur5e_surface_gripper.usd"
ROBOT_ROOT = "/World/nova_carter_ur5e"
ARTICULATION_PATH = f"{ROBOT_ROOT}/chassis_link"
UR5E_BASE_LINK_PATH = f"{ROBOT_ROOT}/arm_mount/ur5e/base_link"
GRIPPER_BASE_PATH = f"{ROBOT_ROOT}/arm_mount/ur5e/GripperBase"
CONTACT_TRIGGER_PATH = f"{GRIPPER_BASE_PATH}/ContactTrigger"
CONE_PATHS = [f"{GRIPPER_BASE_PATH}/Cone_{i}" for i in range(4)]
GRASP_JOINT_PATH = f"{GRIPPER_BASE_PATH}/GraspJoint"
CONE_CONTACT_LOCAL_OFFSET = (0.0, 0.0, -0.0015)

BOX_SIZE = np.array([0.33, 0.23, 0.13])  # 실제 신발 박스 크기(가로x세로x높이, cm->m)
CUBE_INIT_POS = np.array([0.5, 0.42, BOX_SIZE[2] / 2.0])
GOAL_POS = np.array([0.5, -0.42, BOX_SIZE[2] / 2.0])

EVENTS_DT = [0.008, 0.003, 0.05, 0.3, 0.0025, 0.01, 0.0025, 0.15, 0.008, 0.08]


def main():
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    stage = omni.usd.get_context().get_stage()

    robot_prim = UsdGeom.Xform.Define(stage, ROBOT_ROOT).GetPrim()
    robot_prim.GetReferences().AddReference(INPUT_USD)
    for _ in range(20):
        simulation_app.update()

    # ContactTrigger는 더 이상 쓰지 않는다 - Cone 자체(PhysxTriggerAPI는 이미
    # 적용돼 있음)에 PhysxTriggerStateAPI만 추가로 붙이면 4개 콘의 겹침 합집합
    # (OR)으로 동일하게 접촉 판정이 가능하다는 것을 헤드리스 테스트로 확인함.
    ct_prim = stage.GetPrimAtPath(CONTACT_TRIGGER_PATH)
    if ct_prim.IsValid():
        ct_prim.SetActive(False)
    for cone_path in CONE_PATHS:
        cone_prim = stage.GetPrimAtPath(cone_path)
        if cone_prim.IsValid() and not cone_prim.HasAPI(PhysxSchema.PhysxTriggerStateAPI):
            PhysxSchema.PhysxTriggerStateAPI.Apply(cone_prim)

    cube = world.scene.add(
        DynamicCuboid(
            prim_path="/World/target_cube",
            name="target_cube",
            position=CUBE_INIT_POS,
            scale=BOX_SIZE,
            color=np.array([0.0, 0.0, 1.0]),
            mass=1.1,  # 성인 스니커즈 한 켤레 + 신발 박스 실제 무게(0.8~1.4kg) 반영
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

    gripper = TriggerSurfaceGripper(
        end_effector_prim_path=GRIPPER_BASE_PATH,
        trigger_paths=CONE_PATHS,
        joint_path=GRASP_JOINT_PATH,
        tip_local_offset=CONE_CONTACT_LOCAL_OFFSET,
    )
    gripper.initialize(physics_sim_view=world.physics_sim_view, articulation_num_dofs=robot.num_dof)
    gripper.set_default_state(opened=True)
    gripper.post_reset()

    controller = PickPlaceController(
        name="ur5e_surface_gripper_pick_place_controller",
        gripper=gripper,
        robot_articulation=robot,
        end_effector_initial_height=0.55,
        events_dt=EVENTS_DT,
        robot_base_position=base_position,
        robot_base_orientation=base_orientation,
        chassis_obstacle_prim_path=ARTICULATION_PATH,
        contact_trigger_path=CONE_PATHS,
    )
    controller.reset()

    print("[INFO] GUI pick & place 데모 시작 - Isaac Sim 창에서 재생됩니다.", flush=True)
    print(f"[INFO] cube={CUBE_INIT_POS} goal={GOAL_POS}", flush=True)

    world.play()
    cycle = 0
    step = 0
    last_event = -1
    contact_seen = False
    cone_trigger_states = [
        PhysxSchema.PhysxTriggerStateAPI(stage.GetPrimAtPath(p)) for p in CONE_PATHS
    ]
    while simulation_app.is_running():
        cube_pos, _ = cube.get_world_pose()
        current_joints = robot.get_joint_positions()

        # end_effector_offset intentionally aims a few cm past the real
        # surface, since RMPflow never converges tight enough on its own to
        # guarantee actual contact. But once contact really happens, keep
        # asking for that past-surface point (even with the controller's
        # own descend-phase freeze) still applies constant downward force
        # against something solid, which shows up as jitter/shaking. So drop
        # the override back to 0 the moment contact is first seen - the
        # controller's freeze then holds the arm at a position it has
        # actually already reached, not a value driving further pressure.
        if not contact_seen and any(
            s.GetTriggeredCollisionsRel().GetTargets() for s in cone_trigger_states
        ):
            contact_seen = True
        pick_offset_z = 0.0 if contact_seen else -0.015

        actions = controller.forward(
            picking_position=cube_pos,
            placing_position=GOAL_POS,
            current_joint_positions=current_joints,
            end_effector_offset=np.array([0.0, 0.0, pick_offset_z]),
            end_effector_orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        )
        robot.apply_action(actions)
        world.step(render=True)

        event = controller.get_current_event()
        if event != last_event or step % 100 == 0:
            xcache.Clear()
            gb_prim = stage.GetPrimAtPath(GRIPPER_BASE_PATH)
            gb_world = xcache.GetLocalToWorldTransform(gb_prim)
            gb_pos = np.array(gb_world.ExtractTranslation())
            apex_world = np.array(gb_world.Transform((0.0, 0.0, -0.0015)))
            dist_apex = np.linalg.norm(apex_world - cube_pos)
            dist_back = np.linalg.norm(gb_pos - cube_pos)
            facing = "CONE(OK)" if dist_apex < dist_back else "BACK(WRONG)"
            print(
                f"[DIAG] step={step} event={event} cube={np.round(cube_pos,3)} "
                f"gb_pos={np.round(gb_pos,3)} dist_apex={dist_apex:.3f} facing={facing}",
                flush=True,
            )
            last_event = event
        step += 1

        if controller.is_done():
            cycle += 1
            print(f"[INFO] cycle {cycle} 완료 - 큐브 원위치 후 재시작", flush=True)
            cube.set_world_pose(position=CUBE_INIT_POS)
            cube.set_linear_velocity(np.zeros(3))
            cube.set_angular_velocity(np.zeros(3))
            gripper.set_default_state(opened=True)
            gripper.post_reset()
            controller.reset()
            contact_seen = False

    simulation_app.close()


if __name__ == "__main__":
    main()
