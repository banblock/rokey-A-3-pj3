
# ── SimulationApp ─────────────────────────────────────────────
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

from isaacsim.core.utils.extensions import enable_extension
enable_extension("isaacsim.ros2.bridge")
simulation_app.update()

# ── import ────────────────────────────────────────────────────
from pathlib import Path
import sys
import time

import numpy as np
import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics

from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, VisualCuboid
from isaacsim.core.api.tasks import BaseTask
from isaacsim.core.api.materials.physics_material import PhysicsMaterial
from isaacsim.core.prims import SingleGeometryPrim
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.robot.manipulators.manipulators import SingleManipulator

# ============================================================
# 파라미터 — 다른 로봇 적용 시 여기만 수정
# ============================================================
_THIS_DIR = Path(__file__).resolve().parent

if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

# USD
USD_PATH        = str(_THIS_DIR / "Collected_m0609_camera/m0609_camera.usd")
ROBOT_PRIM_PATH = "/World/m0609"

# 링크·관절
EE_LINK_NAME    = "link_6"
GRIPPER_JOINTS  = ["finger_joint", "right_inner_knuckle_joint"]

# Drive 강화
DRIVE_STIFFNESS = 1e8
DRIVE_DAMPING   = 1e4
DRIVE_MAX_FORCE = 1e8

# 그리퍼
GRIPPER_OPEN    = [0.0, 0.0]
GRIPPER_CLOSE   = [0.5, 0.5]
GRIPPER_DELTA   = [-0.5, -0.5]

# 큐브 + 목표
CUBE_INIT_POS   = np.array([0.30, 0.4, 0.0515 / 2.0])
GOAL_POS        = np.array([0.55, -0.35, 0.0])

# 마찰
FINGER_STATIC   = 1.8
FINGER_DYNAMIC  = 1.4
CUBE_STATIC     = 1.2
CUBE_DYNAMIC    = 1.0


# ============================================================
# 유틸
# ============================================================
def find_prim_path_by_name(root_path: str, name: str):
    stage = omni.usd.get_context().get_stage()
    root_prim = stage.GetPrimAtPath(root_path)
    if not root_prim.IsValid():
        return None
    for prim in Usd.PrimRange(root_prim):
        if prim.GetName() == name:
            return str(prim.GetPath())
    return None


# ============================================================
# Task — set_up_scene 5단계 패턴 완성
# ============================================================
class M0609Task(BaseTask):

    def __init__(self, name):
        super().__init__(name=name, offset=None)
        self._task_achieved = False

    def set_up_scene(self, scene):
        super().set_up_scene(scene)
        self._load_usd()             # 1. LOAD
        self._discover_links()       # 2. DISCOVER
        self._setup_physics()        # 3. PHYSICS
        self._register_robot(scene)  # 4. REGISTER
        self._create_scene(scene)    # 5. SCENE
        print("\n  [완료] 씬 구성 성공!\n")

    # ── 1. LOAD ──────────────────────────────────────────────
    def _load_usd(self):
        print("\n" + "=" * 60)
        print("[1.LOAD] USD 로드")
        print("=" * 60)

        stage = omni.usd.get_context().get_stage()
        world_prim = stage.GetPrimAtPath("/World")
        if not world_prim.IsValid():
            world_prim = UsdGeom.Xform.Define(stage, "/World").GetPrim()
        world_prim.GetReferences().AddReference(USD_PATH)

        for _ in range(30):
            simulation_app.update()

        print(f"  [OK] {USD_PATH}")

    # ── 2. DISCOVER ──────────────────────────────────────────
    def _discover_links(self):
        print("\n" + "=" * 60)
        print("[2.DISCOVER] 링크 경로 탐색")
        print("=" * 60)

        self._ee_path = find_prim_path_by_name(ROBOT_PRIM_PATH, EE_LINK_NAME)
        if self._ee_path is None:
            raise RuntimeError(f"'{EE_LINK_NAME}' not found")
        print(f"  EE ({EE_LINK_NAME}) = {self._ee_path}")

        for jn in GRIPPER_JOINTS:
            path = find_prim_path_by_name(ROBOT_PRIM_PATH, jn)
            print(f"  {jn:<35} = {path}")


    # ── 3. PHYSICS ───────────────────────────────────────────
    def _setup_physics(self):
        print("\n" + "=" * 60)
        print("[3.PHYSICS] 물리 설정")
        print("=" * 60)

        stage = omni.usd.get_context().get_stage()

        # b. Drive 강화
        drive_count = 0
        for prim in Usd.PrimRange(stage.GetPrimAtPath(ROBOT_PRIM_PATH)):
            for dt in ["angular", "linear"]:
                drive = UsdPhysics.DriveAPI.Get(prim, dt)
                if drive:
                    drive.GetStiffnessAttr().Set(DRIVE_STIFFNESS)
                    drive.GetDampingAttr().Set(DRIVE_DAMPING)
                    drive.GetMaxForceAttr().Set(DRIVE_MAX_FORCE)
                    drive_count += 1
        print(f"  [OK] drive updated: {drive_count}")

    # ── 4. REGISTER ──────────────────────────────────────────
    def _register_robot(self, scene):
        print("\n" + "=" * 60)
        print("[4.REGISTER] 로봇 등록")
        print("=" * 60)
        simulation_app.update()
        gripper = ParallelGripper(
            end_effector_prim_path=self._ee_path,
            joint_prim_names=GRIPPER_JOINTS,
            joint_opened_positions=np.array(GRIPPER_OPEN),
            joint_closed_positions=np.array(GRIPPER_CLOSE),
            action_deltas=np.array(GRIPPER_DELTA),
        )

        self._robot = scene.add(
            SingleManipulator(
                prim_path=ROBOT_PRIM_PATH,
                name="m0609_robot",
                end_effector_prim_path=self._ee_path,
                gripper=gripper,
            )
        )
        print(f"  [OK] SingleManipulator: {ROBOT_PRIM_PATH}")

    # ── 5. SCENE ─────────────────────────────────────────────
    def _create_scene(self, scene):
        print("\n" + "=" * 60)
        print("[5.SCENE] 작업 환경 구성")
        print("=" * 60)

        # 큐브
        cube_material = PhysicsMaterial(
            prim_path="/World/Physics_Materials/cube_material",
            static_friction=CUBE_STATIC,
            dynamic_friction=CUBE_DYNAMIC,
            restitution=0.0,
        )
        self._cube = scene.add(
            DynamicCuboid(
                prim_path="/World/target_cube",
                name="target_cube",
                position=CUBE_INIT_POS,
                scale=np.array([0.05, 0.05, 0.05]),
                color=np.array([0.0, 0.0, 1.0]),
                mass=0.05,
                physics_material=cube_material,
            )
        )
        print(f"  [OK] cube @ {CUBE_INIT_POS}")

        # 목표 마커
        scene.add(
            VisualCuboid(
                prim_path="/World/goal_marker",
                name="goal_marker",
                position=GOAL_POS,
                scale=np.array([0.06, 0.06, 0.001]),
                color=np.array([0.0, 1.0, 0.0]),
            )
        )
        print(f"  [OK] goal @ {GOAL_POS}")

        # 손가락 마찰
        finger_material = PhysicsMaterial(
            prim_path="/World/Physics_Materials/finger_material",
            static_friction=FINGER_STATIC,
            dynamic_friction=FINGER_DYNAMIC,
            restitution=0.0,
        )
        for link_name in ["left_inner_finger", "right_inner_finger"]:
            link_path = find_prim_path_by_name(ROBOT_PRIM_PATH, link_name)
            if link_path:
                SingleGeometryPrim(
                    prim_path=link_path,
                    name=f"{link_name}_geom",
                ).apply_physics_material(finger_material)
                print(f"  [OK] friction: {link_path}")

    # ── Task 콜백 ────────────────────────────────────────────
    def get_observations(self):
        cube_pos, _ = self._cube.get_world_pose()
        return {
            self._robot.name: {
                "joint_positions": self._robot.get_joint_positions(),
            },
            self._cube.name: {
                "position": cube_pos,
                "goal_position": GOAL_POS,
            },
        }

    def pre_step(self, control_index, simulation_time):
        cube_pos, _ = self._cube.get_world_pose()
        if not self._task_achieved and np.mean(np.abs(GOAL_POS - cube_pos)) < 0.02:
            self._cube.get_applied_visual_material().set_color(np.array([0.0, 1.0, 0.0]))
            self._task_achieved = True

    def post_reset(self):
        self._robot.gripper.set_joint_positions(
            self._robot.gripper.joint_opened_positions
        )
        self._cube.get_applied_visual_material().set_color(np.array([0.0, 0.0, 1.0]))
        self._task_achieved = False


# ============================================================
# 메인
# ============================================================
def main():
    my_world = World(stage_units_in_meters=1.0)

    task = M0609Task(name="m0609_task")
    my_world.add_task(task)
    my_world.reset()

    robot = my_world.scene.get_object("m0609_robot")
    robot.initialize()
    robot.gripper.initialize(
        physics_sim_view=my_world.physics_sim_view,
        articulation_apply_action_func=robot.apply_action,
        get_joint_positions_func=robot.get_joint_positions,
        set_joint_positions_func=robot.set_joint_positions,
        dof_names=robot.dof_names,
    )

    # ── DOF 확인 ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("[DOF] Joint 정보 확인")
    print("=" * 60)
    print(f"  DOF 수: {robot.num_dof}")
    for i, dof_name in enumerate(robot.dof_names):
        print(f"  [{i:2d}] {dof_name}")
    print("=" * 60)

    print("\n[렌더 루프 시작]\n")
    print("  확인 포인트:")
    print("  - 파란 큐브가 바닥에 놓여있는가")
    print("  - 초록 목표 마커가 보이는가")
    print("  - rqt 카메라에 큐브가 보이는가\n")

    was_playing = False

    while simulation_app.is_running():
        my_world.step(render=True)
        time.sleep(0.01)
        is_playing = my_world.is_playing()

        if is_playing and not was_playing:
            my_world.reset()
            robot.initialize()
            robot.gripper.initialize(
                physics_sim_view=my_world.physics_sim_view,
                articulation_apply_action_func=robot.apply_action,
                get_joint_positions_func=robot.get_joint_positions,
                set_joint_positions_func=robot.set_joint_positions,
                dof_names=robot.dof_names,
            )

        was_playing = is_playing

    simulation_app.close()


if __name__ == "__main__":
    main()