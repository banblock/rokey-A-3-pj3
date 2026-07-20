"""
5장. 씬 로드 + 물리 설정
    set_up_scene 패턴 1,2,3 단계: LOAD → DISCOVER → PHYSICS
"""

# ── SimulationApp (반드시 모든 import보다 먼저) ───────────────
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

from isaacsim.core.utils.extensions import enable_extension
enable_extension("isaacsim.ros2.bridge")
simulation_app.update()

# ── import ────────────────────────────────────────────────────
from pathlib import Path
import time

import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics

from isaacsim.core.api import World
from isaacsim.core.api.tasks import BaseTask

# ============================================================
# 파라미터 — 다른 로봇 적용 시 여기만 수정
# ============================================================
_THIS_DIR = Path(__file__).resolve().parent

USD_PATH        = str(_THIS_DIR / "Collected_m0609_camera/m0609_camera.usd")
ROBOT_PRIM_PATH = "/World/m0609"
EE_LINK_NAME    = "link_6"
GRIPPER_JOINTS  = ["finger_joint", "right_inner_knuckle_joint"]

DRIVE_STIFFNESS = 1e8
DRIVE_DAMPING   = 1e4
DRIVE_MAX_FORCE = 1e8


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
# Task — set_up_scene 패턴 1,2,3
# ============================================================
class M0609Task(BaseTask):

    def __init__(self, name):
        super().__init__(name=name, offset=None)

    def set_up_scene(self, scene):
        super().set_up_scene(scene)
        self._load_usd()             # 1. LOAD
        self._discover_links()       # 2. DISCOVER
        self._setup_physics()        # 3. PHYSICS
        # self._register_robot(scene)  # 4. REGISTER  ← 6장
        # self._create_scene(scene)    # 5. SCENE     ← 6장
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

        for _ in range(15):
            simulation_app.update()

        print(f"  [OK] {USD_PATH}")
        print(f"       -> robot prim = {ROBOT_PRIM_PATH}")

    # ── 2. DISCOVER ──────────────────────────────────────────
    def _discover_links(self):
        print("\n" + "=" * 60)
        print("[2.DISCOVER] 링크 경로 탐색")
        print("=" * 60)

        self._ee_path = find_prim_path_by_name(ROBOT_PRIM_PATH, EE_LINK_NAME)
        if self._ee_path is None:
            raise RuntimeError(
                f"'{EE_LINK_NAME}' not found under {ROBOT_PRIM_PATH}"
            )
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

        # Joint Drive 강화
        drive_count = 0
        for prim in Usd.PrimRange(stage.GetPrimAtPath(ROBOT_PRIM_PATH)):
            for dt in ["angular", "linear"]:
                drive = UsdPhysics.DriveAPI.Get(prim, dt)
                if drive:
                    drive.GetStiffnessAttr().Set(DRIVE_STIFFNESS)
                    drive.GetDampingAttr().Set(DRIVE_DAMPING)
                    drive.GetMaxForceAttr().Set(DRIVE_MAX_FORCE)
                    drive_count += 1
        print(f"  [OK] Drive 강화: {drive_count} drives "
              f"(stiffness={DRIVE_STIFFNESS}, damping={DRIVE_DAMPING})")

    def get_observations(self):
        return {}


# ============================================================
# 메인
# ============================================================
def main():
    my_world = World(stage_units_in_meters=1.0)

    task = M0609Task(name="m0609_task")
    my_world.add_task(task)
    my_world.reset()

    print("[렌더 루프 시작 -- Play 버튼을 눌러 확인]\n")
    print("  확인 포인트:")
    print("  - 로봇이 흔들리지 않는가 (Drive 강화)")
    print("  - 카메라가 떨어지지 않는가 (센서 비활성화)")
    print("  - ros2 topic list 에서 /rgb, /depth 확인\n")

    while simulation_app.is_running():
        my_world.step(render=True)
        time.sleep(0.01)

    simulation_app.close()


if __name__ == "__main__":
    main()