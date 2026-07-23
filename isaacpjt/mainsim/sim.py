from isaacsim import SimulationApp

CONFIG = {
    "headless": False,  # True: GUI 없이 실행
    "width": 1280,
    "height": 720,
    "enable": [
        "omni.graph.core",
        "omni.graph.action_nodes",
        "omni.graph.action_nodes_core",
        "isaacsim.ros2.bridge",
    ],
}

simulation_app = SimulationApp(CONFIG)

# ============================================================
# 2. SimulationApp 생성 후 Isaac Sim 모듈 import
# ============================================================

import os
import traceback

import omni.usd
from pxr import Usd

from isaacsim.core.api import World
from isaacsim.core.utils.stage import open_stage

from isaacsim.core.utils.extensions import enable_extension
enable_extension("omni.graph.core")
enable_extension("omni.graph.action_nodes")
enable_extension("omni.graph.action_nodes_core")
enable_extension("isaacsim.ros2.bridge")
enable_extension("omni.physx.graph")
enable_extension("isaacsim.asset.gen.conveyor")
# enable_extension("isaacsim.asset.gen.conveyor.IsaacConveyor")
simulation_app.update()

# ============================================================
# 3. 사용자 설정
# ============================================================

USD_PATH = os.path.expanduser(
    "~/cobot3_ws/isaacpjt/basic/heu/stage_v10.usd"
)

# 작업 대상 Prim 경로 예시
CONVEYOR_GRAPH_PATH = (
    "/World/ReturnCell/Conveyors/Input/"
    "ConveyorTrack/ConveyorBeltGraph"
)

SHOE_ROOT_PATH = "/World/sneakers"


# ============================================================
# 4. Task 클래스
# ============================================================

class StandaloneTask:
    """USD를 불러온 후 실행할 작업을 관리하는 클래스."""

    def __init__(self, world: World, stage: Usd.Stage):
        self.world = world
        self.stage = stage

        self.step_count = 0
        self.elapsed_time = 0.0

        # Task 상태
        self.task_started = False
        self.task_finished = False

        # 필요한 Prim/Attribute를 저장할 변수
        self.conveyor_graph_prim = None
        self.velocity_attribute = None
        self.shoe_prim = None

    def setup(self) -> None:
        """시뮬레이션 시작 전에 한 번 실행되는 초기 설정."""

        print("[Task] 초기 설정 시작")

        # ----------------------------------------------------
        # 컨베이어 그래프 Prim 확인
        # ----------------------------------------------------
        self.conveyor_graph_prim = self.stage.GetPrimAtPath(
            CONVEYOR_GRAPH_PATH
        )

        if not self.conveyor_graph_prim.IsValid():
            raise RuntimeError(
                f"컨베이어 Prim을 찾을 수 없습니다: "
                f"{CONVEYOR_GRAPH_PATH}"
            )

        # Conveyor Action Graph의 Velocity 변수
        self.velocity_attribute = (
            self.conveyor_graph_prim.GetAttribute(
                "variables:Velocity"
            )
        )

        if not self.velocity_attribute.IsValid():
            raise RuntimeError(
                "Velocity Attribute를 찾을 수 없습니다: "
                f"{CONVEYOR_GRAPH_PATH}.variables:Velocity"
            )

        # ----------------------------------------------------
        # 신발 Prim 확인
        # ----------------------------------------------------
        self.shoe_prim = self.stage.GetPrimAtPath(
            SHOE_ROOT_PATH
        )

        if not self.shoe_prim.IsValid():
            print(
                f"[Task 경고] 신발 Prim을 찾을 수 없습니다: "
                f"{SHOE_ROOT_PATH}"
            )

        print("[Task] 초기 설정 완료")

    def start(self) -> None:
        """Task 시작 시 한 번 실행."""

        if self.task_started:
            return

        print("[Task] 작업 시작")

        self.set_conveyor_velocity(1.0)

        self.task_started = True

    def update(self, dt: float) -> None:
        """시뮬레이션 매 프레임마다 실행."""

        if self.task_finished:
            return

        if not self.task_started:
            self.start()

        self.step_count += 1
        self.elapsed_time += dt

        # ----------------------------------------------------
        # Task 동작 작성 영역
        # ----------------------------------------------------
        #
        # 예시:
        # 시작 직후 컨베이어 속도 1.0
        # 1.5초가 지나면 컨베이어 정지
        #
        # 실제 Task에 맞게 이 부분을 변경하면 됩니다.
        # ----------------------------------------------------

        if self.elapsed_time >= 1.5:
            self.set_conveyor_velocity(0.0)
            self.finish()

    def finish(self) -> None:
        """Task가 끝났을 때 한 번 실행."""

        if self.task_finished:
            return

        print(
            f"[Task] 작업 완료: "
            f"{self.elapsed_time:.2f}초, "
            f"{self.step_count} steps"
        )

        self.task_finished = True

    def shutdown(self) -> None:
        """프로그램 종료 직전에 실행."""

        print("[Task] 종료 처리")

        # 안전하게 컨베이어 정지
        self.set_conveyor_velocity(0.0)

    def set_conveyor_velocity(self, velocity: float) -> None:
        """컨베이어 속도를 변경."""

        if self.velocity_attribute is None:
            print("[Task 경고] Velocity Attribute가 없습니다.")
            return

        success = self.velocity_attribute.Set(float(velocity))

        if success:
            print(f"[Conveyor] 속도 변경: {velocity}")
        else:
            print(f"[Conveyor 오류] 속도 변경 실패: {velocity}")


# ============================================================
# 5. USD Stage 로드
# ============================================================

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

    # Stage를 연 직후 몇 프레임 진행
    for _ in range(50):
        simulation_app.update()

    stage = usd_context.get_stage()

    if stage is None:
        raise RuntimeError("USD Stage 열기 실패")

    print(f"[Stage] 로딩 완료: {stage.GetRootLayer().identifier}")

    return stage


# ============================================================
# 6. Main
# ============================================================

import omni.timeline

def main() -> None:
    world = None

    try:
        stage = load_usd_stage(USD_PATH)

        # Stage 로딩과 그래프 초기화 완료
        for _ in range(5):
            simulation_app.update()

        world = World(stage_units_in_meters=1.0)

        print("[World] reset")
        world.reset()

        # PhysX 및 OmniGraph 초기화
        for _ in range(5):
            simulation_app.update()

        timeline = omni.timeline.get_timeline_interface()

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