"""AMR(nova_carter_ur5e_surface_gripper) 로봇팔의 pick(픽업)/place(랙 배치) 자동화.

FMS(fms_node.py)가 로봇이 픽업 지점에 도착하면 /fms/amr_ready(이번 트립에 실을
박스 수 count 포함)를, 랙 목표 지점에 도착하면 /fms/amr_carrying_complete를
(박스 1개당 1번씩, 트립당 여러 번) 발행한다. simulation_node.py가 이 두 토픽을
감지해서 여기 있는 AmrArmController로 실제 로봇팔(RMPflow 기반, UR5E/rmpflow/
의 검증된 pick&place 파이프라인 재사용)을 움직이고, 끝나면 /sim/pick_done,
/sim/place_done을 발행해서 FMS에 알린다(fms_node.py는 이미 이 두 토픽을
구독해서 "loading"/"dwelling" 상태를 해제하도록 만들어져 있다).

pick과 place는 서로 다른 시점(그 사이에 AMR이 실제로 픽업→랙까지 주행)에
트리거된다. AMR 뒤에 붙은 StorageBox(build_v3_assembler.py 참고, chassis_link에
로컬 오프셋만으로 고정된 5슬롯짜리 선반 - fms_node.py의 MAX_SHOES_PER_TRIP과
동일)를 중간 적재소로 써서, 사이클 하나하나는 AMR이 정지해 있는 동안(픽업
대기 중, 랙 도착 후) 완결되는 독립적인 pick&place로 나눈다:

1. amr_ready(count)가 오면: count번 반복해서 - "pallet"(픽업대) 위에 새 박스를
   스폰 -> 흡착 -> 비어있는 슬롯 위에 내려놓기 -> 그 슬롯에 FixedJoint로
   용접(AMR 주행 중 안 떨어지게) - 전부 끝나면 /sim/pick_done 발행.
2. amr_carrying_complete(shelf_num 포함)가 올 때마다(그 사이 AMR은 실제로
   랙까지 주행했다): 그 시점의 AMR 실제 pose로 RMPflow base pose/장애물을
   갱신 -> 가장 먼저 채워진(FIFO) 슬롯의 용접 해제 -> 그 박스를 다시 흡착 ->
   shelf_num에 해당하는 pallet(0=pallet_01/280, 1=pallet_02/260,
   2=pallet_03/240) 위에 내려놓기(그 자리에 그대로 둔다 - 랙에 쌓이는 모습을
   보여주기 위함) -> /sim/place_done 발행.

pick/place 목표는 demo0725.usd의 실제 pallet prim(PICK_PALLET_PATH,
PLACE_PALLET_PATHS)을 쓴다 - fleet_config_test1.py의 내비게이션 노드 좌표는
이 pallet들의 "중심"이 아니라 통로 쪽 정차 위치라서(2026-07-26 확인, pallet
크기의 절반만큼 떨어져 있음), pallet의 중심이 아니라 AMR과 가장 가까운 윗면
가장자리 지점을 목표로 계산한다(_nearest_point_on_pallet_top).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import omni.usd
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

from isaacsim.core.utils.types import ArticulationAction

_RMPFLOW_DIR = Path(__file__).resolve().parent.parent / "UR5E" / "rmpflow"
if str(_RMPFLOW_DIR) not in sys.path:
    sys.path.insert(0, str(_RMPFLOW_DIR))

from ur5e_surface_gripper_pick_place_controller import PickPlaceController  # noqa: E402
from trigger_surface_gripper import TriggerSurfaceGripper  # noqa: E402

CONE_CONTACT_LOCAL_OFFSET = (0.0, 0.0, -0.0015)
# event1(pick 하강)/event4(pick 후 상승)는 원래 0.0015/0.00125로 매우 느렸다 -
# RMPflow가 pallet 근처에서 불안정해지던 시절(관절 속도 폭주) 안전 마진으로
# 낮춰둔 값인데, end_effector_initial_height를 pallet 높이에 맞게 고친 뒤로는
# (2026-07-26) 그 불안정 자체가 해소돼서 더 빨라도 된다 - 사용자가 "내려갈 때/
# 올라올 때 느리다"고 확인, 두 phase를 3배 빠르게 올렸다. event1을 다시
# 느리게(0.0025) 낮춰도 접촉 시 박스가 밀리는 정도(XY로 약 2.5cm)가 거의
# 똑같아서(헤드리스로 실측 비교) 하강 속도 자체가 원인은 아닌 것으로
# 보인다 - 트리거 콘의 접촉 판정 허용 오차 때문에 생기는 것으로 보이고,
# 속도를 늦춰도 줄어들지 않으니 굳이 느리게 할 필요가 없다.
EVENTS_DT = [
    0.008,  # 0. 접근 이동
    0.01,   # 1. 하강
    0.5,    # 2. 그리퍼 닫기 대기
    1,      # 3. 그리퍼 닫힘 유지
    0.008,  # 4. 들어올리기
    0.008,   # 5. Place 위치로 이동
    0.01,   # 6. 하강
    1,      # 7. 그리퍼 열기 대기
    0.01,   # 8. 상승
    0.5   # 9. 복귀
    ]
# event1(하강)/4(들어올리기)/8(상승)은 이 안의 시간 배분이 "이 정도면
# 도달하겠지"라는 보수적 예산일 뿐이라, 그리퍼(콘 접촉점)가 실제로 그
# phase의 목표 높이에 이 오차 이내로 가까워지면 남은 스텝을 기다리지 않고
# 바로 다음 event로 넘어간다(event6의 _carried_box_touches_landing_surface
# 조기 종료와 같은 방식) - 특히 event4는 500스텝(~8.3초)이나 잡혀있어서
# 효과가 크다. event0/5/9(수평 이동)와 event2/3/7(정지/그리퍼 액션)은
# "높이 도달"이라는 개념이 없어서 대상에서 제외한다.
EARLY_EXIT_HEIGHT_TOL_M = 0.01
# 표면보다 살짝 아래를 목표로 잡아 콘 트리거 접촉을 보장한다. 흡착 순간
# 박스가 밀리며 최대 17도까지 회전하는 문제가 있어(2026-07-26, 사용자 확인)
# 두 가지를 시도했지만 둘 다 실패:
#  1) GripperBase/wrist_3_link에 UsdPhysics.FilteredPairsAPI로 박스와의
#     콜리전만 선택적으로 꺼봤더니, PhysX가 이 필터를 리지드바디 단위로
#     적용해서 콘 트리거 감지 자체가 죽어버렸다(픽업이 전부 pick_failed로
#     바뀜 - 되돌림).
#  2) 이 오프셋 자체를 -0.004로 줄여봤지만 회전 드리프트가 전혀 줄지 않았다
#     (여전히 ~17도) - 즉 "얼마나 깊이 파고드는가"가 원인이 아니라는 뜻.
# 원인이 다른 곳(콘 형태/접촉 시점의 비대칭 등)에 있는 것으로 보여 원래
# 값으로 되돌렸다 - 픽업 성공률 자체엔 영향 없음(용접은 항상 성공), 자세
# 일관성 문제만 남아있다.
PICK_DESCEND_OFFSET_Z = -0.015
# 직전 반복(흡착/용접)이 끝난 직후엔 그 충격(jolt)으로 섀시가 아주 잠깐
# 흔들릴 수 있다 - 그 순간에 바로 다음 반복의 섀시 pose를 읽으면(단 1프레임
# 스냅샷) 목표 위치가 완전히 엉뚱하게 계산되는 회귀가 실제로 있었다(헤드리스로
# 확인 - 보관함 목표 z가 정상 범위 밖으로 튐 -> 흡착 실패). 다음 반복 시작 전
# 이만큼 프레임 동안 아무 것도 안 하고 흔들림이 가라앉기를 기다린다.
COOLDOWN_STEPS = 0

# 사이클이 끝난 뒤 팔이 멈추는 관절 자세는(RMPflow가 스스로 고른 경로라)
# 매번 조금씩 다르다 - 그 애매한 자세에서 다음 목표(특히 pallet처럼 높은
# 곳)를 향해 IK를 풀면 가끔 특이점 근처에서 RMPflow가 불안정해져서 관절
# 속도가 초당 10rad/s 이상으로 치솟고, 그 반작용으로 AMR 차체가 실제로
# 밀리는 문제가 헤드리스로 확인됐다. 그래서 매 반복이 끝날 때마다 RMPflow를
# 거치지 않고 직접 관절 제어로 항상 같은 스폰 홈 자세로 확실히 복귀시킨
# 뒤에만 다음 목표의 IK를 시작한다.
UR5E_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
SPAWN_JOINT_ANGLES_DEG = [180, -90, 90, -90, -90, 0]  # build_v3_assembler.py와 동일해야 함
RETURN_HOME_STEPS = 70
RETURN_HOME_TOLERANCE_RAD = 0.05

# pallet처럼 멀리/높이 있는 목표를 향할 때 RMPflow가 가끔(특이점 근처로
# 보임) 불안정해져서 관절 속도가 초당 10rad/s 이상으로 치솟는 게 헤드리스로
# 확인됐다(장애물 회피를 꺼도 재현 - RMPflow 자체 불안정) - 원인을 RMPflow
# 내부에서 고치는 대신, 매 스텝 명령을 이 이상 못 튀게 직접 제한한다.
MAX_JOINT_STEP_RAD = 1  # 60Hz 기준 대략 7.2rad/s에 해당(기존 0.03=1.8rad/s에서 완화)

# 위의 여러 안전장치(팔레트 장애물 등록, 홈 복귀, 속도 클램프)를 적용해도
# RMPflow가 가끔 이번 시도에서만 흡착에 실패할 수 있다 - "픽업은 무조건
# 성공해야 한다"는 요구사항을 실질적으로 만족시키기 위해, 실패한 슬롯은
# 건너뛰지 않고 이 횟수만큼 같은 자리에서 재시도한다.
MAX_PICK_RETRIES = 3

DEFAULT_BOX_SIZE = np.array([0.28, 0.20, 0.11])

# StorageBox(build_v3_assembler.py의 STORAGE_BOX_LOCAL_POS/SIZE와 반드시 같이
# 맞춰야 한다) 위 6칸(2개씩 3층)의 chassis-local 안착 위치 - fms_node.py의
# MAX_SHOES_PER_TRIP(5)보다 1칸 여유. 폭(Y)은 차체 폭과 같아서 옆으로는 2개
# (±0.13, 박스폭 0.23 기준 살짝 여유)만 놓고, 그 위로 박스를 쌓아 올린다(이송
# 중엔 FixedJoint로 용접되니 실제로 박스 위에 박스를 얹어도 안 떨어진다).
# 층 간격 = 박스 높이(0.13) + 여유(0.02). 아래층부터 순서대로 채운다.
STORAGE_LAYER_HEIGHT = 0.15
# "저장소까지 안 내려가짐"(박스가 선반보다 뜬 채로 놓임) 문제는 두 가지
# 원인이 겹쳐 있었다: (1) StorageBox 선반이 chassis_link/visual(차체
# 몸체, 진짜 콜라이더)과 실제로 겹쳐 있던 것 - nova_carter_ur5e_surface_
# gripper.usd의 StorageBox를 X로 0.2m 뒤로 옮겨서(-0.65 -> -0.85)
# 해결(move_storagebox.py, 2026-07-27). (2) 그것과 별개로, 그 자리에서
# 팔이 실제로 내려갈 수 있는 RMPflow 한계가 여전히 있어서(자체 충돌
# 회피로 보임), EVENTS_DT를 바꿔서 하강 속도가 달라질 때마다 실제 도달
# 높이도 같이 바뀐다 - 그래서 층별/슬롯별 Z 보정이 계속 필요하다. 목표를
# 낮추면 오히려 격차만 벌어지고(실측 확인됨), 실측된 "실제 도달 높이"에
# 맞춰 목표를 올려야 한다. check_all_slots_gap.py로 슬롯 6개 전부
# 재실측해서 보정했다(2026-07-27, EVENTS_DT 변경 후 재조정).
#
# 2026-07-27 재보정(1차, diag_storage_place.py로 apex_z-target_z만 보고 Newton
# 스텝 comp_new = comp_old + gap 적용): "target_z 자체가 이미 맞다"고 가정하고
# apex가 target에 가까워지도록만 맞췄는데, 그 전제가 틀렸다 - GUI로 직접
# 보니 박스가 StorageBox 판 위에서 한참(10cm 이상) 뜬 채로 놓이고 있었다
# (사용자 확인, 스크린샷). StorageBox 콜리전 큐브의 실측 로컬 Z는 0.325~0.375
# (윗면 0.375)인데, 그 시점 보정값(+0.08~0.15)까지 다 더한 레이어0 목표
# 로컬 Z는 0.53~0.58 - 애초에 목표 자체가 실제 판 윗면보다 16cm 이상 위에
# 있었다(사용자 지적 "기준값 낮춰도 되니까 임시 저장소 위에 놓이게 해줘").
# base(0.44)는 판 윗면(0.375)+박스 반두께(0.055)=0.43과 이미 거의 맞아서
# 건드릴 필요가 없었고, 진짜 문제는 그 위에 잘못 누적된 보정값이었다 -
# 그래서 보정값을 판 윗면 기준 실제로 필요한 수준(0에 가깝게)으로 초기화하고
# check_all_slots_gap 방식으로 재실측했다. 0으로 두니 레이어0(판에 직접
# 얹히는 슬롯0/1)이 판 실측 윗면(월드 z~1.37)보다 여전히 2~3cm 떠서,
# 6칸 전부에 -0.026(층 간 간격은 그대로 두고 절대 기준만 살짝 더 낮춤)을
# 균일하게 적용했다 - 레이어1/2는 아래 박스 위에 쌓이는 자리라 이미 잘
# 맞고 있어서 상대 간격은 건드리지 않는다.
_STORAGE_SLOT_Z_COMPENSATION = [-0.026, -0.026, -0.026, -0.026, -0.026, -0.026]
# event6(하강)이 Z뿐 아니라 XY도 고정 시간 예산 안에서 보간하는데, 그 시간이
# 끝날 때 XY 수렴이 덜 끝난 채로 놓아버리는 경우가 있다(2026-07-27, 사용자
# 확인 - "쏠려있다"/"조기에 내려놓는다"). diag_early_release_xy.py로 event7
# 진입 시점의 실제 박스 world XY와 목표 world XY 차이를 실측한 뒤, 챗시
# 회전(yaw)만큼 역회전시켜 chassis-local (dx, dy) 보정값으로 변환해서 더한다
# - STORAGE_SLOT_LOCAL_POSITIONS 자체가 chassis-local 좌표라서, world 오차를
# 그대로 더하면 로봇 방향에 따라 엉뚱한 방향으로 보정되기 때문에 반드시
# 로컬로 변환해야 한다.
# 2026-07-27 재측정(calibrate_xy.py, _anchor_chassis로 챗시 drift를 완전히
# 제거한 뒤): chassis_yaw가 45.0도로 완벽하게 고정된 상태에서 측정했고, 두 번
# 반복해도 소수점 4자리까지 완전히 동일해서(결정론적) drift가 아니라 순수한
# RMPflow 잔여 수렴 오차다.
#
# 컬럼(col) 단위로만 보정한다 - 슬롯이 layer*2+col 순서라 같은 col(0,2,4
# 또는 1,3,5)은 원래 XY가 완전히 같아야(레이어만 위로 쌓임) 하는데, 처음에
# 슬롯 6개를 각각 독립적으로 보정했더니 같은 컬럼인데도 층마다 XY가 미세하게
# 달라져서 위층이 아래층 위에 깔끔히 안 얹히는 문제가 있었다(2026-07-27,
# 사용자 지적 - "1,3,5번은 XY 같아야 하는 거 아니야?"). 그래서 각 컬럼의
# 레이어0(그 컬럼의 기초, 실제 판에 직접 얹히는 자리) 실측값을 그 컬럼
# 전체(레이어1,2)에도 동일하게 적용한다 - Newton 반복으로 한 번 더
# 다듬으려 했더니 층 간 상호의존성 때문에 오차가 폭발적으로 커진 적이
# 있어서(14~38cm), 지금은 1차 실측값만 쓴다.
# 2026-07-27: 두 컬럼 사이 간격을 넓혀달라는 요청(사용자 확인 - GUI 스크린샷
# "0/2/4와 1/3/5 사이에 간격 두기") - check_column_gap.py로 실측한 StorageBox
# 로컬 bbox(Y: -0.24~0.24)와 실제 박스 위치를 비교하니, col1(슬롯1/3/5)은
# 오른쪽 가장자리까지 여유가 0.5cm뿐이라 못 움직이고, col0(슬롯0/2/4)은
# 왼쪽 가장자리까지 5.7cm 여유가 있었다. col1은 그대로 두고 col0만 왼쪽으로
# 더 밀어서(Y 보정을 0.0249 -> -0.02로, 약 4.5cm) 간격을 벌렸다 - 실측
# 재확인 필요(check_column_gap.py).
_STORAGE_SLOT_XY_COMPENSATION_PER_COL = [
    (0.0256, -0.02),  # col0 (슬롯0/2/4) - 슬롯0(레이어0) 실측 + 간격 확보용 추가 이동
    (0.0084, 0.0026),  # col1 (슬롯1/3/5) - 슬롯1(레이어0) 실측, 가장자리에 가까워 그대로 유지
]
STORAGE_SLOT_LOCAL_POSITIONS = [
    np.array([
        -0.85 + _STORAGE_SLOT_XY_COMPENSATION_PER_COL[col][0],
        y + _STORAGE_SLOT_XY_COMPENSATION_PER_COL[col][1],
        0.44 + layer * STORAGE_LAYER_HEIGHT + _STORAGE_SLOT_Z_COMPENSATION[layer * 2 + col],
    ])
    for layer in range(3)
    for col, y in enumerate((-0.13, 0.13))
]

# 실제 demo0725.usd의 pallet prim들을 pick/place 대상으로 쓴다(2026-07-26
# 확정): 픽업은 "pallet", 랙 배치는 크기별로 pallet_01(280)/pallet_02(260)/
# pallet_03(240) - fleet_config_test1.py의 PICKUP_A/RackA_280/RackA_260/
# RackA_240과 각각 짝을 이룬다(같은 순서로 SHELF_INDEX가 매겨짐 - fms_node.py
# 참고). fleet_config 노드 좌표는 pallet의 "중심"이 아니라 아이슬 쪽 통로
# 위치라 pallet bbox 중심과는 X는 거의 일치하지만 Y가 pallet 크기(약 1.3m)의
# 절반만큼 떨어져 있다 - 그래서 pallet의 중심이 아니라 AMR과 가장 가까운
# 윗면 가장자리 지점을 목표로 계산한다(_nearest_point_on_pallet_top 참고).
PICK_PALLET_PATH = "/World/pallet"
PLACE_PALLET_PATHS = ["/World/pallet_01", "/World/pallet_02", "/World/pallet_03"]
# pick pallet 바로 옆(약 1.2m)에 실제 컨베이어(convey_01)가 있어서 "옆 컨베이어와
# 부딪힘" 문제가 있었다(2026-07-26). 등록을 아예 껐다가 헤드리스로 비교해보니
# (관절 속도 최대치/완료까지 걸리는 스텝 수 모두) 등록 안 한 쪽이 오히려 더
# 나빴다(등록: ~5100스텝/최대 3.1rad/s, 미등록: ~7000스텝/최대 7rad/s) - 즉
# 이 장애물이 실제로는 안정화에 도움이 된다. 등록은 유지하고, "부딪힘"으로
# 보인 것이 실제 충돌인지 그냥 카메라 각도상 가까워 보이는 정상 이동인지는
# 시각적으로 다시 확인이 필요하다.
EXTRA_STATIC_OBSTACLE_PATHS = ["/World/ReturnCell/convey_01/ConveyorTrack"]  # shelf_num 0/1/2 = 280/260/240
# pallet 가장자리 바로 그 지점은 팔 자신의 정지 자세와 겹칠 수 있어 안쪽으로
# 이만큼 들여서 목표를 잡는다(GUI에서 확인된 회귀 - 박스가 스폰되자마자 팔과
# 충돌해 튕겨나감).
PALLET_EDGE_MARGIN_M = 0.30
# pallet 장애물의 윗면을 이만큼 깎아내린다 - 안 그러면 회피 장애물의 윗면과
# 박스를 놓는/집는 목표 지점(pallet 윗면 바로 위)이 거의 같은 높이라 "닿아야
# 한다"와 "피해야 한다"가 충돌해서 팔이 아예 접근을 거부하는 문제가 있었다
# (헤드리스로 확인 - 장애물 등록 직후 pick이 전부 실패로 바뀜).
PALLET_OBSTACLE_TOP_MARGIN_M = 0.20
# 이미 채워진 슬롯(용접된 박스)도 RMPflow 회피 장애물로 등록할 때 윗면을
# 이만큼만 살짝 깎는다 - pallet(0.20)과 달리 박스 하나 높이가 0.11m밖에
# 안 돼서 그렇게 큰 마진을 쓰면 장애물 자체가 없어져 버린다. 다음 박스가
# 바로 이 윗면에 붙어 놓여야 하는 스태킹 목표라, 살짝만 깎아서 "닿아야
# 한다"와 "피해야 한다"가 충돌하지 않게 한다(위 pallet과 같은 이유,
# 크기만 다름 - 실측 후 조정 필요할 수 있음).
FILLED_SLOT_OBSTACLE_TOP_MARGIN_M = 0.03

# pallet은 벽면만 있고 위가 뚫린 케이지 구조라, 팔이 옆에서 접근하면 실제
# 벽에 스치기 쉽다(GUI에서 확인) - RMPflow의 장애물 회피에 통째로 맡기는
# 대신, 먼저 벽 높이보다 확실히 위(목표와 같은 XY, 벽 위)로 이동시킨 뒤
# 그 지점에서 수직으로만 내려가게 직접 경로를 지정한다.
SAFE_HOVER_MARGIN_M = 0.30
HOVER_STEP_BUDGET = 300
HOVER_TOLERANCE_M = 0.03


def prepare_cone_triggers(robot_prim_path: str) -> None:
    """Cone_0~3(+Plate)에 PhysxTriggerStateAPI를 붙여 접촉 판정을 가능하게 한다.

    반드시 world.reset()(첫 physics step) 이전에 호출해야 한다 - PhysX가
    최초 reset 시점에 콜리전/트리거 표현을 스냅샷하는 것으로 보이며, reset
    이후에 API를 붙이면 HasAPI()는 True로 보여도 실제 겹침 조회
    (GetTriggeredCollisionsRel)가 영원히 빈 리스트만 반환한다(헤드리스로
    직접 확인한 회귀 - reset 전/후로 나눠 비교했을 때 전자만 정상 동작).

    GripperBase 밑의 "Plate"(그리퍼 몸체 자체)는 원래 에셋에 PhysxCollisionAPI +
    PhysxTriggerAPI까지는 붙어있는데 PhysxTriggerStateAPI만 빠져 있었다
    (2026-07-26, check_plate_schemas.py로 확인 - Cone_0과 스키마 목록이
    똑같은데 이것만 없음). 위와 같은 이유로 StateAPI 없이는 실제로는 트리거로
    동작하지 않고 진짜 솔리드 콜라이더처럼 움직여서, 흡착 하강 시 GripperBase
    몸체(Plate)가 박스를 파고들며 최대 17도까지 밀고 돌려버리는 문제가 있었다
    (사용자 확인). Cone과 똑같이 StateAPI를 붙여주면 원래 의도대로 순수
    트리거가 되어 이 문제가 사라진다(헤드리스로 검증 - 회전 드리프트
    0.03~0.06도로 감소, 픽업 성공률도 그대로 유지 - FilteredPairsAPI로
    시도했을 때와 달리 Cone 자체의 트리거 감지엔 영향 없음).
    """
    from pxr import PhysxSchema

    stage = omni.usd.get_context().get_stage()
    gripper_base_path = f"{robot_prim_path}/arm_mount/ur5e/GripperBase"
    trigger_child_names = ["Cone_0", "Cone_1", "Cone_2", "Cone_3", "Plate"]
    for name in trigger_child_names:
        child_prim = stage.GetPrimAtPath(f"{gripper_base_path}/{name}")
        if child_prim.IsValid() and not child_prim.HasAPI(PhysxSchema.PhysxTriggerStateAPI):
            PhysxSchema.PhysxTriggerStateAPI.Apply(child_prim)


class AmrArmController:
    """AMR 한 대(robot_id)의 로봇팔 pick/place 상태 머신.

    phase: "idle" -> "pick_to_storage"(count번 내부 반복) -> "idle" ->
    "place_from_storage" -> "idle"
    """

    def __init__(
        self,
        robot_id: str,
        robot_prim_path: str,
        box_size: np.ndarray = DEFAULT_BOX_SIZE,
        physics_sim_view=None,
    ) -> None:
        self.robot_id = robot_id
        self._robot_prim_path = robot_prim_path
        self._chassis_path = f"{robot_prim_path}/chassis_link"
        self._ur5e_base_link_path = f"{robot_prim_path}/arm_mount/ur5e/base_link"
        self._gripper_base_path = f"{robot_prim_path}/arm_mount/ur5e/GripperBase"
        self._storage_box_path = f"{self._chassis_path}/StorageBox"
        self._cone_paths = [f"{self._gripper_base_path}/Cone_{i}" for i in range(4)]
        self._grasp_joint_path = f"{self._gripper_base_path}/GraspJoint"
        self._box_size = np.array(box_size, dtype=float)

        from isaacsim.core.prims import SingleArticulation

        # NOTE: prepare_cone_triggers(robot_prim_path)를 world.reset() 전에 이미
        # 호출해뒀어야 한다 - 여기(생성자, reset 이후에 호출됨)서 다시 적용해봐야
        # 늦다. 안전망으로 한 번 더 시도는 하되, 이게 실제로 필요했다면 이미 늦은
        # 것이니 호출하는 쪽의 순서를 고쳐야 한다.
        prepare_cone_triggers(robot_prim_path)

        self._robot = SingleArticulation(prim_path=self._chassis_path, name=f"{robot_id}_articulation")
        self._robot.initialize()

        dof_names = list(self._robot.dof_names)
        self._arm_joint_indices = np.array([dof_names.index(name) for name in UR5E_JOINT_NAMES])
        self._arm_home_positions = np.array([np.deg2rad(a) for a in SPAWN_JOINT_ANGLES_DEG], dtype=float)

        # 콘 트리거가 우리가 스폰한 박스(_pick_box_{robot_id}_*) 말고 pallet
        # 벽면 같은 주변 지오메트리와도 겹칠 수 있어서, 실제로 흡착/접촉으로
        # 인정할 대상을 이 패턴으로 제한한다(GUI에서 pallet 벽면을 흡착해버린
        # 사고 확인 후 추가).
        #
        # 2026-07-27: 이 패턴이 "_pick_box_{robot_id}_" 접두사만 봐서, 이미
        # StorageBox에 용접돼 저장소에 놓인 이전 박스들까지 전부 통과시키고
        # 있었다 - 다음 박스를 픽업하러 팔레트로 이동하는 도중 콘이 방금
        # 놓은 박스를 스치기만 해도 "우리 박스니까 접촉 인정"으로 판정돼
        # 그 이미 놓인 박스를 다시 흡착해버리는 사고가 실제로 재현됐다
        # (사용자 확인 - "6번째 픽업 전에 5번째 상자와 다시 결합돼서 6번째가
        # 안 옮겨짐"). 지금 사이클이 실제로 다루고 있는 박스(
        # self._current_box_prim_path) 하나만 허용하도록 좁힌다 - 그 값은
        # start_pick/start_place가 매 사이클 새로 설정하므로, 이전에 이미
        # 놓인 박스는 자동으로 제외된다.
        def _is_our_pick_box(prim_path: str) -> bool:
            return (
                self._current_box_prim_path is not None
                and self._current_box_prim_path in prim_path
            )

        self._target_filter = _is_our_pick_box

        self._gripper = TriggerSurfaceGripper(
            end_effector_prim_path=self._gripper_base_path,
            trigger_paths=self._cone_paths,
            joint_path=self._grasp_joint_path,
            tip_local_offset=CONE_CONTACT_LOCAL_OFFSET,
            target_filter=self._target_filter,
        )
        self._gripper.initialize(
            physics_sim_view=physics_sim_view,
            articulation_num_dofs=self._robot.num_dof,
        )
        self._gripper.set_default_state(opened=True)
        self._gripper.post_reset()

        base_position, base_orientation = self._read_base_pose()
        # exclude_names=("StorageBox",)로 StorageBox 자체는 챗시 장애물 bbox
        # 계산에서 빼지만, 형제 prim인 "visual"(차체 전체를 뭉뚱그린 단일
        # 메시)의 bbox가 StorageBox 실제 높이(~1.40m)보다 훨씬 위인 1.5m
        # 근처까지 그대로 덮고 있어서, 맨 아래 칸(슬롯 목표 1.467m)에 내려놓을
        # 때 RMPflow가 그 상공을 장애물로 착각해 목표보다 한참 위에서 멈추는
        # 문제가 있었다(2026-07-26, 사용자 확인 "저장소까지 안 내려가짐" -
        # check_chassis_obstacle_box.py로 실측). StorageBox의 실제 윗면보다
        # 살짝만 위까지만 챗시 장애물로 인정하도록 상한을 넘긴다.
        stage = omni.usd.get_context().get_stage()
        storage_prim = stage.GetPrimAtPath(self._storage_box_path)
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
        storage_top_z = float(bbox_cache.ComputeWorldBound(storage_prim).ComputeAlignedRange().GetMax()[2])
        self._controller = PickPlaceController(
            name=f"{robot_id}_pick_place_controller",
            gripper=self._gripper,
            robot_articulation=self._robot,
            end_effector_initial_height=0.55,
            events_dt=EVENTS_DT,
            robot_base_position=base_position,
            robot_base_orientation=base_orientation,
            chassis_obstacle_prim_path=self._chassis_path,
            chassis_obstacle_exclude_names=("StorageBox",),
            chassis_obstacle_max_z=storage_top_z + 0.02,
            contact_trigger_path=self._cone_paths,
            contact_target_filter=self._target_filter,
        )
        self._controller.reset()
        self._register_pallet_obstacles()

        self._phase = "idle"
        self._contact_seen = False
        self._grasp_confirmed = False
        self._grasp_rel_rot: Optional[Gf.Quatd] = None
        self._welded_this_rep = False
        self._unwelded_this_rep = False
        self._chassis_anchored = False
        self._retry_count = 0
        self._cycle_target: Optional[np.ndarray] = None

        self._next_box_serial = 0
        # 슬롯 인덱스 -> 그 슬롯에 놓인 박스의 prim path (비어있으면 None)
        self._slot_box_paths: List[Optional[str]] = [None] * len(STORAGE_SLOT_LOCAL_POSITIONS)
        # 채워진 순서(FIFO) - place는 항상 이 리스트의 맨 앞 슬롯부터 뺀다.
        self._slot_fill_order: List[int] = []
        self._current_box_prim_path: Optional[str] = None
        self._current_slot_index: Optional[int] = None
        self._current_shelf_num: Optional[int] = None
        self._pick_remaining = 0
        self._cooldown_remaining = 0
        # cooldown이 끝난 뒤 "pick"(_begin_next_pick_repetition)과
        # "place"(_begin_place_attempt) 중 어느 쪽을 재개할지 - 재시도든
        # 다음 반복이든 이 값으로 분기한다(update()의 cooldown 처리 참고).
        self._cooldown_next_phase = "pick"
        self._return_home_steps_left = 0
        # returning_home 시작 시점의 관절각 - 거기서부터 홈 자세까지 매
        # 프레임 서서히 섞어가며(_step_return_home 참고) 최종 목표를 첫
        # 프레임부터 그대로 명령하지 않도록 한다.
        self._return_home_start_positions: Optional[np.ndarray] = None
        # "returning_home" 단계가 끝난 뒤 실제로 할 일 - "next_rep"(다음 반복
        # 시작) 또는 이번 update() 호출에서 바깥으로 돌려줄 이벤트 문자열.
        self._after_return_home: Optional[str] = None

    # -----------------------------------------------------------------
    # 내부 helper - pose
    # -----------------------------------------------------------------

    def _read_base_pose(self):
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(self._ur5e_base_link_path)
        world = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
        position = np.array(world.ExtractTranslation())
        quat = world.ExtractRotationQuat()
        orientation = np.array([quat.GetReal(), *quat.GetImaginary()])
        return position, orientation

    def _read_chassis_pose(self):
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(self._chassis_path)
        world = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
        position = np.array(world.ExtractTranslation())
        quat = world.ExtractRotationQuat()
        return position, quat

    def _read_box_pose(self, box_prim_path: str) -> np.ndarray:
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(box_prim_path)
        world = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
        return np.array(world.ExtractTranslation())

    def _read_gripper_apex_z(self) -> float:
        """콘 접촉점(그리퍼가 실제로 물건에 닿는 지점) 높이를 실측한다 -
        event1/4/8 조기 종료 판정에 쓴다(아래 update() 참고)."""
        stage = omni.usd.get_context().get_stage()
        gb_world = UsdGeom.XformCache().GetLocalToWorldTransform(
            stage.GetPrimAtPath(self._gripper_base_path)
        )
        apex = gb_world.Transform(Gf.Vec3d(*CONE_CONTACT_LOCAL_OFFSET))
        return float(apex[2])

    def _refresh_base_pose_and_obstacle(self) -> None:
        """AMR이 실제로 이동한 뒤 pick/place를 이어서 하려면, RMPflow가 아는
        base pose와 차체 장애물(spawn 시점에 한 번만 등록된 static obstacle)을
        지금 실제 위치로 다시 맞춰야 한다."""
        position, orientation = self._read_base_pose()
        cspace = self._controller._cspace_controller
        cspace._motion_policy.set_robot_base_pose(robot_position=position, robot_orientation=orientation)

        # RMPFlowController._add_chassis_obstacle와 반드시 같은 경로 규칙(로봇별로
        # 분리)을 써야 한다 - 안 그러면 로봇이 2대 이상일 때 서로 다른 로봇의
        # 챗시 장애물 prim을 덮어써버린다(2026-07-27, amr_2 확장 시 발견).
        obstacle_path = f"/World/_rmpflow_chassis_obstacle_{self.robot_id}"
        stage = omni.usd.get_context().get_stage()
        obstacle_prim = stage.GetPrimAtPath(obstacle_path)
        if obstacle_prim.IsValid():
            from isaacsim.core.api.objects import VisualCuboid

            # RMPFlowController._add_chassis_obstacle와 동일하게 StorageBox는
            # bbox 계산에서 제외한다 - 안 그러면 팔이 실제로 도달해야 하는
            # 적재 선반 자체가 "피해야 할 장애물"이 돼버려 목표와 회피가
            # 충돌한다.
            chassis_prim = stage.GetPrimAtPath(self._chassis_path)
            bbox_cache = UsdGeom.BBoxCache(0, ["default", "render"])
            bmin = None
            bmax = None
            for child in chassis_prim.GetChildren():
                if child.GetName() == "StorageBox":
                    continue
                child_rng = bbox_cache.ComputeWorldBound(child).ComputeAlignedRange()
                child_min, child_max = np.array(child_rng.GetMin()), np.array(child_rng.GetMax())
                if child_min[0] > child_max[0]:
                    continue
                bmin = child_min if bmin is None else np.minimum(bmin, child_min)
                bmax = child_max if bmax is None else np.maximum(bmax, child_max)
            if bmin is not None:
                center = (bmin + bmax) / 2.0
                obstacle = VisualCuboid(prim_path=obstacle_path)
                obstacle.set_world_pose(position=center)

    def _register_pallet_obstacles(self) -> None:
        """pallet(들)을 RMPflow의 정적 회피 장애물로 등록한다.

        chassis_obstacle_prim_path로 섀시는 이미 등록해뒀지만, pallet(케이지
        구조물) 자체는 등록이 안 돼 있어서 팔이 접근/이동하는 도중 실제로
        forearm_link가 pallet과 겹치는(충돌하는) 게 물리 오버랩 쿼리로
        확인됐다 - 그 충격으로 관절 속도가 초당 10rad/s 이상 치솟고 AMR 차체
        까지 밀리는 문제로 이어졌다. pallet은 고정된 static prim이라 여기서
        한 번만 등록하면 된다.
        """
        from isaacsim.core.api.objects import VisualCuboid

        stage = omni.usd.get_context().get_stage()
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
        rmp_flow = self._controller._cspace_controller.rmp_flow
        pick_pallet_bmin = None
        for i, pallet_path in enumerate([PICK_PALLET_PATH] + PLACE_PALLET_PATHS):
            pallet_prim = stage.GetPrimAtPath(pallet_path)
            if not pallet_prim.IsValid():
                continue
            rng = bbox_cache.ComputeWorldBound(pallet_prim).ComputeAlignedRange()
            bmin, bmax = np.array(rng.GetMin()), np.array(rng.GetMax())
            if i == 0:
                pick_pallet_bmin = bmin.copy()
            bmax = bmax.copy()
            bmax[2] = max(bmin[2], bmax[2] - PALLET_OBSTACLE_TOP_MARGIN_M)
            center = (bmin + bmax) / 2.0
            size = bmax - bmin
            obstacle_path = f"/World/_rmpflow_pallet_obstacle_{self.robot_id}_{i}"
            if not stage.GetPrimAtPath(obstacle_path).IsValid():
                obstacle = VisualCuboid(prim_path=obstacle_path, position=center, scale=size, visible=False)
            else:
                obstacle = VisualCuboid(prim_path=obstacle_path)
                obstacle.set_world_pose(position=center)
            rmp_flow.add_cuboid(obstacle, static=True)
            print(f"[amr_pick_place] {self.robot_id}: registered pallet obstacle {obstacle_path} center={center} size={size}", flush=True)

        for j, extra_path in enumerate(EXTRA_STATIC_OBSTACLE_PATHS):
            extra_prim = stage.GetPrimAtPath(extra_path)
            if not extra_prim.IsValid():
                continue
            rng = bbox_cache.ComputeWorldBound(extra_prim).ComputeAlignedRange()
            bmin, bmax = np.array(rng.GetMin()), np.array(rng.GetMax())
            bmax = bmax.copy()
            # 컨베이어(convey_01)의 원본 bbox가 실제로 pick pallet의 bbox와
            # X방향으로 0.7m 가까이 겹친다(2026-07-26 확인) - 그대로 등록하면
            # pick 목표 지점 자체가 이 장애물 안에 들어가버려서 RMPflow가
            # 영원히 목표에 도달 못 하고 헤맨다(헤드리스로 확인 - 관절 속도
            # 계속 치솟고 반복 재시도). pallet 쪽으로는 pick pallet의 가장자리
            # 못 미쳐서 잘라내(clip) 목표 지점을 침범하지 않게 한다.
            if pick_pallet_bmin is not None:
                bmax[0] = min(bmax[0], pick_pallet_bmin[0] - 0.02)
            if bmax[0] <= bmin[0]:
                continue
            center = (bmin + bmax) / 2.0
            size = bmax - bmin
            obstacle_path = f"/World/_rmpflow_extra_obstacle_{self.robot_id}_{j}"
            if not stage.GetPrimAtPath(obstacle_path).IsValid():
                obstacle = VisualCuboid(prim_path=obstacle_path, position=center, scale=size, visible=False)
            else:
                obstacle = VisualCuboid(prim_path=obstacle_path)
                obstacle.set_world_pose(position=center)
            rmp_flow.add_cuboid(obstacle, static=True)
            print(f"[amr_pick_place] {self.robot_id}: registered extra obstacle {obstacle_path} center={center} size={size}", flush=True)

    def _register_filled_slot_obstacles(self) -> None:
        """이미 채워진(용접된) 슬롯의 박스들을 RMPflow 장애물로 등록한다 -
        안 그러면 다음 박스를 쌓거나(pick_to_storage) 꺼낼(place_from_storage)
        때 팔이 이미 있는 박스와 실제로 부딪힐 수 있다(2026-07-27, 사용자
        확인 - "쌓으러 갈 때 다른 박스와 충돌").

        지금 이 사이클이 다루고 있는 박스(self._current_box_prim_path)는
        그 자체가 목표 지점(놓을 자리 또는 집을 대상)이라 반드시 제외한다
        - StorageBox 자체를 챗시 장애물에서 뺀 것과 같은 이유로, 안 빼면
        목표와 회피가 충돌해서 팔이 접근을 거부한다. 윗면은
        FILLED_SLOT_OBSTACLE_TOP_MARGIN_M만큼만 깎아서, 다음 박스가 바로
        위에 붙어 놓이는 스태킹 목표를 침범하지 않게 한다.

        2026-07-27: 그것만으로는 부족했다 - 목표 슬롯 바로 아래층(같은
        컬럼, 이전 레이어 - 스태킹 시 실제로 받침대가 되는 박스)까지
        3cm 마진만으로 깎아서는, 이미 1~2cm까지 딱 붙게 보정해둔 실제
        목표 지점보다 장애물 윗면이 오히려 더 높이 남는 경우가 있어서
        "닿아야 한다"와 "피해라"가 다시 충돌했다(GUI에서 확인 - 그리퍼가
        상자에 도달하지 못하고 옆으로 휘어짐). 슬롯 배치가
        "layer*2+col" 순서라 슬롯 i의 받침대는 항상 슬롯 i-2이므로,
        그 받침대 슬롯도 통째로 장애물에서 제외한다 - 받침대 박스는
        실제로 목표 지점과 맞닿아야 하는 표면이라, 어차피 장애물로
        등록해봐야 회피가 아니라 충돌만 일으킨다."""
        from isaacsim.core.api.objects import VisualCuboid

        support_slot_index = (
            self._current_slot_index - 2
            if self._current_slot_index is not None and self._current_slot_index >= 2
            else None
        )

        stage = omni.usd.get_context().get_stage()
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
        rmp_flow = self._controller._cspace_controller.rmp_flow
        for slot_index, box_path in enumerate(self._slot_box_paths):
            obstacle_path = f"/World/_rmpflow_slot_obstacle_{self.robot_id}_{slot_index}"
            if (
                box_path is None
                or box_path == self._current_box_prim_path
                or slot_index == support_slot_index
            ):
                # 비어있거나, 지금 다루는 대상이거나, 그 받침대 슬롯이면
                # 등록하지 않는다 - 이전 사이클에서 등록된 잔여물이 있어도
                # reset()이 매번 싹 지워서 다시 등록 안 하면 자동으로
                # 없어진다(추가 정리 불필요).
                continue
            box_prim = stage.GetPrimAtPath(box_path)
            if not box_prim.IsValid():
                continue
            rng = bbox_cache.ComputeWorldBound(box_prim).ComputeAlignedRange()
            bmin, bmax = np.array(rng.GetMin()), np.array(rng.GetMax())
            if bmin[0] > bmax[0]:
                continue
            bmax = bmax.copy()
            bmax[2] = bmax[2] - FILLED_SLOT_OBSTACLE_TOP_MARGIN_M
            # 깎아낸 뒤 두께가 0 이하(또는 bbox 자체가 이 프레임에 아직
            # 유효하게 안 잡힌 경우)면 VisualCuboid의 scale에 0을 넣게 돼서
            # 변환행렬이 특이(singular)해지고 get_world_pose가 그대로
            # 터진다(2026-07-27, 헤드리스로 실제 크래시 확인) - 이 프레임은
            # 그냥 건너뛰고 다음 사이클에서 다시 시도한다.
            if bmax[2] <= bmin[2] or np.any(bmax - bmin <= 0):
                continue
            center = (bmin + bmax) / 2.0
            size = bmax - bmin
            if not stage.GetPrimAtPath(obstacle_path).IsValid():
                obstacle = VisualCuboid(prim_path=obstacle_path, position=center, scale=size, visible=False)
            else:
                obstacle = VisualCuboid(prim_path=obstacle_path)
                obstacle.set_world_pose(position=center)
            rmp_flow.add_cuboid(obstacle, static=True)
            print(
                f"[amr_pick_place] {self.robot_id}: registered filled-slot obstacle "
                f"{obstacle_path}(slot={slot_index}) center={center} size={size}",
                flush=True,
            )

    def _pallet_yaw(self, pallet_path: str) -> float:
        """pallet의 world-frame yaw(라디안, Z축 기준)를 계산한다.

        place pallet들(pallet_01/02/03)은 축 정렬(yaw=0)이지만, pick
        pallet(/World/pallet)은 실제로 45도 회전되어 있다(2026-07-26 확인,
        check_pallet_yaw2.py로 world_x_axis 직접 측정). 박스를 world 축에
        맞춰(회전 없이) 스폰하면 실제로는 마름모꼴로 놓인 pallet 바닥
        모서리 밖으로 박스 모서리가 삐져나가 AMR 쪽으로 떨어지는 문제가
        있었다(사용자 확인) - 스폰 방향을 이 yaw만큼 맞춰 돌려야 한다.
        """
        stage = omni.usd.get_context().get_stage()
        pallet_prim = stage.GetPrimAtPath(pallet_path)
        mat = UsdGeom.Xformable(pallet_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        x_axis = mat.TransformDir(Gf.Vec3d(1.0, 0.0, 0.0))
        return float(np.arctan2(x_axis[1], x_axis[0]))

    def _nearest_point_on_pallet_top(self, pallet_path: str) -> np.ndarray:
        """pallet의 윗면에서 AMR(팔 기준점)과 가장 가까운 지점을 계산한다.

        fleet_config 노드는 pallet "중심"이 아니라 통로 쪽 정차 위치라, pallet
        중심을 그대로 목표로 쓰면 pallet 크기(약 1.3m)의 절반만큼 팔 리치
        밖으로 벗어난다(2026-07-26 확인). 대신 pallet의 실제 bbox 안에서 AMR과
        가장 가까운 XY 지점(= AMR이 서 있는 쪽 가장자리)을 목표로 삼는다.

        가장자리에 딱 붙이면(margin=0) 그 지점이 팔 자신의 정지 자세
        범위(shoulder_link/upper_arm_link 등)와 거의 겹쳐서, 박스가 스폰되자마자
        팔과 충돌해 튕겨나가는 문제가 있었다(GUI에서 확인된 회귀) - 그래서
        가장자리에서 PALLET_EDGE_MARGIN_M만큼 안쪽으로 들여서 스폰/목표 지점을
        잡는다. 이 마진에는 박스 자신의 반너비도 더한다 - 마진을 중심점
        기준으로만 계산하면 박스 자체가 커서 몸통 절반이 pallet 바닥 밖으로
        걸쳐 나가 AMR 쪽으로 떨어지는 문제가 있었다(GUI에서 확인된 회귀).

        pallet 내부에 벽/기둥이 있어도, 목표 높이를 "바닥"이 아니라 항상
        pallet 전체 bbox의 꼭대기로 잡아두면 그 어떤 XY를 골라도 Z가 모든
        내부 구조물보다 위에 있어서 절대 겹칠 수 없다 - 한때 팔 리치가
        부족해서(벽이 너무 높아져서) 바닥/raycast 기반으로 바꿔봤는데,
        그러자 목표 Z가 벽의 Z 범위 안으로 들어가면서 XY가 벽 위치와
        겹치는 경우 실제로 충돌해 박스가 튕겨나가는 회귀가 생겼다
        (2026-07-27, 사용자 확인 - "기존에는 왜 안 부딪혔지?"에 대한 답:
        원래는 Z가 항상 벽보다 높아서 안 부딪혔던 것). 지금은 벽 높이가
        다시 리치 안쪽으로 낮아졌으니, 원래의 "무조건 꼭대기" 방식으로
        되돌린다 - 더 이상 raycast/바닥 탐색이 필요 없다.
        """
        stage = omni.usd.get_context().get_stage()
        pallet_prim = stage.GetPrimAtPath(pallet_path)
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
        rng = bbox_cache.ComputeWorldBound(pallet_prim).ComputeAlignedRange()
        bmin, bmax = np.array(rng.GetMin()), np.array(rng.GetMax())

        reference_xy, _ = self._read_base_pose()
        margin = PALLET_EDGE_MARGIN_M + self._box_size[:2] / 2.0
        margin = np.minimum(margin, (bmax[:2] - bmin[:2]) / 2.0 - 0.01)
        nearest_xy = np.clip(reference_xy[:2], bmin[:2] + margin, bmax[:2] - margin)
        target_z = bmax[2] + self._box_size[2] / 2.0
        return np.array([nearest_xy[0], nearest_xy[1], target_z])

    def _compute_safe_hover_height(self, pallet_path: str) -> float:
        """pallet 벽 꼭대기보다 확실히 높은 절대 높이를 계산한다.

        base PickPlaceController의 event 0("move above pick")은
        end_effector_initial_height를 목표보다 얼마나 "더 높이"가 아니라
        절대 높이(self._h1)로 쓴다 - 지면 위 데모(z~0.065)에서 튜닝된
        0.55m는 pallet처럼 목표 자체가 1.5m 이상인 경우 오히려 목표보다
        훨씬 낮은 높이라서, "위로 이동"이 실제로는 옆 벽 높이 아래로
        내려갔다가 다시 오르내리며 벽에 스치는 원인이었다(GUI에서 확인된
        회귀). 매 반복마다 실제 목표(pallet)에 맞는 절대 높이를 다시
        계산해서 self._controller.reset(end_effector_initial_height=...)로
        넘겨줘야 한다.
        """
        stage = omni.usd.get_context().get_stage()
        pallet_prim = stage.GetPrimAtPath(pallet_path)
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
        rng = bbox_cache.ComputeWorldBound(pallet_prim).ComputeAlignedRange()
        wall_top_z = rng.GetMax()[2]
        return float(wall_top_z + SAFE_HOVER_MARGIN_M)

    def _compute_cycle_hover_height(self, pallet_path: str, slot_index: int) -> float:
        """이번 사이클의 h1(호버 높이)을 pallet 쪽과 저장소 슬롯 쪽 중 더 높은
        쪽 기준으로 계산한다.

        h1은 reset() 시점에 한 번만 정해져서 사이클 전체(pallet 구간 +
        저장소 구간)에 그대로 쓰이는데, 원래는 pallet 벽 높이만 보고
        계산해서 저장소 목표 높이는 고려하지 않았다. 그 결과(2026-07-27,
        check_hover_heights.py로 실측):
        - 낮은 레이어(0/1)는 h1이 목표보다 27cm 가까이 높아서 불필요하게
          멀리 오르내리고(잔여 수렴 지연을 키우는 원인 중 하나로 보임)
        - 가장 높은 레이어(2)는 h1이 목표보다 오히려 3.2cm *낮아서*,
          event6(h1→목표 보간)이 "하강"이 아니라 실제로는 "상승"이
          돼버리는 역전이 있었다(사용자 확인 - "원래는 위→하강해야
          하는데 이상하게 움직인다"의 원인으로 보임).
        둘 중 더 높은 쪽 + 여유(SAFE_HOVER_MARGIN_M)로 h1을 잡으면, 항상
        양쪽 목표보다 위에서 접근하는 원래 의도를 지키면서 불필요한
        초과분만 줄어든다."""
        pallet_hover = self._compute_safe_hover_height(pallet_path)
        slot_target_z = float(self._compute_slot_target(slot_index)[2])
        return max(pallet_hover, slot_target_z + SAFE_HOVER_MARGIN_M)

    def _compute_slot_target(self, slot_index: int) -> np.ndarray:
        """슬롯의 chassis-local 오프셋을 실제 world 좌표로 바꾼다.

        예전엔 Gf.Matrix3d(Gf.Rotation(quat)) * Gf.Vec3d(offset)로 직접
        회전시켰는데, Gf의 행렬*벡터 곱 관례상 이게 실제로는 반대 방향
        회전(R 대신 사실상 R^T)을 적용해서, 계산된 슬롯 목표가 실제
        StorageBox(섀시 뒤, chassis yaw=45도 기준 world Y~-1.44) 대신 정반대
        방향인 pick pallet/컨베이어 쪽(world Y~-0.6)으로 나왔다(2026-07-26,
        check_slot_targets.py로 실제 StorageBox bbox와 비교해서 확인 - "이상한
        곳(컨베이어 방향)으로 간다"는 사용자 리포트의 원인). ComputeLocalToWorldTransform
        (다른 곳의 _pallet_yaw/_nearest_point_on_pallet_top과 동일한 방식)으로
        섀시의 실제 전체 변환을 직접 써서 이 실수를 피한다.
        """
        stage = omni.usd.get_context().get_stage()
        chassis_prim = stage.GetPrimAtPath(self._chassis_path)
        mat = UsdGeom.Xformable(chassis_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        world_point = mat.Transform(Gf.Vec3d(*STORAGE_SLOT_LOCAL_POSITIONS[slot_index]))
        return np.array(world_point)

    # -----------------------------------------------------------------
    # 내부 helper - 픽업 박스 스폰
    # -----------------------------------------------------------------

    def _clamp_action(self, action, current_joints: np.ndarray):
        """RMPflow가 이번 스텝에 요청한 관절 목표가 현재 위치에서 너무 크게
        튀지 않도록 델타를 MAX_JOINT_STEP_RAD로 제한한다(위 상수 설명 참고 -
        pallet처럼 먼 목표에서 RMPflow가 가끔 불안정해지는 것에 대한 안전판)."""
        if action.joint_positions is not None and action.joint_indices is not None:
            idx = np.array(action.joint_indices)
            current_at_idx = np.array(current_joints)[idx]
            target = np.array(action.joint_positions, dtype=float)
            delta = np.clip(target - current_at_idx, -MAX_JOINT_STEP_RAD, MAX_JOINT_STEP_RAD)
            action.joint_positions = current_at_idx + delta
        return action

    def _spawn_pick_box(self) -> str:
        from isaacsim.core.api.objects import DynamicCuboid

        prim_path = f"/World/_pick_box_{self.robot_id}_{self._next_box_serial}"
        self._next_box_serial += 1
        # _nearest_point_on_pallet_top이 이미 raycast로 실제 pallet 표면을
        # 찾아서 주는 지점이니, 그 자리에 정확히 스폰한다(2026-07-27, 사용자
        # 요청 - "착지하는 위치 확인하고 그 위치에 스폰"). 예전엔 스폰 직후
        # 자유낙하로 자연스럽게 안착시키려고 위쪽에 여유(0.05m)를 두고
        # 스폰했는데, 그러면 착지 직후 얼마간은 반드시 dynamic 상태로
        # 흔들림/구르는 게 생긴다 - 처음부터 정확한 위치에 놓고 kinematic으로
        # 스폰하면 그 흔들림 자체가 생기지 않는다.
        spawn_pos = self._nearest_point_on_pallet_top(PICK_PALLET_PATH)
        # pallet 바닥의 실제 회전(yaw)에 박스도 맞춰 돌려서 스폰한다 - 그래야
        # world 축 기준으로는 넓어 보이는 마름모꼴 바닥 안에 박스가 실제로도
        # 딱 들어간다(위 _pallet_yaw 설명 참고).
        yaw = self._pallet_yaw(PICK_PALLET_PATH)
        orientation = np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])
        DynamicCuboid(
            prim_path=prim_path,
            name=prim_path.replace("/", "_"),
            position=spawn_pos,
            orientation=orientation,
            scale=self._box_size,
            color=np.array([0.0, 0.0, 1.0]),
            mass=1.1,
        )
        # kinematic으로 스폰했다가 흡착 시점에 dynamic으로 되돌리는 방식을
        # 시도해봤는데(2026-07-27), 실행 중(physics_sim_view 활성화 후)
        # kinematicEnabled를 바꾸는 게 tensor 기반 physics view에 제대로
        # 반영이 안 돼서 흡착 자체가 안 되고(closed=False 유지) 섀시가 1m
        # 넘게 표류하는 심각한 회귀가 생겼다 - 되돌렸다. 스폰 위치를
        # raycast로 찾은 정확한 표면으로 맞춘 것(위 여유 0.05m 제거)만
        # 유지한다.
        return prim_path

    # -----------------------------------------------------------------
    # 내부 helper - 운반 중 자세 보정(용접 시점까지 기다리지 않고 미리 각도를 맞춤)
    # -----------------------------------------------------------------

    def _compute_grasp_rel_rot(self) -> Gf.Quatd:
        """흡착된 그 순간, 박스가 그리퍼에 대해 어떤 상대 회전으로 붙었는지
        기록해둔다(trigger_surface_gripper.close()의 local_rot1과 동일한
        관례: rel_rot = box_rot^-1 * gripper_rot, 즉 gripper_rot = box_rot *
        rel_rot). 박스는 그리퍼에 강체로 고정돼 있으니, 이후 그리퍼의 목표
        회전을 바꾸면 이 관계를 그대로 유지한 채 박스도 같이 돈다 - 이 값을
        알아야 "그리퍼가 몇 도를 향해야 박스가 원하는 각도가 되는지"를
        역산할 수 있다."""
        stage = omni.usd.get_context().get_stage()
        gripper_mat = UsdGeom.Xformable(
            stage.GetPrimAtPath(self._gripper_base_path)
        ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        box_mat = UsdGeom.Xformable(
            stage.GetPrimAtPath(self._current_box_prim_path)
        ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        gripper_rot = gripper_mat.ExtractRotationQuat()
        box_rot = box_mat.ExtractRotationQuat()
        return box_rot.GetInverse() * gripper_rot

    def _compute_carry_orientation(self) -> np.ndarray:
        """운반 중(놓기 전) 미리 목표 각도로 그리퍼를 돌려서, 용접되는
        순간까지 기다리지 않고도 박스가 눈으로 보기에 반듯하게 옮겨지도록
        한다(2026-07-27, 사용자 요청 - "운반 중에 각도 보정하면 안 돼?").
        흡착 직후 바로 틀면 그 반동이 흡착 순간의 충격과 겹칠 수 있어서,
        "놓으러 이동하는" phase(event>=5, RMPflow event 5=목표 xy로 이동)
        부터만 적용한다. pick_to_storage(임시 보관함에 내려놓는 구간)만
        보정하고, place_from_storage(랙에 최종 배치)는 각도 요구사항이
        없어서 그대로 둔다."""
        default = np.array([1.0, 0.0, 0.0, 0.0])
        if (
            self._grasp_rel_rot is None
            or self._phase != "pick_to_storage"
            or self._controller.get_current_event() < 5
        ):
            return default
        stage = omni.usd.get_context().get_stage()
        storage_mat = UsdGeom.Xformable(
            stage.GetPrimAtPath(self._storage_box_path)
        ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        target_rot = storage_mat.ExtractRotationQuat()
        gripper_rot_new = target_rot * self._grasp_rel_rot
        return np.array([gripper_rot_new.GetReal(), *gripper_rot_new.GetImaginary()])

    def _carried_box_touches_landing_surface(self) -> bool:
        """지금 옮기고 있는 박스가 StorageBox 선반이나 이미 그 슬롯 칸에
        놓여있는 다른 박스와 실제로(물리적으로) 맞닿았는지 physx overlap
        쿼리로 직접 확인한다. event6(내려놓으러 하강) 도중 이게 True가 되면
        곧바로 놓기(event7)로 넘어간다 - 목표 높이가 정확히 몇 cm인지
        몰라도, 실제로 닿는 순간 알아서 멈추게 하기 위함이다."""
        import carb
        from omni.physx import get_physx_scene_query_interface

        stage = omni.usd.get_context().get_stage()
        box_prim = stage.GetPrimAtPath(self._current_box_prim_path)
        if not box_prim.IsValid():
            return False
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
        rng = bbox_cache.ComputeWorldBound(box_prim).ComputeAlignedRange()
        bmin, bmax = np.array(rng.GetMin()), np.array(rng.GetMax())
        if bmin[0] > bmax[0]:
            return False
        center = (bmin + bmax) / 2.0
        half_ext = (bmax - bmin) / 2.0
        target_prefixes = [self._storage_box_path] + [
            p for p in self._slot_box_paths if p and p != self._current_box_prim_path
        ]
        hits = []

        def report_hit(hit):
            # StorageBox 선반 자체는 RigidBodyAPI 없이 CollisionAPI만 있는 Cube라서
            # PhysX가 그 콜리전을 별도 body가 아니라 부모 articulation link
            # (chassis_link)에 속한 것으로 취급한다 - hit.rigid_body는 항상
            # ".../chassis_link"만 돌려주고 ".../chassis_link/StorageBox"로는 절대
            # 시작하지 않아서, rigid_body로 prefix 매칭하면 이 선반과의 접촉은
            # 영원히 감지되지 않는다(2026-07-27, 헤드리스로 확인 - contact가 항상
            # False). hit.collision은 실제 콜리전 shape prim(StorageBox 자신, 또는
            # 이미 놓인 박스 자신 - DynamicCuboid는 rigid_body==collision)을 그대로
            # 주므로 이걸로 매칭해야 한다.
            col = str(hit.collision)
            if any(col.startswith(p) for p in target_prefixes):
                hits.append(col)
                return False
            return True

        get_physx_scene_query_interface().overlap_box(
            carb.Float3(float(half_ext[0]), float(half_ext[1]), float(half_ext[2])),
            carb.Float3(float(center[0]), float(center[1]), float(center[2])),
            carb.Float4(0.0, 0.0, 0.0, 1.0),
            report_hit,
            False,
        )
        return len(hits) > 0

    # -----------------------------------------------------------------
    # 내부 helper - StorageBox 용접(gripper와 별개, 슬롯<->박스 고정)
    # -----------------------------------------------------------------

    def _weld_joint_path(self, slot_index: int) -> str:
        return f"{self._storage_box_path}/StorageWeldJoint_{slot_index}"

    def _weld_box_to_slot(self, slot_index: int, box_prim_path: str) -> None:
        stage = omni.usd.get_context().get_stage()
        joint_path = self._weld_joint_path(slot_index)
        if stage.GetPrimAtPath(joint_path).IsValid():
            return
        storage_prim = stage.GetPrimAtPath(self._storage_box_path)
        box_prim = stage.GetPrimAtPath(box_prim_path)
        if not storage_prim.IsValid() or not box_prim.IsValid():
            return
        storage_mat = UsdGeom.Xformable(storage_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        box_mat = UsdGeom.Xformable(box_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        rel_local = storage_mat.GetInverse().Transform(box_mat.ExtractTranslation())
        # 박스는 pick pallet의 45도 회전에 맞춰 스폰되고(_spawn_pick_box의
        # _pallet_yaw 참고) 그 각도를 그대로 들고 옮겨오는데, StorageBox
        # 선반은 챗시에 맞춰 축 정렬돼 있어서 실제 회전(box_mat의 회전)을
        # 그대로 용접하면 박스가 선반 위에 삐딱하게(대각선으로) 얹힌
        # 것처럼 보인다(2026-07-27, 사용자 스크린샷 확인). 위치(rel_local)는
        # 실제 도착 지점을 그대로 쓰되, 회전은 무시하고 선반과 축을 맞춘다
        # (identity) - 어차피 흡착 관절(GraspJoint)이 팔-박스 상대 회전을
        # 따로 잡고 있어서 여기서 박스만 시각적으로 반듯하게 재정렬해도
        # 흡착 자체엔 영향 없다.
        rel_rot = Gf.Quatf(1.0, 0.0, 0.0, 0.0)

        joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
        joint.CreateBody0Rel().SetTargets([Sdf.Path(self._storage_box_path)])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(box_prim_path)])
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(rel_local))
        joint.CreateLocalRot0Attr().Set(rel_rot)
        joint.GetPrim().GetAttribute("physics:excludeFromArticulation").Set(True)

    def _unweld_slot(self, slot_index: int) -> None:
        stage = omni.usd.get_context().get_stage()
        joint_path = self._weld_joint_path(slot_index)
        if stage.GetPrimAtPath(joint_path).IsValid():
            stage.RemovePrim(joint_path)

    # -----------------------------------------------------------------
    # 챗시 임시 고정 - pick&place 중 팔 반작용으로 챗시가 조금씩 밀리는
    # drift를 원천 차단한다(2026-07-27, 사용자 확인 - 픽업 사이클마다 챗시
    # yaw가 44.1도->40.6도로 계속 줄어드는 것을 실측으로 확인, "정지 위치에서
    # 조금씩 움직임"). Body0Rel을 비워두면(=world) 그 프레임에서 챗시의
    # "지금 이 순간" 월드 pose를 그대로 LocalPos0/LocalRot0으로 고정하는
    # 뜻이 된다 - 그 시점 위치에 못 박아버리는 것과 같다.
    # -----------------------------------------------------------------

    def _chassis_anchor_joint_path(self) -> str:
        return f"{self._chassis_path}/ChassisAnchorJoint"

    def _anchor_chassis(self) -> None:
        if self._chassis_anchored:
            return
        stage = omni.usd.get_context().get_stage()
        chassis_prim = stage.GetPrimAtPath(self._chassis_path)
        if not chassis_prim.IsValid():
            return
        chassis_mat = UsdGeom.Xformable(chassis_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        world_pos = chassis_mat.ExtractTranslation()
        world_rot = chassis_mat.ExtractRotationQuat()

        joint_path = self._chassis_anchor_joint_path()
        joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
        joint.CreateBody1Rel().SetTargets([Sdf.Path(self._chassis_path)])
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(world_pos))
        joint.CreateLocalRot0Attr().Set(Gf.Quatf(world_rot))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        joint.GetPrim().GetAttribute("physics:excludeFromArticulation").Set(True)
        self._chassis_anchored = True

    def _release_chassis_anchor(self) -> None:
        if not self._chassis_anchored:
            return
        stage = omni.usd.get_context().get_stage()
        joint_path = self._chassis_anchor_joint_path()
        if stage.GetPrimAtPath(joint_path).IsValid():
            stage.RemovePrim(joint_path)
        self._chassis_anchored = False

    # -----------------------------------------------------------------
    # 외부 API
    # -----------------------------------------------------------------

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def occupied_slot_count(self) -> int:
        return len(self._slot_fill_order)

    def start_pick(self, count: int = 1) -> bool:
        """idle일 때만 pick(픽업 -> 빈 슬롯) 시작. count번 내부적으로
        반복한다(슬롯이 모자라면 들어가는 만큼만)."""
        if self._phase != "idle" or count < 1:
            return False
        free_slots = sum(1 for p in self._slot_box_paths if p is None)
        if free_slots < 1:
            print(f"[amr_pick_place] {self.robot_id}: 보관함에 빈 슬롯이 없음 - pick 시작 불가", flush=True)
            return False
        if free_slots < count:
            print(
                f"[amr_pick_place] {self.robot_id}: 빈 슬롯 부족(요청 {count}, 여유 {free_slots}) - "
                f"{free_slots}개만 픽업합니다",
                flush=True,
            )
            count = free_slots
        self._pick_remaining = count
        self._anchor_chassis()
        self._begin_next_pick_repetition()
        return True

    def _begin_next_pick_repetition(self) -> None:
        self._current_box_prim_path = self._spawn_pick_box()
        self._current_slot_index = next(i for i, p in enumerate(self._slot_box_paths) if p is None)
        # NOTE: PickPlaceController.reset()이 내부적으로 RmpFlow.reset()도
        # 호출하는데, 이게 (1) add_cuboid로 등록해둔 장애물(pallet 포함)을
        # 전부 지워버리고(헤드리스로 확인 - __init__에서 한 번만 등록해뒀더니
        # 첫 reset()에서 바로 사라짐), (2) RMPFlowController.reset()이 로봇
        # base pose를 __init__ 시점(=스폰 시점)의 self._default_position/
        # orientation으로 되돌려버린다(2026-07-27, 사용자 확인 - amr_2가
        # 실제로는 PICKUP_A에 도착해 있는데도 팔이 스폰 위치였던
        # PICKUP_WAIT_A 기준 방향으로 뻗던 원인. amr_1은 스폰 위치 자체가
        # PICKUP_A라 우연히 안 드러났었다). 그래서 _refresh_base_pose_and_
        # obstacle()을 reset() "전"이 아니라 반드시 "후"에 호출해야 한다 -
        # 전에 호출하면 reset()이 그 갱신을 곧바로 덮어써버린다.
        # end_effector_initial_height도 매번 pallet 벽 높이와 이번에 갈
        # 저장소 슬롯 높이 중 더 높은 쪽으로 다시 계산해서 넘긴다(위
        # _compute_cycle_hover_height 설명 참고).
        self._controller.reset(
            end_effector_initial_height=self._compute_cycle_hover_height(PICK_PALLET_PATH, self._current_slot_index)
        )
        self._refresh_base_pose_and_obstacle()
        self._register_pallet_obstacles()
        self._register_filled_slot_obstacles()
        self._gripper.set_default_state(opened=True)
        self._gripper.post_reset()
        self._contact_seen = False
        self._grasp_confirmed = False
        self._grasp_rel_rot = None
        self._welded_this_rep = False
        self._cycle_target = self._compute_slot_target(self._current_slot_index)
        self._phase = "pick_to_storage"

    def start_place(self, shelf_num: int) -> bool:
        """비어있지 않은 가장 최근에 채운(LIFO) 슬롯 하나를 꺼내 place(슬롯 ->
        shelf_num에 해당하는 pallet) 시작. shelf_num은 fms_node.py의
        SHELF_INDEX와 동일한 규약(0=280/pallet_01, 1=260/pallet_02,
        2=240/pallet_03). idle이 아니거나 채워진 슬롯이 없거나 shelf_num이
        범위 밖이면 False.

        2026-07-27: FIFO(가장 오래된 것부터)에서 LIFO로 바꿨다 - 박스들은
        서로 위에 얹혀 쌓이는 구조라, 가장 오래된(대개 맨 아래층) 박스부터
        꺼내려 하면 그 위에 쌓인 나중 박스들을 팔이 넘어가거나 스쳐야 해서
        충돌 위험이 크다. 가장 최근에 채운(대개 맨 위/가장 접근하기 쉬운)
        박스부터 꺼내면 다른 박스를 거치지 않고 바로 접근할 수 있다."""
        if self._phase != "idle" or not self._slot_fill_order:
            return False
        if not (0 <= shelf_num < len(PLACE_PALLET_PATHS)):
            print(f"[amr_pick_place] {self.robot_id}: 알 수 없는 shelf_num={shelf_num}", flush=True)
            return False
        slot_index = self._slot_fill_order[-1]
        box_prim_path = self._slot_box_paths[slot_index]
        if box_prim_path is None:
            return False
        self._retry_count = 0
        self._current_shelf_num = shelf_num
        self._anchor_chassis()
        self._begin_place_attempt(slot_index, box_prim_path, shelf_num)
        return True

    def _begin_place_attempt(self, slot_index: int, box_prim_path: str, shelf_num: int) -> None:
        """start_place()의 실제 준비 작업 - 최초 시도와 흡착 실패 재시도
        (_retry_count, update() 참고) 양쪽에서 재사용한다."""
        self._current_slot_index = slot_index
        self._current_box_prim_path = box_prim_path
        # NOTE: PickPlaceController.reset()이 내부적으로 RmpFlow.reset()도
        # 호출하는데, 이게 (1) add_cuboid로 등록해둔 장애물(pallet 포함)을
        # 전부 지워버리고, (2) RMPFlowController.reset()이 로봇 base pose를
        # __init__(스폰) 시점 값으로 되돌려버린다 - 그래서
        # _refresh_base_pose_and_obstacle()은 반드시 reset() "후"에 호출해야
        # 한다(_begin_next_pick_repetition과 동일한 이유, 2026-07-27 발견).
        # end_effector_initial_height도 이번에 놓을 pallet 벽 높이와 지금
        # 꺼내는 저장소 슬롯 높이 중 더 높은 쪽으로 다시 계산해서 넘긴다
        # (위 _compute_cycle_hover_height 설명 참고).
        target_pallet_path = PLACE_PALLET_PATHS[shelf_num]
        self._controller.reset(
            end_effector_initial_height=self._compute_cycle_hover_height(target_pallet_path, slot_index)
        )
        self._refresh_base_pose_and_obstacle()
        self._register_pallet_obstacles()
        self._register_filled_slot_obstacles()
        self._gripper.set_default_state(opened=True)
        self._gripper.post_reset()
        self._contact_seen = False
        self._grasp_confirmed = False
        self._grasp_rel_rot = None
        # 예전엔 트립 시작 시점(여기)에 바로 StorageWeldJoint를 풀었는데,
        # 그러면 접근(event0/1, ~2초) 내내 박스가 아무 지지 없이 자유낙하해서
        # 팔이 도착했을 땐 이미 엉뚱한 곳으로 떨어져 있어 제대로 못 집는
        # 문제가 있었다(2026-07-27, 사용자 확인 - "임시 저장소에서 rack으로
        # 박스 내릴 때 박스를 제대로 못 집어"). 그리퍼가 실제로 흡착에
        # 성공한(_grasp_confirmed) 그 순간까지는 StorageWeldJoint를 그대로
        # 둔다 - 흡착 중엔 GraspJoint와 StorageWeldJoint가 잠깐 같이
        # 박스를 붙들어도 문제없고(둘 다 FixedJoint), 흡착이 확정된 뒤에야
        # unweld해서 "창고에 고정된 채"에서 "그리퍼가 붙든 채"로 끊김 없이
        # 넘어간다(update()의 _grasp_confirmed 처리 참고).
        self._unwelded_this_rep = False
        self._cycle_target = self._nearest_point_on_pallet_top(target_pallet_path)
        self._phase = "place_from_storage"

    def update(self) -> Optional[str]:
        """매 프레임 호출. "pick_done"/"place_done"/"pick_failed" 이벤트가 막
        발생했으면 그 문자열을 반환하고, 아니면 None을 반환한다. count>1인
        pick은 내부적으로 여러 번 반복하고 전부 끝났을 때만 "pick_done"을
        반환한다(중간 반복은 None)."""
        if self._phase == "idle":
            return None

        if self._phase == "cooldown":
            self._cooldown_remaining -= 1
            if self._cooldown_remaining <= 0:
                if self._cooldown_next_phase == "place":
                    self._begin_place_attempt(
                        self._current_slot_index, self._current_box_prim_path, self._current_shelf_num
                    )
                else:
                    self._begin_next_pick_repetition()
            return None

        if self._phase == "returning_home":
            return self._step_return_home()

        current_joints = self._robot.get_joint_positions()
        source_pos = self._read_box_pose(self._current_box_prim_path)

        # _is_contact_now()는 contact_target_filter를 적용해서, 우리가 스폰한
        # 박스가 아니라 pallet 벽면 같은 주변 지오메트리에 스치는 건 접촉으로
        # 안 친다(위 클래스 docstring과 trigger_surface_gripper.py 참고).
        overlap_now = self._controller._is_contact_now()
        if not self._contact_seen and overlap_now:
            self._contact_seen = True
        pick_offset_z = 0.0 if self._contact_seen else PICK_DESCEND_OFFSET_Z
        if self._gripper.is_closed():
            if not self._grasp_confirmed:
                self._grasp_rel_rot = self._compute_grasp_rel_rot()
            self._grasp_confirmed = True
            if self._phase == "place_from_storage" and not self._unwelded_this_rep:
                self._unweld_slot(self._current_slot_index)
                self._unwelded_this_rep = True

        # base PickPlaceController는 event가 0/1일 때만 self._h0(상승 시작
        # 기준 높이)를 picking_position 기준으로 갱신하고, 그 뒤(정지 대기
        # event2/grasp event3)엔 그 값을 그대로 얼려서 쓴다. 그런데 접촉이
        # 감지된 그 프레임 이후로도 박스가 마저 가라앉으며 실제 높이가
        # 계속 바뀌는데, h0는 이미 얼어붙은 값이라 실제 위치와 최대 5cm
        # 넘게 어긋난다(2026-07-27, 사용자 확인 - 상승 시작할 때 그리퍼가
        # 잠깐 박스 안으로 파고들었다 올라오는 것으로 관찰됨. h0가 박스보다
        # 높으면 반대로 뜬 채로 시작해버림 - 실측: 1.4877 vs 1.4359). event4
        # (상승) 시작 전까지 h0을 박스의 실시간 높이로 계속 맞춰서, 상승이
        # 실제 안착 높이에서 정확히 시작하게 한다.
        if self._controller.get_current_event() in (2, 3):
            self._controller._h0 = float(source_pos[2])

        # 놓는 쪽(event7)도 같은 문제가 있다 - event8(놓은 뒤 상승)의 시작
        # 높이는 self._cycle_target[2](명목상의 목표 높이)를 그대로 쓰는데,
        # 실제로 팔이 도달/정착한 높이가 그 명목값과 다르면(선반이나 랙
        # 표면에 걸쳐서 정확히 못 맞추는 경우 등) event8 시작 순간 그리퍼가
        # 명목 높이로 순간 이동했다가 올라가면서 박스를 파고드는 것처럼
        # 보인다(2026-07-27, 사용자 확인 - "place하고... 그리퍼가 상자
        # 안으로 들어갔다가 올라옴"). event7 동안 cycle_target의 높이를
        # 박스의 실시간 높이로 계속 맞춰서, event8이 실제 안착 높이에서
        # 정확히 시작하게 한다.
        if self._controller.get_current_event() == 7:
            self._cycle_target[2] = float(source_pos[2])

        # event6(내려놓으러 하강)는 pick과 달리 실제 접촉을 확인하지 않고
        # 정해진 시간(EVENTS_DT[6])만큼 내려가다 그냥 멈춘다 - 그래서 물리
        # 파라미터(감쇠 등)가 바뀔 때마다 "실제로 도달하는 높이"가 달라져도
        # 알아챌 방법이 없어서, 슬롯마다 목표 높이를 수동으로 다시 재보정해야
        # 했다(2026-07-27, 사용자 지적 - "박스가 임시 저장소랑 충돌했을 때
        # 내려놓는 거 맞아?" - 답은 "아니오"였음). pick의 콘 트리거처럼, 옮기고
        # 있는 박스가 실제로 선반/이미 놓인 박스와 맞닿으면 그 즉시 event를
        # 7(놓기)로 넘겨서, 목표 높이 보정 없이도 항상 정확히 닿는 순간
        # 내려놓게 한다.
        if self._phase == "pick_to_storage" and self._controller.get_current_event() == 6:
            if self._carried_box_touches_landing_surface():
                self._controller._event = 7
                self._controller._t = 0.0

        # event1(하강)/4(들어올리기)/8(상승) 조기 종료 - 실측 그리퍼 높이가
        # 이 phase의 목표 높이에 이미 충분히 가까우면 남은 예산 시간을 그냥
        # 흘려보내지 않고 바로 다음 event로 넘어간다.
        cur_event = self._controller.get_current_event()
        if cur_event in (1, 4, 8):
            target_h = self._controller._h0 if cur_event == 1 else self._controller._h1
            if abs(self._read_gripper_apex_z() - target_h) <= EARLY_EXIT_HEIGHT_TOL_M:
                self._controller._event += 1
                self._controller._t = 0.0

        actions = self._controller.forward(
            picking_position=source_pos,
            placing_position=self._cycle_target,
            current_joint_positions=current_joints,
            end_effector_offset=np.array([0.0, 0.0, pick_offset_z]),
            end_effector_orientation=self._compute_carry_orientation(),
        )
        self._robot.apply_action(self._clamp_action(actions, current_joints))

        # 박스를 슬롯에 용접하는 건 사이클이 다 끝난 뒤(is_done(), 팔이
        # 다시 들어올려 복귀까지 마친 시점)가 아니라, 그리퍼가 실제로 놓는
        # "바로 그 순간"(event 7)에 해야 한다 - 늦게 용접하면 그 사이(event
        # 8~9, 팔이 올라갔다 돌아오는 동안) 박스가 아무 지지 없이 중력으로
        # 바닥까지 떨어져버린 뒤에야 용접하는 꼴이 된다(헤드리스로 확인한
        # 회귀 - 슬롯 목표(z~0.44)가 아니라 항상 바닥(z~0.065)에 놓였었음).
        if (
            self._phase == "pick_to_storage"
            and self._grasp_confirmed
            and not self._welded_this_rep
            and not self._gripper.is_closed()
        ):
            self._weld_box_to_slot(self._current_slot_index, self._current_box_prim_path)
            self._welded_this_rep = True

        if not self._controller.is_done():
            return None

        if self._phase == "pick_to_storage":
            self._cooldown_next_phase = "pick"
            if self._grasp_confirmed:
                self._slot_box_paths[self._current_slot_index] = self._current_box_prim_path
                self._slot_fill_order.append(self._current_slot_index)
                self._retry_count = 0
                self._pick_remaining -= 1
                self._after_return_home = "next_rep" if self._pick_remaining > 0 else (
                    "pick_done" if self._slot_fill_order else "pick_failed"
                )
            else:
                self._retry_count += 1
                if self._retry_count <= MAX_PICK_RETRIES:
                    # "픽업은 무조건 성공해야 한다" - 실패한 슬롯은 건너뛰지
                    # 않고 같은 자리에서 재시도한다(_begin_next_pick_repetition이
                    # 매번 "아직 안 채워진" 가장 앞 슬롯을 다시 고르므로,
                    # pick_remaining을 그대로 두면 자동으로 이 슬롯을 다시
                    # 시도하게 된다).
                    print(
                        f"[amr_pick_place] {self.robot_id}: pick 흡착 실패 - "
                        f"재시도 {self._retry_count}/{MAX_PICK_RETRIES}",
                        flush=True,
                    )
                    self._after_return_home = "next_rep"
                else:
                    print(
                        f"[amr_pick_place] {self.robot_id}: pick 흡착 실패 - "
                        f"재시도 {MAX_PICK_RETRIES}회 모두 실패, 박스 1개 건너뜀",
                        flush=True,
                    )
                    self._retry_count = 0
                    self._pick_remaining -= 1
                    self._after_return_home = "next_rep" if self._pick_remaining > 0 else (
                        "pick_done" if self._slot_fill_order else "pick_failed"
                    )
            # 다음에 뭘 할지(반복/재시도를 더 할지, 이번 update()가 이벤트를
            # 반환할지)와 무관하게, RMPflow에 다음 목표를 맡기기 전엔 항상
            # 먼저 관절 공간 홈 자세로 확실히 복귀시킨다 - 그래야 IK가 매번
            # 같은 안전한 자세에서 시작해서 특이점 근처에서 불안정해지는 걸
            # 막을 수 있다(위 SPAWN_JOINT_ANGLES_DEG 설명 참고).
            self._start_return_home()
            return None

        if self._phase == "place_from_storage":
            self._cooldown_next_phase = "place"
            if self._grasp_confirmed:
                # 실제로 흡착에 성공했을 때만 슬롯을 비운다 - 예전엔 이
                # 흡착 성공 여부를 아예 확인 안 하고 무조건 슬롯을 비우고
                # "place_done"을 돌려줬어서, 흡착이 실패해도(박스는 그대로
                # 창고에 용접된 채 남아있는데) 마치 성공한 것처럼 보고되는
                # 버그가 있었다(2026-07-27, 사용자 확인 - "임시 저장소 1번째
                # 층 박스가 그리퍼랑 닿았는데 흡착이 안 됐는데"도 결국
                # place_done으로 잘못 보고됐던 사례).
                self._slot_box_paths[self._current_slot_index] = None
                self._slot_fill_order.remove(self._current_slot_index)
                self._retry_count = 0
                self._after_return_home = "place_done"
            else:
                self._retry_count += 1
                if self._retry_count <= MAX_PICK_RETRIES:
                    # 흡착 실패 - 박스는 여전히 창고에 용접된 채 그대로다
                    # (_unwelded_this_rep이 False로 남아있음). 같은
                    # slot_index/box_prim_path/shelf_num으로 그대로 재시도한다.
                    print(
                        f"[amr_pick_place] {self.robot_id}: place 흡착 실패 - "
                        f"재시도 {self._retry_count}/{MAX_PICK_RETRIES}",
                        flush=True,
                    )
                    self._after_return_home = "next_rep"
                else:
                    print(
                        f"[amr_pick_place] {self.robot_id}: place 흡착 실패 - "
                        f"재시도 {MAX_PICK_RETRIES}회 모두 실패, 이번엔 포기",
                        flush=True,
                    )
                    self._retry_count = 0
                    self._after_return_home = "place_failed"
            self._start_return_home()
            return None

        return None

    def _start_return_home(self) -> None:
        self._phase = "returning_home"
        self._return_home_steps_left = RETURN_HOME_STEPS
        self._return_home_start_positions = np.array(
            self._robot.get_joint_positions()
        )[self._arm_joint_indices]

    def _step_return_home(self) -> Optional[str]:
        # RETURN_HOME_STEPS는 "몇 스텝이나 지켜볼지"일 뿐, 그 자체로는 움직임을
        # 부드럽게 만들어주지 않는다 - 매 프레임 최종 목표(홈 관절각)를 그대로
        # 명령하면 첫 프레임부터 큰 위치 오차가 실려서 관절 구동이 거칠게
        # 반응한다(RETURN_HOME_STEPS를 200으로 늘려도 똑같이 안 부드러웠던
        # 이유 - 사용자 확인, 2026-07-27). 시작 시점 관절각에서 홈 관절각까지
        # sin 이징으로 서서히 섞어서, 매 프레임 다른(점점 목표에 가까워지는)
        # 중간 목표를 명령한다.
        total_steps = max(RETURN_HOME_STEPS, 1)
        progress = 1.0 - (self._return_home_steps_left - 1) / total_steps
        alpha = 0.5 * (1.0 - math.cos(min(max(progress, 0.0), 1.0) * math.pi))
        blended = (
            (1.0 - alpha) * self._return_home_start_positions
            + alpha * self._arm_home_positions
        )
        self._robot.apply_action(
            ArticulationAction(joint_positions=blended, joint_indices=self._arm_joint_indices)
        )
        self._return_home_steps_left -= 1
        current = np.array(self._robot.get_joint_positions())[self._arm_joint_indices]
        settled = float(np.max(np.abs(current - self._arm_home_positions))) < RETURN_HOME_TOLERANCE_RAD
        if not settled and self._return_home_steps_left > 0:
            return None

        after = self._after_return_home
        self._after_return_home = None
        if after == "next_rep":
            # 다음 반복을 바로 시작하지 않고 잠깐 더 대기한다 - 방금 홈
            # 복귀하며 생긴 충격이 가라앉기 전에 섀시 pose를 읽으면 목표가
            # 잘못 계산될 수 있다(위 COOLDOWN_STEPS 설명 참고).
            self._phase = "cooldown"
            self._cooldown_remaining = COOLDOWN_STEPS
            return None
        self._phase = "idle"
        self._release_chassis_anchor()
        return after
