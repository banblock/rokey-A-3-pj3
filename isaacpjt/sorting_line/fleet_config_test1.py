#!/usr/bin/env python3
"""노드 그래프 / 로봇-종류 배정 — fms_node.py, fleet_driver.py, Isaac Sim
스크립트(1_conveyor_sorter_env.py, 3_amr_fleet.py)가 전부 공유하는 순수 데이터 모듈.

fleet_config.py에서 신발 종류를 A 하나만 남긴 최소 테스트용 버전 — 라인 1개
(랙 1개, 로봇 2대)만 있는 상황을 빠르게 재현/디버깅하기 위함. B/C/D의
PICKUP/HUB/Rack 블록을 통째로 지웠고, A가 SHOE_TYPES 맨 앞이었기 때문에
로봇 번호(amr_1, amr_2)는 원본과 동일하게 유지된다. ROBOTS_PER_TYPE은
원본과 동일하게 2로 뒀다 — 종류를 1개로 줄인 목적이 "여러 종류가 뒤섞이는
경우"를 배제하고 "같은 종류 로봇 2대끼리의 상호작용"(근접/detour 경쟁,
배치 디스패치 등)만 집중적으로 테스트하기 위함이라, 로봇 수까지 줄이면 그
테스트 대상 자체가 없어진다. 로봇 1대만 필요하면 ROBOTS_PER_TYPE만 1로
바꾸면 된다(그 값 하나로 전체가 자동으로 맞춰짐).

rclpy, m0609_interfaces 등 ROS2 관련 패키지를 절대 import하지 않는다 — Isaac Sim은
자체 번들 파이썬(3.11)을 쓰는데, 시스템 rclpy는 다른 파이썬 ABI(3.10)로 빌드돼
있고 커스텀 srv 패키지(m0609_interfaces)는 애초에 그 환경에 없다. 이 상수들을
얻으려고 fms_node.py(=rclpy·m0609_interfaces 의존)를 통째로 import하면 Isaac Sim
쪽에서 ModuleNotFoundError로 죽으므로, 좌표계를 공유하는 순수 데이터만 떼어냈다.
"""

import heapq
import math

# ╔══════════════════════════════════════════════════════════════╗
# ║  A. 로봇 배정 (신발 종류 1개, 로봇 2대 전담)                          ║
# ╚══════════════════════════════════════════════════════════════╝
SHOE_TYPES = ["A"]
ROBOTS_PER_TYPE = 2

ROBOT_SHOE_TYPE = {
    f"amr_{i + 1}": shoe_type
    for i, shoe_type in enumerate(
        shoe_type for shoe_type in SHOE_TYPES for _ in range(ROBOTS_PER_TYPE)
    )
}
# {"amr_1": "A", "amr_2": "A"}

# 신발 종류별 전용 픽업 지점 — 로봇은 자기 담당 종류의 픽업 지점에서만 신발을 받는다.
PICKUP_NODE = {shoe_type: f"PICKUP_{shoe_type}" for shoe_type in SHOE_TYPES}
# 배치를 다 끝낸 로봇이 PICKUP이 이미 점유돼 있을 때 대신 대기하는 지점(종류별 전용) —
# 대기열의 맨 앞 슬롯. RackX_OUT→...→PICKUP_X 복귀 경로 구조상 ROBOTS_PER_TYPE이
# 몇이든(1이어도) 항상 존재해야 하는 노드라 별도로 무조건 만든다.
PICKUP_WAIT_NODE = {shoe_type: f"PICKUP_WAIT_{shoe_type}" for shoe_type in SHOE_TYPES}
# ROBOTS_PER_TYPE이 3 이상일 때만 필요한 "추가" 대기 슬롯(WAIT2, WAIT3, ...) —
# PICKUP_WAIT_X 뒤로 한 줄로 이어붙는다. 로봇 수가 2 이하면 빈 리스트.
PICKUP_EXTRA_WAIT_NODES = {
    shoe_type: [f"PICKUP_WAIT{k}_{shoe_type}" for k in range(2, ROBOTS_PER_TYPE)]
    for shoe_type in SHOE_TYPES
}
# 타입별 대기 슬롯 전체(앞→뒤 순서) — 맨 앞이 PICKUP_WAIT_X. fms_node.py의
# _pick_pickup_target()이 복귀할 로봇을 앞에서부터 순서대로 빈 슬롯에 배정할 때 쓴다.
PICKUP_WAIT_SLOTS = {
    shoe_type: [PICKUP_WAIT_NODE[shoe_type]] + PICKUP_EXTRA_WAIT_NODES[shoe_type]
    for shoe_type in SHOE_TYPES
}

# 로봇은 기본적으로 자기 종류의 PICKUP_X에 가 있다가, 이미 다른 로봇이 거기
# 있으면 대기 슬롯(PICKUP_WAIT_X, WAIT2_X, ...)에서 대기한다 — 그래서 별도의
# "홈 슬롯(WAIT_N)" 개념 자체가 필요 없다. 로봇 0번은 PICKUP_X, 그 뒤 로봇들은
# 대기 슬롯을 앞에서부터 하나씩(1번→WAIT_X, 2번→WAIT2_X, ...) 스폰 위치 겸
# 초기 홈으로 재사용한다 — 초기 등록 시점에만 쓰이고, 이후 복귀 목적지는 항상
# fms_node.py의 _pick_pickup_target()이 그때그때 점유 상태를 보고 동적으로 정한다.
# ROBOTS_PER_TYPE을 바꾸면 슬롯 개수(HOME_SLOTS)도 자동으로 맞춰 늘어나므로
# ROBOTS_PER_TYPE 값 하나만 바꿔도 로봇 수 스케일링이 그대로 동작한다.
HOME_SLOTS = {
    shoe_type: [PICKUP_NODE[shoe_type]] + PICKUP_WAIT_SLOTS[shoe_type][: ROBOTS_PER_TYPE - 1]
    for shoe_type in SHOE_TYPES
}
ROBOT_HOME_NODE = {}
for _i, _robot_id in enumerate(ROBOT_SHOE_TYPE):
    _home_shoe_type = ROBOT_SHOE_TYPE[_robot_id]
    _within_type_idx = _i % ROBOTS_PER_TYPE
    ROBOT_HOME_NODE[_robot_id] = HOME_SLOTS[_home_shoe_type][_within_type_idx]


# ╔══════════════════════════════════════════════════════════════╗
# ║  B. 노드 그래프 정의 (완벽한 일방통행 및 순환 구조 적용)               ║
# ╚══════════════════════════════════════════════════════════════╝

NODE_GRAPH = {
    # 1. 픽업 지점 — 실제 컨베이어는 아직 단일 벨트/단일 픽업 지점이라
    # 아래 좌표는 임시값이다 — 종류별 벨트/분기 배치가 실제로 정해지면
    # 실측치로 교체해야 한다.
    "PICKUP_A":     {"position": (5.02, -0.98, 1.03), "neighbors": ["HUB_A_APPROACH"]},

    "HUB_A_APPROACH" :  {"position": (7.02, 1.02, 1.03), "neighbors": ["HUB_A"]},

    # 2. 종류별 전용 경유지. HUB의 Y가 랙 안쪽(280/260/240)보다 더 크다(더
    # "깊다") — 즉 로봇은 픽업에서 나와 랙 구역 중 가장 깊은 지점(HUB)까지
    # 갔다가 얕은 쪽(240)으로 내려오면서 신발을 놓고, 가장 얕은 지점(OUT)에서
    # 바로 복귀한다. HUB→랙 진입 구간은 이미 세분화(_subdivide_edge)돼 있어
    # 길어져도 안전하지만, 복귀 구간은 세분화가 안 돼 있어서 짧게 유지하는
    # 쪽이 유리하다.
    "HUB_A":        {"position": (7.02, -4.3, 1.03), "neighbors": ["RackA_280", "HUB_A_detour"]},

    # 2.5. 우회(detour) 통로 전용 허브 — 근접 통로의 HUB_A와 대칭되는 지점.
    # HUB_A에서 근접(RackA_280)과 우회(HUB_A_detour) 두 갈래로 갈라진 뒤, 우회
    # 쪽은 이 노드를 거쳐 detour 레인 Y(근접보다 2.4m 더 안쪽)로 이동한 다음
    # RackA_280_detour로 들어간다 — HUB_A_APPROACH가 근접 진입 전 굴절점 역할을
    # 하는 것과 같은 구조를 우회 쪽에도 그대로 만든 것.
    "HUB_A_detour": {"position": (7.02, -6.7, 1.03), "neighbors": ["RackA_280_detour"]},

    # 3. 랙 A. 근접/detour는 두 개의 완전히 분리된 통로. 신발은 항상
    # 280→260→240 순서로 내려놓는다(사용자 지정). 두 통로가 같은 노드를
    # 반대 방향으로 공유하면 서로 마주보고 맞물려 데드락이 나므로(한 로봇은
    # 260을 들고 240을 기다리고, 다른 로봇은 240을 들고 260을 기다리는 식),
    # 아예 노드 자체를 안 겹치게 detour용 사본을 따로 둔다. 240/260/280(근접)은
    # 원래 X,Y가 완전히 같고 Z(선반 높이)만 달랐는데, 지상 로봇은 Z를
    # 무시하므로 실제로는 같은 지점이었다 — 사이즈별로 Y(랙 안쪽 깊이)를
    # 갈라 물리적으로도 분리했고, detour 통로는 아예 다른 X대(근접 통로 옆,
    # 중심에서 바깥 방향으로 0.6m)를 써서 두 직선이 좌우로 나란히 떨어지게 했다.
    "RackA_280":       {"position": (2.5, -4.3, 1.03), "neighbors": ["RackA_260"]},
    "RackA_260":       {"position": (-1.0, -4.3, 1.03), "neighbors": ["RackA_240"]},
    "RackA_240":       {"position": (-4.5, -4.3, 1.03), "neighbors": ["RackA_OUT"]},
    "RackA_280_detour":   {"position": (2.5, -6.7, 1.03), "neighbors": ["RackA_260_detour"]},
    "RackA_260_detour":   {"position": (-1.0, -6.7, 1.03), "neighbors": ["RackA_240_detour"]},
    "RackA_240_detour":   {"position": (-4.5, -6.7, 1.03), "neighbors": ["RackA_OUT"]},
    "RackA_OUT":     {"position": (-7.25, -4.3, 0.0), "neighbors": ["PICKUP_A_APPROACH"]},
    "PICKUP_A_APPROACH":    {"position": (-4.3, -2.33, 1.03), "neighbors": ["PICKUP_WAIT_A"]},
    "PICKUP_WAIT_A":    {"position": (3.1, -2.85, 1.03), "neighbors": ["PICKUP_A"]},
    
}

# 배치(5켤레) 작업을 전부 마친 로봇, 혹은 그냥 다른 로봇에게 PICKUP_X를 양보해야
# 하는 로봇이 대신 대기하는 지점(종류별 전용). PICKUP_X 바로 뒤(HUB 반대 방향)에
# 한 줄로 늘어놓는다 — 초기 스폰 위치로도 재사용된다(ROBOT_HOME_NODE 참고).
# WAIT_X(맨 앞)는 항상 만들고, WAIT2_X 이후는 ROBOTS_PER_TYPE이 3 이상일 때만
# PICKUP_EXTRA_WAIT_NODES에 담겨 있으므로 자동으로 그만큼만 이어붙는다.
    # _WAIT_SLOT_SPACING = 0.5  # 슬롯 간 Y 간격(스케일 전) — 기존 WAIT_X 오프셋 그대로 유지
    # for _shoe_type in SHOE_TYPES:
    #     _pickup_pos = NODE_GRAPH[PICKUP_NODE[_shoe_type]]["position"]
    #     NODE_GRAPH[PICKUP_WAIT_NODE[_shoe_type]] = {
    #         "position": (_pickup_pos[0], -_WAIT_SLOT_SPACING, _pickup_pos[2]),
    #         "neighbors": [PICKUP_NODE[_shoe_type]],
    #     }
    #     _prev_slot = PICKUP_WAIT_NODE[_shoe_type]
    #     for _extra_idx, _extra_slot in enumerate(PICKUP_EXTRA_WAIT_NODES[_shoe_type]):
    #         NODE_GRAPH[_extra_slot] = {
    #             "position": (_pickup_pos[0], -_WAIT_SLOT_SPACING * (_extra_idx + 2), _pickup_pos[2]),
    #             "neighbors": [_prev_slot],
    #         }
    #         _prev_slot = _extra_slot

# PICKUP_X → HUB_X 굴절점(종류별 하나) — HUB_X_APPROACH를 "HUB의 X, 픽업의 Y"로
# 두면 HUB_APPROACH→HUB 구간(수직)이 HUB와 같은 X를 쓰는 근접 레인 랙 컬럼
# (RackX_280/260/240)을 그대로 관통해버린다 — 대각선만 없앴지 "세로로 랙을
# 뚫고 지나가는" 문제가 남아있었다. 대신 "픽업의 X, HUB의 Y"로 둬서: 픽업
# 자신의 전용 컬럼을 타고 랙 꼭대기보다 높은 HUB의 Y까지 먼저 수직으로 올라간
# 다음, 랙 위쪽(랙 영역보다 높은 층)에서만 수평으로 HUB까지 이동한다 —
# 이러면 랙 영역을 수직/수평 어느 구간에서도 절대 지나가지 않는다.
# for _shoe_type in SHOE_TYPES:
#     _pickup_x = NODE_GRAPH[PICKUP_NODE[_shoe_type]]["position"][0]
#     _hub_y = NODE_GRAPH[f"HUB_{_shoe_type}"]["position"][1]
#     NODE_GRAPH[f"HUB_{_shoe_type}_APPROACH"] = {
#         "position": (_pickup_x, _hub_y, 0.0),
#         "neighbors": [f"HUB_{_shoe_type}"],
#     }
#     NODE_GRAPH[PICKUP_NODE[_shoe_type]]["neighbors"] = [f"HUB_{_shoe_type}_APPROACH"]

# RackX_OUT → PICKUP_X 굴절점(종류별 하나) — 복귀할 때도 OUT에서 픽업까지
# 대각선으로 바로 가면 랙 구역을 가로지른다. OUT의 X를 그대로 따라 수직으로
# 내려온 다음, 픽업 열의 깊이(Y=0)에서 수평으로 이동하게 중간점을 끼워 넣는다
# — 나가는 길(HUB_X_APPROACH)과 정확히 대칭되는 구조. 종류가 하나뿐이라 다른
# 타입과의 동선 교차를 피하려고 Y를 낮추는 오프셋(원본의 B/C 전용 처리)은
# 필요 없어서 0으로 둔다.
# _APPROACH_Y_OFFSET = {"A": 0.0}
# for _shoe_type in SHOE_TYPES:
#     _out_x = NODE_GRAPH[f"Rack{_shoe_type}_OUT"]["position"][0]
#     _approach_y = NODE_GRAPH[PICKUP_NODE[_shoe_type]]["position"][1] + _APPROACH_Y_OFFSET[_shoe_type]
#     NODE_GRAPH[f"PICKUP_{_shoe_type}_APPROACH"] = {
#         "position": (_out_x, _approach_y, 0.0),
#         # PICKUP_X가 비어있어도 APPROACH에서 곧장 들어가지 않고 항상
#         # PICKUP_WAIT_X를 먼저 거치게 한다 — 복귀 경로를 단일 통로로 고정해서
#         # (PICKUP_X 직행/PICKUP_WAIT_X 대기 두 갈래로 갈리지 않게) 진입 방향을
#         # 하나로 통일한다. PICKUP_WAIT_X → PICKUP_X 엣지는 원래 있던 걸 그대로 탄다.
#         "neighbors": [PICKUP_WAIT_NODE[_shoe_type]],
#     }
#     NODE_GRAPH[f"Rack{_shoe_type}_OUT"]["neighbors"] = [f"PICKUP_{_shoe_type}_APPROACH"]

# 전체 포인트가 서로 너무 가깝다는 피드백으로 X,Y만 일괄로 넓힌다(Z는 선반
# 높이라 스케일하면 랙이 비정상적으로 커지므로 그대로 둔다). 아래쪽 본선
# 세분화(_subdivide_edge)는 이 스케일이 다 적용된 뒤의 실제 거리를 기준으로
# 칸을 나누므로, 칸 크기(1.2m 목표)는 이 배율과 무관하게 항상 정확하게 유지된다.
_GRAPH_SCALE = 1
for _node_data in NODE_GRAPH.values():
    _sx, _sy, _sz = _node_data["position"]
    _node_data["position"] = (_sx * _GRAPH_SCALE, _sy * _GRAPH_SCALE, _sz)


def _distance(a, b):
    ax, ay, az = NODE_GRAPH[a]["position"]
    bx, by, bz = NODE_GRAPH[b]["position"]
    return ((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2) ** 0.5


def _subdivide_edge(from_node, to_node, max_segment_m):
    """from_node → to_node 사이 긴 간선 하나를 로봇 한 대가 차지할 만한(약
    max_segment_m 이하) 동일 간격 칸들로 잘게 쪼갠다.

    간선 하나를 통째로 잠그면 로봇 한 대가 그 구간을 다 지나갈 때까지 뒤차가
    출발을 못 한다 — 중간 노드를 넣어 칸마다 따로 락을 걸고 풀게 하면, 로봇들이
    기차처럼 꼬리를 물고 이동할 수 있다(플래투닝). 실제 길이를 max_segment_m로
    나눈 몫을 올림해서 칸 수를 정하므로, 어떤 칸도 max_segment_m를 넘지 않는다.
    이미 그보다 짧은 간선은 그대로 둔다(쪼갤 필요 없음).
    """
    import math

    if to_node not in NODE_GRAPH[from_node]["neighbors"]:
        raise ValueError(f"{from_node} -> {to_node} 간선이 없어서 쪼갤 수 없음")

    ax, ay, az = NODE_GRAPH[from_node]["position"]
    bx, by, bz = NODE_GRAPH[to_node]["position"]
    total = ((bx - ax) ** 2 + (by - ay) ** 2 + (bz - az) ** 2) ** 0.5

    num_segments = max(1, math.ceil(total / max_segment_m))
    if num_segments <= 1:
        return  # 이미 충분히 짧음

    chain = [from_node]
    for i in range(1, num_segments):
        t = i / num_segments
        seg_id = f"{from_node}__{to_node}_{i}"
        NODE_GRAPH[seg_id] = {
            "position": (ax + (bx - ax) * t, ay + (by - ay) * t, az + (bz - az) * t),
            "neighbors": [],
        }
        chain.append(seg_id)
    chain.append(to_node)

    # from_node의 neighbors에서 원래 to_node로 가던 간선을 체인의 첫 칸으로 교체
    NODE_GRAPH[from_node]["neighbors"] = [
        n for n in NODE_GRAPH[from_node]["neighbors"] if n != to_node
    ] + [chain[1]]
    for i in range(1, len(chain) - 1):
        NODE_GRAPH[chain[i]]["neighbors"] = [chain[i + 1]]


def shortest_path(start, goal):
    """일방통행(Directed Graph)에서도 정상 작동하는 다익스트라 최단 경로"""
    dist = {start: 0.0}
    prev = {}
    visited = set()
    heap = [(0.0, start)]
    while heap:
        d, node = heapq.heappop(heap)
        if node in visited:
            continue
        visited.add(node)
        if node == goal:
            break
        for nb in NODE_GRAPH[node]["neighbors"]:  # 오직 '나가는 방향(neighbors)'만 탐색
            nd = d + _distance(node, nb)
            if nd < dist.get(nb, float("inf")):
                dist[nb] = nd
                prev[nb] = node
                heapq.heappush(heap, (nd, nb))

    if goal not in dist:
        return None

    path = [goal]
    while path[-1] != start:
        path.append(prev[path[-1]])
    path.reverse()
    return path


TARGET_SLOT_NODE = {
    (t, s): f"Rack{t}_{s}"
    for t in SHOE_TYPES
    for s in ["240", "260", "280"]
}


# ╔══════════════════════════════════════════════════════════════╗
# ║  C. 본선 구간 세분화 (플래투닝)                                    ║
# ╚══════════════════════════════════════════════════════════════╝
# 종류가 하나뿐이어도, PICKUP_A → HUB_A_APPROACH → HUB_A 구간은 그 종류를
# 담당하는 로봇 2대가 같이 쓰는 "본선"이다(그 뒤 HUB_A → RackA_280/280_detour는
# 둘 중 하나로만 갈라지는 지선이라 원래도 안 겹침). 이 본선을 통짜 간선으로
# 두면 로봇 1대가 다 지나갈 때까지 파트너가 못 나가므로, 로봇 한 대 칸(약
# 1.2m) 단위로 잘게 쪼개서 두 로봇이 기차처럼 꼬리를 물고 지나갈 수 있게
# 한다. 굴절점(HUB_A_APPROACH) 전후 두 구간 다 세분화한다.
_MAIN_LINE_SEGMENT_M = 1.2
for _shoe_type in SHOE_TYPES:
    _subdivide_edge(PICKUP_NODE[_shoe_type], f"HUB_{_shoe_type}_APPROACH", _MAIN_LINE_SEGMENT_M)
    _subdivide_edge(f"HUB_{_shoe_type}_APPROACH", f"HUB_{_shoe_type}", _MAIN_LINE_SEGMENT_M)
    # HUB_X → 랙 진입 두 갈래(근접 RackX_280 / 우회 HUB_X_detour → RackX_280_detour)도
    # 같은 이유로 세분화한다 — 이 두 구간은 각각 근접/우회 한 갈래로만 가는
    # 단일 경로라 세분화해도 280→260→240처럼 "체크포인트 순서를 지켜야 하는
    # 사슬"과 안 섞인다. 근접/우회가 갈라진 *뒤*의 랙 내부 사슬(RackX_280→260→240,
    # RackX_280_detour→260_detour→240_detour)은 각 정지 지점이 실제 배치
    # 위치라 일부러 세분화하지 않는다(슬롯별 점유 판정이 필요해서).
    _subdivide_edge(f"HUB_{_shoe_type}", f"Rack{_shoe_type}_280", _MAIN_LINE_SEGMENT_M)
    _subdivide_edge(f"HUB_{_shoe_type}", f"HUB_{_shoe_type}_detour", _MAIN_LINE_SEGMENT_M)
    _subdivide_edge(f"HUB_{_shoe_type}_detour", f"Rack{_shoe_type}_280_detour", _MAIN_LINE_SEGMENT_M)
    # 복귀 본선(OUT → PICKUP_X_APPROACH → PICKUP_WAIT_X → PICKUP_X)도 같은
    # 이유로 세분화한다 — PICKUP_X_APPROACH는 이제 PICKUP_WAIT_X 하나로만
    # 이어지므로(PICKUP_X 직행 갈래 없음) 갈림 없는 단일 경로만 쪼개면 된다.
    _subdivide_edge(f"Rack{_shoe_type}_OUT", f"PICKUP_{_shoe_type}_APPROACH", _MAIN_LINE_SEGMENT_M)
    _subdivide_edge(f"PICKUP_{_shoe_type}_APPROACH", PICKUP_WAIT_NODE[_shoe_type], _MAIN_LINE_SEGMENT_M)


def robot_spawn_yaw(robot_id):
    """로봇의 스폰 자세(yaw, 라디안)를 그 로봇 홈 노드에서의 첫 이동 방향과
    일치시키기 위한 값을 계산한다 — 안 맞추면 Isaac Sim 기본 스폰 자세(보통
    월드 X축 방향)와 첫 이동 방향이 달라서, 시작하자마자 제자리에서 크게
    회전부터 하고 출발하는 것처럼 보인다(실제로 스폰 스크립트가 자세를
    아예 안 지정해서 생긴 문제였다).

    3_amr_fleet.py(스폰 시 이 값으로 실제 자세를 맞춤)와 fleet_driver.py
    (오도메트리를 월드 좌표로 바꿀 때 이 값만큼 회전 보정)가 반드시 똑같은
    값을 써야 하므로, 각자 계산하지 않고 이 함수 하나로 공유한다.
    """
    home_node = ROBOT_HOME_NODE[robot_id]
    neighbors = NODE_GRAPH[home_node]["neighbors"]
    if not neighbors:
        return 0.0
    hx, hy, _hz = NODE_GRAPH[home_node]["position"]
    nx, ny, _nz = NODE_GRAPH[neighbors[0]]["position"]
    return math.atan2(ny - hy, nx - hx)
