#!/usr/bin/env python3
"""노드 그래프 / 로봇-종류 배정 — fms_node.py, fleet_driver.py, Isaac Sim
스크립트(1_conveyor_sorter_env.py, 3_amr_fleet.py)가 전부 공유하는 순수 데이터 모듈.

fleet_config.py의 D(신발 종류) 관련 내용을 전부 뺀 버전 — A/B/C 3종류, 로봇
6대(종류당 2대) 구성. D의 PICKUP/HUB/Rack 블록을 통째로 지웠고, D가 SHOE_TYPES
맨 끝이었기 때문에 남은 A/B/C의 로봇 번호(amr_1~6)는 원본과 동일하게 유지된다.

rclpy, m0609_interfaces 등 ROS2 관련 패키지를 절대 import하지 않는다 — Isaac Sim은
자체 번들 파이썬(3.11)을 쓰는데, 시스템 rclpy는 다른 파이썬 ABI(3.10)로 빌드돼
있고 커스텀 srv 패키지(m0609_interfaces)는 애초에 그 환경에 없다. 이 상수들을
얻으려고 fms_node.py(=rclpy·m0609_interfaces 의존)를 통째로 import하면 Isaac Sim
쪽에서 ModuleNotFoundError로 죽으므로, 좌표계를 공유하는 순수 데이터만 떼어냈다.
"""

import heapq

# ╔══════════════════════════════════════════════════════════════╗
# ║  A. 로봇 배정 (신발 종류별 2대 전담)                                ║
# ╚══════════════════════════════════════════════════════════════╝
SHOE_TYPES = ["A", "B", "C"]
ROBOTS_PER_TYPE = 1

ROBOT_SHOE_TYPE = {
    f"amr_{i + 1}": shoe_type
    for i, shoe_type in enumerate(
        shoe_type for shoe_type in SHOE_TYPES for _ in range(ROBOTS_PER_TYPE)
    )
}
# {"amr_1": "A", "amr_2": "A", "amr_3": "B", "amr_4": "B",
#  "amr_5": "C", "amr_6": "C"}

# 신발 종류별 전용 픽업 지점 — 로봇은 자기 담당 종류의 픽업 지점에서만 신발을 받는다.
PICKUP_NODE = {shoe_type: f"PICKUP_{shoe_type}" for shoe_type in SHOE_TYPES}
# 배치를 다 끝낸 로봇이 PICKUP이 이미 점유돼 있을 때 대신 대기하는 지점(종류별 전용).
PICKUP_WAIT_NODE = {shoe_type: f"PICKUP_WAIT_{shoe_type}" for shoe_type in SHOE_TYPES}

# 로봇은 기본적으로 자기 종류의 PICKUP_X에 가 있다가, 이미 파트너 로봇이 거기
# 있으면 PICKUP_WAIT_X에서 대기한다 — 그래서 별도의 "홈 슬롯(WAIT_N)" 개념 자체가
# 필요 없다. 예전엔 WAIT_N이라는 제3의 지점을 따로 두고 관리했는데, 결국 로봇이
# 쉴 때 가는 곳은 PICKUP_X 아니면 PICKUP_WAIT_X뿐이라 중복이었다. 로봇 0번은
# PICKUP_X, 1번(파트너)은 PICKUP_WAIT_X를 스폰 위치 겸 초기 홈으로 그대로
# 재사용한다 — 초기 등록 시점에만 쓰이고, 이후 복귀 목적지는 항상
# fms_node.py의 _pick_pickup_target()이 그때그때 점유 상태를 보고 동적으로 정한다.
ROBOT_HOME_NODE = {}
for _i, _robot_id in enumerate(ROBOT_SHOE_TYPE):
    _home_shoe_type = ROBOT_SHOE_TYPE[_robot_id]
    _within_type_idx = _i % ROBOTS_PER_TYPE
    ROBOT_HOME_NODE[_robot_id] = (
        PICKUP_NODE[_home_shoe_type] if _within_type_idx == 0 else PICKUP_WAIT_NODE[_home_shoe_type]
    )


# ╔══════════════════════════════════════════════════════════════╗
# ║  B. 노드 그래프 정의 (완벽한 일방통행 및 순환 구조 적용)               ║
# ╚══════════════════════════════════════════════════════════════╝

NODE_GRAPH = {
    # 1. 픽업 지점 (신발 종류별로 분리 — 각 로봇은 자기 담당 종류의 픽업
    # 지점으로만 들어간다). 실제 컨베이어는 아직 단일 벨트/단일 픽업 지점이라
    # 아래 좌표는 임시값이다 — 종류별 벨트/분기 배치가 실제로 정해지면
    # 실측치로 교체해야 한다. 간격을 2.0m로 둔 건 Nova Carter 스폰 충돌
    # 전례(0.5m 확실히 충돌, 1.0m도 부족, 2.0m에서 해결됨) 때문.
    "PICKUP_A":     {"position": (1.7, 0.0, 0.794), "neighbors": ["HUB_A"]},
    "PICKUP_B":     {"position": (3.7, 0.0, 0.794), "neighbors": ["HUB_B"]},
    "PICKUP_C":     {"position": (5.7, 0.0, 0.794), "neighbors": ["HUB_C"]},

    # 2. 종류별 전용 경유지 — 다른 종류 로봇과는 어떤 노드도 공유하지 않는다
    # (자기 종류 2대끼리만 노드 락을 놓고 경쟁). 예전에 로봇 전체가 HUB_1~4
    # 순환 루프 하나를 공유했을 때는, A종류 로봇이 경로 중간에 멈추면 그
    # 루프를 지나야 하는 다른 종류 로봇까지 덩달아 막히는 문제가 있었다
    # (본선-지선 미분리 문제) — 지금은 종류별로 픽업→경유지→랙→복귀까지
    # 완전히 독립된 통로를 쓴다.
    #
    # HUB의 Y가 랙 안쪽(280/260/240)보다 더 크다(더 "깊다") — 즉 로봇은 픽업에서
    # 나와 랙 구역 중 가장 깊은 지점(HUB)까지 갔다가 얕은 쪽(240)으로 내려오면서
    # 신발을 놓고, 가장 얕은 지점(OUT)에서 바로 복귀한다. 예전에는 반대였는데
    # (HUB가 얕고 OUT이 깊음), 그러면 복귀 구간(OUT→픽업)이 랙 전체 깊이만큼
    # 길어져서 왕복 경로끼리 겹칠 위험이 컸다. HUB→랙 진입 구간은 이미
    # 세분화(_subdivide_edge)돼 있어 길어져도 안전하지만, 복귀 구간은 세분화가
    # 안 돼 있어서 짧게 유지하는 쪽이 유리하다.
    "HUB_A":        {"position": (-1.25, 4.3, 0.0), "neighbors": ["RackA_280", "RackA_280_detour"]},
    "HUB_B":        {"position": (-3.75, 4.8, 0.0), "neighbors": ["RackB_280", "RackB_280_detour"]},
    "HUB_C":        {"position": (13.15, 4.8, 0.0), "neighbors": ["RackC_280", "RackC_280_detour"]},

    # 3. 랙 A — 사용자 도면 기준 컨베이어에 더 가까운("안쪽") 랙. A/B 한 쌍
    # 중에서는 A가 안쪽, B가 바깥쪽 — 근접/detour 통로 전체를 B 대비 X를 픽업
    # 열 중심에 더 가깝게 둬서 표현한다.
    #
    # 근접/detour는 두 개의 완전히 분리된 통로. 신발은 항상 280→260→240 순서로
    # 내려놓는다(사용자 지정). 두 통로가 같은 노드를 반대 방향으로 공유하면
    # 서로 마주보고 맞물려 데드락이 나므로(A는 260을 들고 240을 기다리고, B는
    # 240을 들고 260을 기다리는 식), 아예 노드 자체를 안 겹치게 detour용 사본을
    # 따로 둔다. 240/260/280(근접)은 원래 X,Y가 완전히 같고 Z(선반 높이)만
    # 달랐는데, 지상 로봇은 Z를 무시하므로 실제로는 같은 지점이었다 —
    # 사이즈별로 Y(랙 안쪽 깊이)를 갈라 물리적으로도 분리했고, detour 통로는 아예
    # 다른 X대(근접 통로 옆, 중심에서 바깥 방향으로 0.6m)를 써서 두 직선이
    # 좌우로 나란히 떨어지게 했다.
    "RackA_280":       {"position": (-1.25, 3.5, 2.1), "neighbors": ["RackA_260"]},
    "RackA_260":       {"position": (-1.25, 3.0, 1.2), "neighbors": ["RackA_240"]},
    "RackA_240":       {"position": (-1.25, 2.5, 0.3), "neighbors": ["RackA_OUT"]},
    "RackA_280_detour":   {"position": (-1.85, 3.5, 2.1), "neighbors": ["RackA_260_detour"]},
    "RackA_260_detour":   {"position": (-1.85, 3.0, 1.2), "neighbors": ["RackA_240_detour"]},
    "RackA_240_detour":   {"position": (-1.85, 2.5, 0.3), "neighbors": ["RackA_OUT"]},
    "RackA_OUT":     {"position": (-1.25, 2.0, 0.0), "neighbors": [PICKUP_NODE["A"], PICKUP_WAIT_NODE["A"]]},

    # 4. 랙 B — A보다 컨베이어에서 더 먼("바깥쪽") 랙.
    "RackB_280":       {"position": (-3.75, 3.5, 2.1), "neighbors": ["RackB_260"]},
    "RackB_260":       {"position": (-3.75, 3.0, 1.2), "neighbors": ["RackB_240"]},
    "RackB_240":       {"position": (-3.75, 2.5, 0.3), "neighbors": ["RackB_OUT"]},
    "RackB_280_detour":   {"position": (-4.35, 3.5, 2.1), "neighbors": ["RackB_260_detour"]},
    "RackB_260_detour":   {"position": (-4.35, 3.0, 1.2), "neighbors": ["RackB_240_detour"]},
    "RackB_240_detour":   {"position": (-4.35, 2.5, 0.3), "neighbors": ["RackB_OUT"]},
    "RackB_OUT":     {"position": (-3.75, 2.0, 0.0), "neighbors": [PICKUP_NODE["B"], PICKUP_WAIT_NODE["B"]]},

    # 5. 랙 C — 픽업/대기 구역 전체를 오른쪽으로 크게 밀어냈다(원래
    # 1.25 → 13.15, +11.9m 이동). 픽업 지점(PICKUP_C)은 그대로 두고
    # 랙+경유지만 옮겼으므로 PICKUP_C→HUB_C 본선이 그만큼 길어지는데,
    # 이미 세분화돼 있어 안전하다.
    "RackC_280":       {"position": (13.15, 3.5, 2.1), "neighbors": ["RackC_260"]},
    "RackC_260":       {"position": (13.15, 3.0, 1.2), "neighbors": ["RackC_240"]},
    "RackC_240":       {"position": (13.15, 2.5, 0.3), "neighbors": ["RackC_OUT"]},
    "RackC_280_detour":   {"position": (13.75, 3.5, 2.1), "neighbors": ["RackC_260_detour"]},
    "RackC_260_detour":   {"position": (13.75, 3.0, 1.2), "neighbors": ["RackC_240_detour"]},
    "RackC_240_detour":   {"position": (13.75, 2.5, 0.3), "neighbors": ["RackC_OUT"]},
    "RackC_OUT":     {"position": (13.75, 2.0, 0.0), "neighbors": [PICKUP_NODE["C"], PICKUP_WAIT_NODE["C"]]},
}

# 배치(5켤레) 작업을 전부 마친 로봇, 혹은 그냥 파트너에게 PICKUP_X를 양보해야
# 하는 로봇이 대신 대기하는 지점(종류별 전용). PICKUP_X 바로 뒤(HUB 반대 방향)에
# 배치했다 — 초기 스폰 위치로도 재사용된다(ROBOT_HOME_NODE 참고).
for _shoe_type in SHOE_TYPES:
    _pickup_pos = NODE_GRAPH[PICKUP_NODE[_shoe_type]]["position"]
    NODE_GRAPH[PICKUP_WAIT_NODE[_shoe_type]] = {
        "position": (_pickup_pos[0], -0.5, _pickup_pos[2]),
        "neighbors": [PICKUP_NODE[_shoe_type]],
    }

# PICKUP_X → HUB_X 굴절점(종류별 하나) — HUB_X_APPROACH를 "HUB의 X, 픽업의 Y"로
# 두면 HUB_APPROACH→HUB 구간(수직)이 HUB와 같은 X를 쓰는 근접 레인 랙 컬럼
# (RackX_280/260/240, Y=6.25~8.75대)을 그대로 관통해버린다 — 대각선만 없앴지
# "세로로 랙을 뚫고 지나가는" 문제가 남아있었다. 대신 "픽업의 X, HUB의 Y"로
# 둬서: 픽업 자신의 전용 컬럼(다른 종류의 랙과 절대 안 겹치는 X)을 타고 랙
# 꼭대기보다 높은 HUB의 Y까지 먼저 수직으로 올라간 다음, 랙 위쪽(랙 영역보다
# 높은 층)에서만 수평으로 HUB까지 이동한다 — 이러면 랙 영역(Y=6.25~8.75대)을
# 수직/수평 어느 구간에서도 절대 지나가지 않는다.
for _shoe_type in SHOE_TYPES:
    _pickup_x = NODE_GRAPH[PICKUP_NODE[_shoe_type]]["position"][0]
    _hub_y = NODE_GRAPH[f"HUB_{_shoe_type}"]["position"][1]
    NODE_GRAPH[f"HUB_{_shoe_type}_APPROACH"] = {
        "position": (_pickup_x, _hub_y, 0.0),
        "neighbors": [f"HUB_{_shoe_type}"],
    }
    NODE_GRAPH[PICKUP_NODE[_shoe_type]]["neighbors"] = [f"HUB_{_shoe_type}_APPROACH"]

# RackX_OUT → PICKUP_X 굴절점(종류별 하나) — 복귀할 때도 OUT에서 픽업까지
# 대각선으로 바로 가면 랙 구역을 가로지른다. OUT의 X를 그대로 따라 수직으로
# 내려온 다음, 픽업 열의 깊이(Y=0)에서 수평으로 이동하게 중간점을 끼워 넣는다
# — 나가는 길(HUB_X_APPROACH)과 정확히 대칭되는 구조.
#
# B는 "바깥쪽" 타입이라 RackB_OUT의 X가 짝인 안쪽 타입(A)보다 픽업 열
# 중심에서 더 멀다. APPROACH를 전부 같은 Y(픽업 열 깊이)에 두면, B의
# OUT→APPROACH→PICKUP 구간이 그 사이에 낀 안쪽 타입(A)의 픽업 지점을 그대로
# 가로지른다 — B만 APPROACH의 Y를 한 단 더 낮춰서(픽업 열보다 얕은 깊이)
# 안쪽 타입 동선과 아예 다른 깊이에서 지나가게 분리한다. C는 짝이 없어졌지만
# (원본에서 D와 짝이었음) 좌표는 원본 그대로 유지했다.
_APPROACH_Y_OFFSET = {"A": 0.0, "B": -1.0, "C": -1.0}
for _shoe_type in SHOE_TYPES:
    _out_x = NODE_GRAPH[f"Rack{_shoe_type}_OUT"]["position"][0]
    _approach_y = NODE_GRAPH[PICKUP_NODE[_shoe_type]]["position"][1] + _APPROACH_Y_OFFSET[_shoe_type]
    NODE_GRAPH[f"PICKUP_{_shoe_type}_APPROACH"] = {
        "position": (_out_x, _approach_y, 0.0),
        # PICKUP_X가 비어있어도 APPROACH에서 곧장 들어가지 않고 항상
        # PICKUP_WAIT_X를 먼저 거치게 한다 — 복귀 경로를 단일 통로로 고정해서
        # (PICKUP_X 직행/PICKUP_WAIT_X 대기 두 갈래로 갈리지 않게) 진입 방향을
        # 하나로 통일한다. PICKUP_WAIT_X → PICKUP_X 엣지는 원래 있던 걸 그대로 탄다.
        "neighbors": [PICKUP_WAIT_NODE[_shoe_type]],
    }
    NODE_GRAPH[f"Rack{_shoe_type}_OUT"]["neighbors"] = [f"PICKUP_{_shoe_type}_APPROACH"]

# 전체 포인트가 서로 너무 가깝다는 피드백으로 X,Y만 일괄로 넓힌다(Z는 선반
# 높이라 스케일하면 랙이 비정상적으로 커지므로 그대로 둔다). 아래쪽 본선
# 세분화(_subdivide_edge)는 이 스케일이 다 적용된 뒤의 실제 거리를 기준으로
# 칸을 나누므로, 칸 크기(1.2m 목표)는 이 배율과 무관하게 항상 정확하게 유지된다.
_GRAPH_SCALE = 2.5
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
# 종류별로 완전히 분리된 지금도, PICKUP_X → HUB_X_APPROACH → HUB_X 구간만큼은
# 그 종류를 담당하는 로봇 2대가 같이 쓰는 "본선"이다(그 뒤 HUB_X →
# RackX_280/280_detour는 둘 중 하나로만 갈라지는 지선이라 원래도 안 겹침). 이
# 본선을 통짜 간선으로 두면 로봇 1대가 다 지나갈 때까지 파트너가 못 나가므로,
# 로봇 한 대 칸(약 1.2m) 단위로 잘게 쪼개서 두 로봇이 기차처럼 꼬리를 물고
# 지나갈 수 있게 한다. 굴절점(HUB_X_APPROACH) 전후 두 구간 다 세분화한다.
_MAIN_LINE_SEGMENT_M = 1.2
for _shoe_type in SHOE_TYPES:
    _subdivide_edge(PICKUP_NODE[_shoe_type], f"HUB_{_shoe_type}_APPROACH", _MAIN_LINE_SEGMENT_M)
    _subdivide_edge(f"HUB_{_shoe_type}_APPROACH", f"HUB_{_shoe_type}", _MAIN_LINE_SEGMENT_M)
    # 복귀 본선(OUT → PICKUP_X_APPROACH → PICKUP_WAIT_X → PICKUP_X)도 같은
    # 이유로 세분화한다 — PICKUP_X_APPROACH는 이제 PICKUP_WAIT_X 하나로만
    # 이어지므로(PICKUP_X 직행 갈래 없음) 갈림 없는 단일 경로만 쪼개면 된다.
    _subdivide_edge(f"Rack{_shoe_type}_OUT", f"PICKUP_{_shoe_type}_APPROACH", _MAIN_LINE_SEGMENT_M)
    _subdivide_edge(f"PICKUP_{_shoe_type}_APPROACH", PICKUP_WAIT_NODE[_shoe_type], _MAIN_LINE_SEGMENT_M)
