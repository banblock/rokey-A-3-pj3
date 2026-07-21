#!/usr/bin/env python3
"""노드 그래프 / 로봇-종류 배정 — fms_node.py, fleet_driver.py, Isaac Sim
스크립트(1_conveyor_sorter_env.py, 3_amr_fleet.py)가 전부 공유하는 순수 데이터 모듈.

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
SHOE_TYPES = ["A", "B", "C", "D"]
ROBOTS_PER_TYPE = 2

ROBOT_SHOE_TYPE = {
    f"amr_{i + 1}": shoe_type
    for i, shoe_type in enumerate(
        shoe_type for shoe_type in SHOE_TYPES for _ in range(ROBOTS_PER_TYPE)
    )
}
# {"amr_1": "A", "amr_2": "A", "amr_3": "B", "amr_4": "B",
#  "amr_5": "C", "amr_6": "C", "amr_7": "D", "amr_8": "D"}

WAIT_NODE_IDS = [f"WAIT_{i + 1}" for i in range(len(ROBOT_SHOE_TYPE))]
ROBOT_HOME_NODE = dict(zip(ROBOT_SHOE_TYPE.keys(), WAIT_NODE_IDS))

# 신발 종류별 전용 픽업 지점 — 로봇은 자기 담당 종류의 픽업 지점에서만 신발을 받는다.
PICKUP_NODE = {shoe_type: f"PICKUP_{shoe_type}" for shoe_type in SHOE_TYPES}
# 배치를 다 끝낸 로봇이 PICKUP이 이미 점유돼 있을 때 대신 대기하는 지점(종류별 전용).
PICKUP_WAIT_NODE = {shoe_type: f"PICKUP_WAIT_{shoe_type}" for shoe_type in SHOE_TYPES}

# 신발 종류별 전용 홈 슬롯 목록 — 랙에서 나온 로봇이 다른 종류와 공유하는
# 경유지 없이 바로 자기 종류의 홈으로 돌아갈 수 있게 한다.
TYPE_WAIT_NODES = {
    shoe_type: [
        ROBOT_HOME_NODE[robot_id]
        for robot_id in ROBOT_SHOE_TYPE
        if ROBOT_SHOE_TYPE[robot_id] == shoe_type
    ]
    for shoe_type in SHOE_TYPES
}


# ╔══════════════════════════════════════════════════════════════╗
# ║  B. 노드 그래프 정의 (완벽한 일방통행 및 순환 구조 적용)               ║
# ╚══════════════════════════════════════════════════════════════╝

NODE_GRAPH = {
    # 1. 픽업 지점 (신발 종류별로 분리 — 각 WAIT_N은 자기 담당 종류의 픽업
    # 지점으로만 들어간다). 실제 컨베이어는 아직 단일 벨트/단일 픽업 지점이라
    # 아래 4곳의 좌표는 임시값이다 — 종류별 벨트/분기 배치가 실제로 정해지면
    # 실측치로 교체해야 한다. 간격을 2.0m로 둔 건 Nova Carter 스폰 충돌
    # 전례(0.5m 확실히 충돌, 1.0m도 부족, 2.0m에서 해결됨) 때문 — WAIT_N도
    # 이 X를 그대로 따라가므로, 여기가 좁으면 로봇 스폰 위치도 같이 좁아진다.
    "PICKUP_A":     {"position": (1.7, 0.0, 0.794), "neighbors": ["HUB_A"]},
    "PICKUP_B":     {"position": (3.7, 0.0, 0.794), "neighbors": ["HUB_B"]},
    "PICKUP_C":     {"position": (5.7, 0.0, 0.794), "neighbors": ["HUB_C"]},
    "PICKUP_D":     {"position": (7.7, 0.0, 0.794), "neighbors": ["HUB_D"]},

    # 2. 종류별 전용 경유지 — 다른 종류 로봇과는 어떤 노드도 공유하지 않는다
    # (자기 종류 2대끼리만 노드 락을 놓고 경쟁). 예전에 8대 전체가 HUB_1~4
    # 순환 루프 하나를 공유했을 때는, A종류 로봇이 경로 중간에 멈추면 그
    # 루프를 지나야 하는 C/D종류 로봇까지 덩달아 막히는 문제가 있었다
    # (본선-지선 미분리 문제) — 지금은 종류별로 픽업→경유지→랙→복귀까지
    # 완전히 독립된 통로를 쓴다.
    "HUB_A":        {"position": (-3.75, 2.0, 0.0), "neighbors": ["RackA_대", "RackA_대_우회"]},
    "HUB_B":        {"position": (-1.25, 2.0, 0.0), "neighbors": ["RackB_대", "RackB_대_우회"]},
    "HUB_C":        {"position": (1.25, 2.0, 0.0),  "neighbors": ["RackC_대", "RackC_대_우회"]},
    "HUB_D":        {"position": (3.75, 2.0, 0.0),  "neighbors": ["RackD_대", "RackD_대_우회"]},

    # 3. 랙 A — 컨베이어에서 가장 가까운 랙(오른쪽 앞줄). 실제 창고 배치는
    # 컨베이어 기준 앞뒤로 두 랙씩 겹쳐있다: 오른쪽은 앞(A)/뒤(B), 왼쪽은
    # 앞(D)/뒤(C) — 뒤쪽 랙(B, C)은 아래에서 Y를 더 뒤로 밀어 물리적으로 더
    # 먼 거리를 표현한다.
    #
    # 근접/우회는 두 개의 완전히 분리된 통로. 신발은 항상 대→중→소 순서로
    # 내려놓는다(사용자 지정). 두 통로가 같은 노드를 반대 방향으로 공유하면
    # 서로 마주보고 맞물려 데드락이 나므로(A는 중을 들고 소를 기다리고, B는
    # 소를 들고 중을 기다리는 식), 아예 노드 자체를 안 겹치게 우회용 사본을
    # 따로 둔다. 소/중/대(근접)는 원래 X,Y가 완전히 같고 Z(선반 높이)만
    # 달랐는데, 지상 로봇은 Z를 무시하므로 실제로는 같은 지점이었다 —
    # 사이즈별로 Y(랙 안쪽 깊이)를 갈라 물리적으로도 분리했고, 우회 통로는 아예
    # 다른 Y대(허브와 근접 통로 사이 빈 공간)를 쓴다. 근접/우회 두 직선(대-중-소를
    # 잇는 선)이 X까지 같으면 앞뒤로 겹쳐 보이므로, 우회 쪽 X를 중심에서 바깥
    # 방향으로 0.6m 밀어서 두 직선이 좌우로 나란히 떨어지게 했다.
    "RackA_대":       {"position": (-3.75, 2.5, 2.1), "neighbors": ["RackA_중"]},
    "RackA_중":       {"position": (-3.75, 3.0, 1.2), "neighbors": ["RackA_소"]},
    "RackA_소":       {"position": (-3.75, 3.5, 0.3), "neighbors": ["RackA_OUT"]},
    "RackA_대_우회":   {"position": (-4.35, 2.5, 2.1), "neighbors": ["RackA_중_우회"]},
    "RackA_중_우회":   {"position": (-4.35, 3.0, 1.2), "neighbors": ["RackA_소_우회"]},
    "RackA_소_우회":   {"position": (-4.35, 3.5, 0.3), "neighbors": ["RackA_OUT"]},
    "RackA_OUT":     {"position": (-2.0, 4.3, 0.0),  "neighbors": TYPE_WAIT_NODES["A"]},  # 작업 후 바로 자기 홈으로 복귀

    # 4. 랙 B — 컨베이어에서 A보다 더 먼 안쪽 랙. 실제 창고 배치상 A/B가 앞뒤로
    # 겹쳐있는 한 쌍이라, 근접/우회 통로 전체를 A 대비 Y로 3.0m씩 더 뒤로
    # 밀어서 "더 멀다"를 물리적으로 표현한다. 이 정도는 밀어야 HUB_1에서
    # 잰 실제 직선거리가 A보다 확실히 길어진다 — B가 A보다 HUB_1의 X(=0.0)에
    # 더 가까운 X(-1.25 vs -3.75)에 있어서, 조금만 밀면 X 이득 때문에 오히려
    # B가 더 가깝게 계산돼버린다(1.7m로는 부족, 2.0m도 근소해서 3.0m로 확보).
    "RackB_대":       {"position": (-1.25, 2.5, 2.1), "neighbors": ["RackB_중"]},
    "RackB_중":       {"position": (-1.25, 3.0, 1.2), "neighbors": ["RackB_소"]},
    "RackB_소":       {"position": (-1.25, 3.5, 0.3), "neighbors": ["RackB_OUT"]},
    "RackB_대_우회":   {"position": (-1.85, 2.5, 2.1), "neighbors": ["RackB_중_우회"]},
    "RackB_중_우회":   {"position": (-1.85, 3.0, 1.2), "neighbors": ["RackB_소_우회"]},
    "RackB_소_우회":   {"position": (-1.85, 3.5, 0.3), "neighbors": ["RackB_OUT"]},
    "RackB_OUT":     {"position": (-0.5, 4.3, 0.0),  "neighbors": TYPE_WAIT_NODES["B"]},

    # 5. 랙 C — 컨베이어에서 D보다 더 먼 안쪽 랙(B와 같은 이유로 Y를 3.0m 더 뒤로 뺌).
    "RackC_대":       {"position": (1.25, 2.5, 2.1),  "neighbors": ["RackC_중"]},
    "RackC_중":       {"position": (1.25, 3.0, 1.2),  "neighbors": ["RackC_소"]},
    "RackC_소":       {"position": (1.25, 3.5, 0.3),  "neighbors": ["RackC_OUT"]},
    "RackC_대_우회":   {"position": (1.85, 2.5, 2.1),  "neighbors": ["RackC_중_우회"]},
    "RackC_중_우회":   {"position": (1.85, 3.0, 1.2),  "neighbors": ["RackC_소_우회"]},
    "RackC_소_우회":   {"position": (1.85, 3.5, 0.3),  "neighbors": ["RackC_OUT"]},
    "RackC_OUT":     {"position": (2.0, 4.3, 0.0),   "neighbors": TYPE_WAIT_NODES["C"]},

    # 6. 랙 D
    "RackD_대":       {"position": (3.75, 2.5, 2.1),  "neighbors": ["RackD_중"]},
    "RackD_중":       {"position": (3.75, 3.0, 1.2),  "neighbors": ["RackD_소"]},
    "RackD_소":       {"position": (3.75, 3.5, 0.3),  "neighbors": ["RackD_OUT"]},
    "RackD_대_우회":   {"position": (4.35, 2.5, 2.1),  "neighbors": ["RackD_중_우회"]},
    "RackD_중_우회":   {"position": (4.35, 3.0, 1.2),  "neighbors": ["RackD_소_우회"]},
    "RackD_소_우회":   {"position": (4.35, 3.5, 0.3),  "neighbors": ["RackD_OUT"]},
    "RackD_OUT":     {"position": (4.5, 4.3, 0.0),   "neighbors": TYPE_WAIT_NODES["D"]},
}

# 로봇 수만큼 전용 대기 슬롯을 생성해 그래프에 붙인다. 예전엔 한 줄로 쭉
# 늘어놓고 픽업 지점과의 거리(_WAIT_BASE_Y)를 일부러 멀리(-6.0) 둬서 왕복
# 구간을 눈으로 관측하기 쉽게 했는데, 실제로 써보니 매 작업마다 왕복이 너무
# 오래 걸려서 각자 자기 종류의 PICKUP_X 바로 앞으로 붙였다. 같은 종류 2대는
# 앞뒤로(_WAIT_DEPTH_SPACING_M) 벌리는데, 이 값도 Nova Carter 스폰 충돌
# 전례(0.5m 확실히 충돌, 1.0m도 부족, 2.0m에서 해결됨)를 따라 2.0m로 잡았다.
# PICKUP_X 간격도 2.0m라서(위 섹션 참고) 종류끼리 X만으로 이미 2.0m 이상
# 떨어지므로 더 이상 Y를 어긋나게(stagger) 둘 필요가 없다.
_WAIT_DEPTH_SPACING_M = 2.0
_WAIT_BASE_Y, _WAIT_BASE_Z = -1.5, 0.0
_ROBOT_IDS_IN_ORDER = list(ROBOT_SHOE_TYPE.keys())
for _i, _wait_node in enumerate(WAIT_NODE_IDS):
    _robot_id = _ROBOT_IDS_IN_ORDER[_i]
    _wait_shoe_type = ROBOT_SHOE_TYPE[_robot_id]
    _within_type_idx = _i % ROBOTS_PER_TYPE
    _wait_x = NODE_GRAPH[PICKUP_NODE[_wait_shoe_type]]["position"][0]
    NODE_GRAPH[_wait_node] = {
        "position": (
            _wait_x,
            _WAIT_BASE_Y - _within_type_idx * _WAIT_DEPTH_SPACING_M,
            _WAIT_BASE_Z,
        ),
        # 자기 담당 종류의 픽업 지점/임시 대기 지점으로만 들어간다 — 다른
        # 종류의 픽업 지점을 거칠 일이 없다.
        "neighbors": [PICKUP_NODE[_wait_shoe_type], PICKUP_WAIT_NODE[_wait_shoe_type]],
    }

# 배치(5켤레) 작업을 전부 마친 로봇이 다음 배치를 기다리며 대기하는 지점(종류별
# 전용). PICKUP_X가 이미 같은 종류의 파트너 로봇에게 점유돼 있으면 여기서 약간
# 뒤에 물러나 대기한다 — 각자 자기 PICKUP_X 바로 앞(HUB 방향)에 배치했다.
for _shoe_type in SHOE_TYPES:
    _pickup_pos = NODE_GRAPH[PICKUP_NODE[_shoe_type]]["position"]
    NODE_GRAPH[PICKUP_WAIT_NODE[_shoe_type]] = {
        # WAIT_N을 픽업 바로 앞(-1.5)까지 당겨놔서, 여기 Y를 -1.0으로 두면
        # WAIT_N과 0.5m밖에 안 떨어져 로봇 두 대가 겹칠 수 있다 — PICKUP 쪽으로
        # 더 붙여서(-0.5) WAIT_N과는 충분히, PICKUP과는 "바로 뒤" 정도로 유지.
        "position": (_pickup_pos[0], -0.5, _pickup_pos[2]),
        "neighbors": [PICKUP_NODE[_shoe_type]],
    }


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
    for t in ["A", "B", "C", "D"]
    for s in ["소", "중", "대"]
}


# ╔══════════════════════════════════════════════════════════════╗
# ║  C. 본선 구간 세분화 (플래투닝)                                    ║
# ╚══════════════════════════════════════════════════════════════╝
# 종류별로 완전히 분리된 지금도, PICKUP_X → HUB_X 구간만큼은 그 종류를 담당하는
# 로봇 2대가 같이 쓰는 "본선"이다(그 뒤 HUB_X → RackX_대/대_우회는 둘 중 하나로만
# 갈라지는 지선이라 원래도 안 겹침). 이 본선을 통짜 간선 하나로 두면 로봇 1대가
# 다 지나갈 때까지 파트너가 못 나가므로, 로봇 한 대 칸(약 1.2m) 단위로 잘게
# 쪼개서 두 로봇이 기차처럼 꼬리를 물고 지나갈 수 있게 한다.
_MAIN_LINE_SEGMENT_M = 1.2
for _shoe_type in SHOE_TYPES:
    _subdivide_edge(PICKUP_NODE[_shoe_type], f"HUB_{_shoe_type}", _MAIN_LINE_SEGMENT_M)
