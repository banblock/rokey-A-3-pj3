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


# ╔══════════════════════════════════════════════════════════════╗
# ║  B. 노드 그래프 정의 (완벽한 일방통행 및 순환 구조 적용)               ║
# ╚══════════════════════════════════════════════════════════════╝
# 주의: 이전에 있던 "양방향 자동 보정 코드"는 데드락을 유발하므로 완전히 삭제했습니다.

NODE_GRAPH = {
    # 1. 픽업 지점 (WAIT_N 각각에서 개별적으로 들어와 하나의 PICKUP으로 합류)
    "PICKUP":       {"position": (1.7, 0.0, 0.794), "neighbors": ["HUB_1"]},

    # 2. 회전교차로 (HUB를 4개로 쪼개어 로봇들이 기차처럼 순환할 수 있게 함)
    "HUB_1":        {"position": (0.0, 2.0, 0.0),   "neighbors": ["RackA_IN", "RackB_IN", "HUB_2"]},
    "HUB_2":        {"position": (0.0, 3.4, 0.0),   "neighbors": ["RackC_IN", "RackD_IN", "HUB_3"]},
    "HUB_3":        {"position": (3.0, 3.4, 0.0),   "neighbors": ["BUFFER1", "BUFFER2", "HUB_4"]},
    "HUB_4":        {"position": (3.0, 2.0, 0.0),   "neighbors": WAIT_NODE_IDS + ["HUB_1"]},  # 각 로봇 홈 슬롯 + 루프 계속

    # 3. 랙 A (들어가는 길과 나오는 길을 분리하여 일방통행 유지)
    "RackA_IN":     {"position": (-3.75, 3.4, 0.0), "neighbors": ["RackA_소", "RackA_중", "RackA_대"]},
    "RackA_소":      {"position": (-3.75, 4.3, 0.3), "neighbors": ["RackA_OUT"]},
    "RackA_중":      {"position": (-3.75, 4.3, 1.2), "neighbors": ["RackA_OUT"]},
    "RackA_대":      {"position": (-3.75, 4.3, 2.1), "neighbors": ["RackA_OUT"]},
    "RackA_OUT":    {"position": (-2.0, 4.3, 0.0),  "neighbors": ["HUB_2"]},  # 작업 후 다음 교차로 합류

    # 4. 랙 B
    "RackB_IN":     {"position": (-1.25, 3.4, 0.0), "neighbors": ["RackB_소", "RackB_중", "RackB_대"]},
    "RackB_소":      {"position": (-1.25, 4.3, 0.3), "neighbors": ["RackB_OUT"]},
    "RackB_중":      {"position": (-1.25, 4.3, 1.2), "neighbors": ["RackB_OUT"]},
    "RackB_대":      {"position": (-1.25, 4.3, 2.1), "neighbors": ["RackB_OUT"]},
    "RackB_OUT":    {"position": (-0.5, 4.3, 0.0),  "neighbors": ["HUB_2"]},

    # 5. 랙 C
    "RackC_IN":     {"position": (1.25, 3.4, 0.0),  "neighbors": ["RackC_소", "RackC_중", "RackC_대"]},
    "RackC_소":      {"position": (1.25, 4.3, 0.3),  "neighbors": ["RackC_OUT"]},
    "RackC_중":      {"position": (1.25, 4.3, 1.2),  "neighbors": ["RackC_OUT"]},
    "RackC_대":      {"position": (1.25, 4.3, 2.1),  "neighbors": ["RackC_OUT"]},
    "RackC_OUT":    {"position": (2.0, 4.3, 0.0),   "neighbors": ["HUB_3"]},

    # 6. 랙 D
    "RackD_IN":     {"position": (3.75, 3.4, 0.0),  "neighbors": ["RackD_소", "RackD_중", "RackD_대"]},
    "RackD_소":      {"position": (3.75, 4.3, 0.3),  "neighbors": ["RackD_OUT"]},
    "RackD_중":      {"position": (3.75, 4.3, 1.2),  "neighbors": ["RackD_OUT"]},
    "RackD_대":      {"position": (3.75, 4.3, 2.1),  "neighbors": ["RackD_OUT"]},
    "RackD_OUT":    {"position": (4.5, 4.3, 0.0),   "neighbors": ["HUB_3"]},

    # 7. 대기 구역 (버퍼)
    "BUFFER1":      {"position": (1.5, 5.7, 0.0),   "neighbors": ["HUB_4"]},
    "BUFFER2":      {"position": (3.5, 5.7, 0.0),   "neighbors": ["HUB_4"]},
}

# 로봇 수만큼 전용 대기 슬롯을 생성해 그래프에 붙인다 (PICKUP 앞에 나란히 배치).
# 간격은 Nova Carter 실측 트랙폭(0.4132m)보다 충분히 여유 있게 잡는다 — 차체(캐스터
# 포함)는 트랙폭보다 더 커서, 너무 좁게 스폰하면 스폰 직후 서로 겹쳐 PhysX가
# 충돌로 인식하고 강하게 밀어내면서 로봇이 튕겨나가는 문제가 생긴다.
#
# PICKUP과의 거리(_WAIT_BASE_Y)는 일부러 멀리 뒀다 — 랙/컨베이어 노드들은 실제
# 스폰된 USD 에셋 위치와 맞춰져 있어서 함부로 못 늘리지만, WAIT 슬롯은 어떤
# 실물 에셋과도 안 묶인 빈 공간이라 여기를 멀리 떨어뜨리면 매 태스크마다
# 왕복 구간이 길어져서 실제 이동을 눈으로 관측하기 훨씬 쉬워진다.
_WAIT_SPACING_M = 1.0
_WAIT_BASE_X, _WAIT_BASE_Y, _WAIT_BASE_Z = 1.7, -6.0, 0.0
for _i, _wait_node in enumerate(WAIT_NODE_IDS):
    NODE_GRAPH[_wait_node] = {
        "position": (_WAIT_BASE_X - _i * _WAIT_SPACING_M, _WAIT_BASE_Y, _WAIT_BASE_Z),
        "neighbors": ["PICKUP"],
    }


def _distance(a, b):
    ax, ay, az = NODE_GRAPH[a]["position"]
    bx, by, bz = NODE_GRAPH[b]["position"]
    return ((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2) ** 0.5


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
