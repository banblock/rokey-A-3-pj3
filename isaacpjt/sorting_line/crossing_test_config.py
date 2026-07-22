"""edge_conflict(구간 충돌) 회피 로직 전용 테스트 그래프.

실제 운영 그래프(fleet_config.py)는 로봇마다 경로를 물리적으로 거의 분리해뒀기
때문에, 두 로봇이 "노드와 노드 사이"에서 마주치는(공유하는 노드 없이 구간이
기하학적으로만 교차하는) 상황이 사실상 안 생긴다 — 그래서 fms_node.py의
_find_edge_conflict가 실제로 로봇을 멈춰 세우는 걸 눈으로 확인하기 어렵다. 이
모듈은 오직 그 로직 하나만 확인하기 위한 용도로, 로봇 2대가 X자로 교차하는
대각선을 영원히 왕복하게 만든 아주 작은 그래프다.

    R1_HOME(-2,-2)  \\           /  R2_FAR(2,-2)
                      \\         /
                        (0,0)  <- 교차점(그래프 노드 아님, 두 간선이 기하학적으로만 겹침)
                      /         \\
    R2_HOME(-2, 2)  /           \\  R1_FAR(2, 2)

R1: R1_HOME ↔ R1_FAR (좌하 ↔ 우상)을 왕복
R2: R2_HOME ↔ R2_FAR (좌상 ↔ 우하)을 왕복

두 간선 다 중간에 노드가 없는 단일 구간이고 두 로봇은 공유하는 노드가 하나도
없다 — node_locks(노드 단위 뮤텍스)만으로는 절대 충돌을 못 막는 상황을
일부러 만든 것이다. 오직 _find_edge_conflict(두 이동 구간의 기하학적 교차
판정)만이 이 상황에서 로봇을 멈춰 세울 수 있다.
"""

import heapq
import math

NODE_GRAPH = {
    "R1_HOME": {"position": (-2.0, -2.0, 0.0), "neighbors": ["R1_FAR"]},
    "R1_FAR":  {"position": (2.0, 2.0, 0.0), "neighbors": ["R1_HOME"]},
    "R2_HOME": {"position": (-2.0, 2.0, 0.0), "neighbors": ["R2_FAR"]},
    "R2_FAR":  {"position": (2.0, -2.0, 0.0), "neighbors": ["R2_HOME"]},
}

ROBOT_HOME_NODE = {"amr_1": "R1_HOME", "amr_2": "R2_HOME"}
# 로봇 구분용 색(비콘 표시 등) — fleet_config의 ROBOT_SHOE_TYPE과 같은 역할이지만
# 여기선 신발 종류가 아니라 그냥 로봇 1/2 구분이 목적이라 이름을 다르게 뒀다.
ROBOT_LABEL = {"amr_1": "R1", "amr_2": "R2"}

CROSSING_POINT = (0.0, 0.0, 0.0)  # 두 대각선이 겹치는 지점 — 그래프 노드 아님, 시각화 전용


def _distance(a, b):
    ax, ay, az = NODE_GRAPH[a]["position"]
    bx, by, bz = NODE_GRAPH[b]["position"]
    return ((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2) ** 0.5


def shortest_path(start, goal):
    """fleet_config.shortest_path와 동일한 다익스트라 구현. fleet_config 쪽
    shortest_path는 그 모듈 전역 NODE_GRAPH를 닫혀서(closure) 참조하므로 다른
    그래프에 재사용할 수 없어서, 이 모듈 전용으로 하나 더 둔다."""
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


def robot_spawn_yaw(robot_id):
    """fleet_config.robot_spawn_yaw와 동일한 로직(스폰 자세를 첫 이동 방향과
    맞춘다) — fleet_driver.py가 오도메트리를 월드 좌표로 바꿀 때도 같은 값을
    써야 하므로 그래프별로 하나씩 둔다."""
    home_node = ROBOT_HOME_NODE[robot_id]
    neighbors = NODE_GRAPH[home_node]["neighbors"]
    if not neighbors:
        return 0.0
    hx, hy, _hz = NODE_GRAPH[home_node]["position"]
    nx, ny, _nz = NODE_GRAPH[neighbors[0]]["position"]
    return math.atan2(ny - hy, nx - hx)
