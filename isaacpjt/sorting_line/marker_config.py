"""AGV 스타일 AprilTag/ArUco 마커 위치추정을 위한 공용 데이터/로직 모듈.

fleet_config.py와 같은 이유로 순수 데이터/로직만 담고 rclpy나 Isaac Sim(pxr,
omni.*) 의존성이 전혀 없다 — Isaac Sim 씬 스크립트(마커 배치)와 ROS2 노드
(마커 검출/보정, generate_markers.py)가 전부 그대로 가져다 쓸 수 있어야
하기 때문이다. setup.py의 data_files로 fleet_config.py와 같은 방식으로
share/sorting_line_fms/ 밑에 같이 설치한다.

NODE_GRAPH를 인자로 받는 함수 형태로 짠 이유는 fleet_driver.py의
config_module 패턴과 같다 — 어떤 그래프(fleet_config, fleet_config_test1 등)를
쓰는지는 호출하는 쪽이 정하고, 이 모듈은 "정밀도가 중요한 지점"을 어떻게
고르고 번호를 매기는지만 책임진다.
"""

MARKER_DICT_NAME = "DICT_APRILTAG_36h11"  # cv2.aruco의 속성 이름 문자열(587개 고유 ID)
MARKER_IMAGE_PX = 400  # 생성할 태그 이미지 한 변 픽셀 수
MARKER_SIZE_M = 0.15  # 실측 후 조정 예정 — 현재는 자리표시용 플레이스홀더(바닥 부착 태그 한 변 길이)

_RACK_SIZES = ("280", "260", "240")


def is_marker_node(node_id):
    """정밀도가 중요해서 마커를 붙여야 하는 노드인지 판정한다 — PICKUP_X,
    PICKUP_WAIT_X(및 확장 대기슬롯), RackX_280/260/240(근접+detour). 세분화
    칸("__" 포함)이나 APPROACH/HUB/OUT 같은 통과 지점은 대상이 아니다 —
    거긴 로봇이 잠깐 지나가기만 할 뿐, 정확한 정지가 필요한 자리가 아니라서
    드리프트 보정의 의미가 크지 않다."""
    if "__" in node_id:
        return False
    if node_id.startswith("PICKUP_") and "APPROACH" not in node_id:
        return True
    if node_id.startswith("Rack"):
        parts = node_id.split("_")
        return len(parts) >= 2 and parts[1] in _RACK_SIZES
    return False


def build_marker_maps(node_graph):
    """node_graph(예: fleet_config.NODE_GRAPH)에서 마커 대상 노드를 추려
    (node_id -> marker_id), (marker_id -> node_id) 두 매핑을 만든다.

    번호는 노드 이름 정렬 순서로 0부터 순차 부여한다 — 그래프에 랙/픽업이
    늘어나도(SHELF_INDEX와 같은 패턴) 코드 변경 없이 자동으로 늘어난다.
    두 그래프(fleet_config vs fleet_config_test1 등)가 노드 이름이 다르면
    같은 이름이라도 서로 다른 번호를 받을 수 있다 — 마커 검출 노드를 어떤
    config_module로 띄우는지가 항상 씬 쪽 config_module과 일치해야 한다.
    """
    marker_nodes = sorted(node_id for node_id in node_graph if is_marker_node(node_id))
    node_id_by_marker_id = dict(enumerate(marker_nodes))
    marker_id_by_node_id = {node_id: marker_id for marker_id, node_id in node_id_by_marker_id.items()}
    return marker_id_by_node_id, node_id_by_marker_id
