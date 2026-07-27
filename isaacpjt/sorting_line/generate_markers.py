#!/usr/bin/env python3
"""AGV 마커 이미지 일괄 생성 — marker_config.build_marker_maps()가 고른 노드마다
AprilTag PNG를 만들어 markers/ 밑에 저장한다. Isaac Sim 쪽(1_conveyor_sorter_env.py)이
이 이미지를 바닥 마커 프림의 텍스처로 그대로 참조한다.

Isaac Sim 의존성이 전혀 없는 순수 스크립트라 이 컴퓨터에서도 바로 실행/검증 가능.
그래프가 바뀌면(랙/픽업 추가 등) 다시 실행해서 이미지를 갱신하면 된다.
"""
import argparse
import os

import cv2
import cv2.aruco as aruco

from fleet_config import NODE_GRAPH
from marker_config import MARKER_DICT_NAME, MARKER_IMAGE_PX, build_marker_maps

DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "markers")


def generate(output_dir=DEFAULT_OUTPUT_DIR):
    id_by_node, _ = build_marker_maps(NODE_GRAPH)
    dictionary = aruco.getPredefinedDictionary(getattr(aruco, MARKER_DICT_NAME))

    os.makedirs(output_dir, exist_ok=True)
    for node_id, marker_id in sorted(id_by_node.items(), key=lambda kv: kv[1]):
        image = aruco.generateImageMarker(dictionary, marker_id, MARKER_IMAGE_PX)
        out_path = os.path.join(output_dir, f"{node_id}.png")
        cv2.imwrite(out_path, image)
        print(f"[{marker_id:3d}] {node_id:30s} -> {out_path}")

    print(f"\n총 {len(id_by_node)}개 마커 생성 완료 -> {output_dir}")
    return id_by_node


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR,
        help=f"생성된 PNG를 저장할 디렉터리 (기본값: {DEFAULT_OUTPUT_DIR})",
    )
    args = parser.parse_args()
    generate(args.output_dir)
