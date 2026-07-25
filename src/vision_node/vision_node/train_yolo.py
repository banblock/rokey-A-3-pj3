"""
YOLO11-seg 학습 - Isaac Sim SDG로 생성한 신발 정상/훼손(normal/defect) 데이터셋

사용:
    pip install ultralytics
    python train_yolo_seg.py --data dataset_shoe_v1/data.yaml --epochs 100
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="convert_sdg_to_yolo.py/build_crop_dataset.py가 생성한 data.yaml 경로")
    ap.add_argument("--model", default="yolo26m-seg.pt", help="yolo11n/s/m-seg.pt 중 선택")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=960, help="학습 해상도 (원본 1440은 학습 속도상 과함)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--project", default="runs_shoe_seg")
    # --name을 안 주면 데이터셋 폴더 이름으로 자동 설정 (예: dataset_shoe_v5_crop -> v5_crop).
    # 다른 데이터셋으로 여러 번 학습해도 결과 폴더가 안 겹치게 하기 위함 — 예전엔 기본값이
    # "v2"로 고정돼 있어서 다른 데이터셋을 학습해도 계속 같은 이름으로 저장되던 문제가 있었다.
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    if a.name is None:
        a.name = Path(a.data).resolve().parent.name.replace("dataset_shoe_", "")

    model = YOLO(a.model)
    model.train(
        data=a.data,
        epochs=a.epochs,
        imgsz=a.imgsz,
        batch=a.batch,
        project=a.project,
        name=a.name,
    )


if __name__ == "__main__":
    main()