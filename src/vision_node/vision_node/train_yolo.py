"""
YOLO11-seg 학습 - Isaac Sim SDG로 생성한 신발 정상/훼손(normal/defect) 데이터셋

사용:
    pip install ultralytics
    python train_yolo_seg.py --data dataset_shoe_v1/data.yaml --epochs 100
"""

import argparse

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="convert_sdg_to_yolo.py가 생성한 data.yaml 경로")
    ap.add_argument("--model", default="yolo26m-seg.pt", help="yolo11n/s/m-seg.pt 중 선택")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=960, help="학습 해상도 (원본 1440은 학습 속도상 과함)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--project", default="runs_shoe_seg")
    ap.add_argument("--name", default="v2")
    a = ap.parse_args()

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