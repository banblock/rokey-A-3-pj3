"""
RT-DETR 학습 (train_yolo.py의 실험용 복사본) - stage2(tear crop) 모델을 YOLO 대신
RT-DETR로 학습해서 성능을 비교해보기 위한 스크립트.

RT-DETR도 train_yolo.py와 같은 ultralytics 패키지 안에 들어있어서, API가 거의 동일하다.
차이는 딱 두 곳뿐:
  1. `from ultralytics import YOLO` 대신 `from ultralytics import RTDETR`
  2. 기본 가중치가 "yolo11n.pt" 계열이 아니라 "rtdetr-l.pt"/"rtdetr-x.pt" (n/s/m 사이즈가
     없고 l/x 두 가지만 제공됨 - 그만큼 기본적으로 더 무거운 모델)

data.yaml, 클래스 정의(shoe/tear), 학습 결과 폴더 구조는 YOLO 학습 때와 완전히 동일하게
convert_sdg_to_yolo.py / build_crop_dataset.py가 만든 데이터셋을 그대로 재사용한다.

사용:
    pip install ultralytics   # RT-DETR도 이 안에 포함되어 있어 별도 설치 불필요
    python train_rtdetr.py --data dataset_shoe_v1/data.yaml --epochs 100

학습이 끝나면 runs_shoe_rtdetr/<name>/weights/best.pt가 생기고, 이걸
vision_node.py의 model_path_stage2 파라미터에 넣어서(RTDETR로 로드하도록 바꾼 뒤)
기존 YOLO stage2 모델과 tear 검출 정확도를 비교해보면 된다.
"""

import argparse
from pathlib import Path

from ultralytics import RTDETR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="convert_sdg_to_yolo.py/build_crop_dataset.py가 생성한 data.yaml 경로")
    # RT-DETR은 n/s/m 사이즈가 없고 l(large)/x(xlarge) 두 종류만 제공된다.
    # rtdetr-l이 기본값, 더 정교하게(느리더라도) 보고 싶으면 rtdetr-x로 바꿔서 시도.
    ap.add_argument("--model", default="rtdetr-l.pt", help="rtdetr-l.pt 또는 rtdetr-x.pt")
    ap.add_argument("--epochs", type=int, default=100)
    # stage2(tear crop) 추론이 vision_node.py에서 imgsz=320으로 도는 것과 맞춰서 기본값을
    # 320으로 둔다 (train_yolo.py의 960은 stage1용 원본 이미지 크기 기준이라 여기선 안 맞음).
    ap.add_argument("--imgsz", type=int, default=320, help="학습 해상도 (stage2 crop 추론 해상도와 맞춤)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--project", default="runs_shoe_rtdetr")
    # --name을 안 주면 데이터셋 폴더 이름으로 자동 설정 (train_yolo.py와 동일한 규칙).
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    if a.name is None:
        a.name = Path(a.data).resolve().parent.name.replace("dataset_shoe_", "")

    model = RTDETR(a.model)
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
