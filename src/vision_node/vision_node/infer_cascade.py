"""
2단계 캐스케이드 추론: 원본 이미지에서 shoe(1단계 모델, 검출률 매우 높음)를 먼저 찾고,
검출된 shoe 박스마다 무조건 크롭+확대해서 tear 전용 2단계 모델(build_crop_dataset.py로
만든 크롭 데이터셋으로 학습한 모델)을 돌려 tear를 다시 찾는다.

예전 버전은 "1단계에서 tear를 컨피던스 낮게라도 찾아야" 크롭할 위치가 나왔는데,
원본에서 tear가 너무 작으면 1단계가 후보 자체를 아예 못 낼 수 있어서 2단계가 발동을
안 하는 문제가 있었다. shoe는 거의 항상 정확히 잡히니(mAP50 0.995), shoe 박스를
기준으로 무조건 크롭하는 쪽이 훨씬 안정적이다.

전제:
    - 1단계 가중치: dataset_shoe_v5(원본 해상도, shoe+tear 2클래스)로 학습한 모델
    - 2단계 가중치: dataset_shoe_v5_crop(build_crop_dataset.py 결과, tear 1클래스)로 학습한 모델

사용:
    python infer_cascade.py \\
        --weights1 runs_shoe_seg/v2-6/weights/best.pt \\
        --weights2 runs_shoe_seg/v5_crop/weights/best.pt \\
        --source some.png --out_dir cascade_out
"""

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

SHOE_CLASS_NAME = "shoe"
TEAR_CLASS_NAME = "tear"


def detect(model, img, imgsz, class_filter=None):
    result = model.predict(img, imgsz=imgsz, verbose=False)[0]
    dets = []
    for b in result.boxes:
        cls_name = model.names[int(b.cls.item())]
        if class_filter is not None and cls_name != class_filter:
            continue
        conf = float(b.conf.item())
        x0, y0, x1, y1 = [float(v) for v in b.xyxy[0].tolist()]
        dets.append({"cls": cls_name, "conf": conf, "box": (x0, y0, x1, y1)})
    return dets


def crop_with_padding(img, box, pad_ratio):
    h, w = img.shape[:2]
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    pad_x, pad_y = bw * pad_ratio, bh * pad_ratio
    cx0 = max(0, int(x0 - pad_x))
    cy0 = max(0, int(y0 - pad_y))
    cx1 = min(w, int(x1 + pad_x))
    cy1 = min(h, int(y1 + pad_y))
    return img[cy0:cy1, cx0:cx1], (cx0, cy0)


def find_tears_in_shoe_crops(model2, img, shoe_boxes, pad_ratio, imgsz2, min_conf2):
    tears = []
    for shoe in shoe_boxes:
        crop, (ox, oy) = crop_with_padding(img, shoe["box"], pad_ratio)
        if crop.size == 0:
            continue
        dets = detect(model2, crop, imgsz2, class_filter=TEAR_CLASS_NAME)
        for d in dets:
            if d["conf"] < min_conf2:
                continue
            cx0, cy0, cx1, cy1 = d["box"]
            tears.append({
                "cls": TEAR_CLASS_NAME,
                "conf": d["conf"],
                "box": (cx0 + ox, cy0 + oy, cx1 + ox, cy1 + oy),
                "stage": 2,
                "parent_shoe": shoe["box"],
            })
    return tears


def draw_detections(img, shoe_boxes, tear_boxes):
    out = img.copy()
    for det in shoe_boxes:
        x0, y0, x1, y1 = [int(v) for v in det["box"]]
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 255, 0), 1)
        cv2.putText(out, f"shoe {det['conf']:.2f}", (x0, max(0, y0 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    for det in tear_boxes:
        x0, y0, x1, y1 = [int(v) for v in det["box"]]
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 0, 255), 2)
        cv2.putText(out, f"tear {det['conf']:.2f}", (x0, max(0, y0 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return out


def process_image(model1, model2, img_path, out_dir, imgsz1, imgsz2, pad_ratio, min_conf2):
    img = cv2.imread(str(img_path))
    shoe_boxes = detect(model1, img, imgsz1, class_filter=SHOE_CLASS_NAME)
    tear_boxes = find_tears_in_shoe_crops(model2, img, shoe_boxes, pad_ratio, imgsz2, min_conf2)

    print(f"[{img_path.name}] shoe={len(shoe_boxes)} tear(2단계)={len(tear_boxes)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / img_path.name), draw_detections(img, shoe_boxes, tear_boxes))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights1", required=True, help="1단계(shoe) 모델 가중치 — dataset_shoe_v5로 학습")
    ap.add_argument("--weights2", required=True, help="2단계(tear 전용) 모델 가중치 — dataset_shoe_v5_crop으로 학습")
    ap.add_argument("--source", required=True, help="이미지 파일 또는 이미지가 든 디렉터리")
    ap.add_argument("--out_dir", default="cascade_out")
    ap.add_argument("--imgsz1", type=int, default=960, help="1단계(전체 이미지) 추론 해상도")
    ap.add_argument("--imgsz2", type=int, default=320, help="2단계(크롭) 추론 해상도 — build_crop_dataset.py의 CROP_OUT_SIZE와 맞춤")
    ap.add_argument("--pad_ratio", type=float, default=0.3, help="shoe 박스 크롭 시 여유 비율 — build_crop_dataset.py의 STAGE2_PAD_RATIO와 반드시 같아야 학습/추론 배율이 맞는다")
    ap.add_argument("--min_conf2", type=float, default=0.25, help="2단계 tear 검출 최소 컨피던스")
    a = ap.parse_args()

    model1 = YOLO(a.weights1)
    model2 = YOLO(a.weights2)
    source = Path(a.source)
    out_dir = Path(a.out_dir)

    img_paths = [source] if source.is_file() else sorted(source.glob("*.png")) + sorted(source.glob("*.jpg"))
    for p in img_paths:
        process_image(model1, model2, p, out_dir, a.imgsz1, a.imgsz2, a.pad_ratio, a.min_conf2)

    print(f"[done] {len(img_paths)}장 처리 -> {out_dir}/")


if __name__ == "__main__":
    main()
