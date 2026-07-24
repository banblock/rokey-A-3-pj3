"""
2단계 캐스케이드 추론: 신발+tear를 전체 이미지에서 한 번 검출한 뒤,
tear 컨피던스가 낮게 나온 후보 주변만 크롭해서 같은 모델로 다시(확대) 검출해
작은 tear를 더 높은 신뢰도로 잡아본다.

전제: train_yolo.py로 학습한 shoe/tear 2클래스 모델 가중치(best.pt)가 있어야 함.

사용:
    python infer_cascade.py --weights runs_shoe_seg/v2/weights/best.pt --source some.png
    python infer_cascade.py --weights .../best.pt --source dir_of_images/ --out_dir cascade_out
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

TEAR_CLASS_NAME = "tear"


def run_stage1(model, img, imgsz):
    result = model.predict(img, imgsz=imgsz, verbose=False)[0]
    boxes = []
    for b in result.boxes:
        cls_id = int(b.cls.item())
        conf = float(b.conf.item())
        x0, y0, x1, y1 = [float(v) for v in b.xyxy[0].tolist()]
        cls_name = model.names[cls_id]
        boxes.append({"cls": cls_name, "conf": conf, "box": (x0, y0, x1, y1), "stage": 1})
    return boxes


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


def refine_low_conf_tears(model, img, stage1_boxes, low_conf, pad_ratio, imgsz2):
    refined = []
    for det in stage1_boxes:
        if det["cls"] != TEAR_CLASS_NAME or det["conf"] >= low_conf:
            continue
        crop, (ox, oy) = crop_with_padding(img, det["box"], pad_ratio)
        if crop.size == 0:
            continue
        result = model.predict(crop, imgsz=imgsz2, verbose=False)[0]
        best = None
        for b in result.boxes:
            cls_id = int(b.cls.item())
            if model.names[cls_id] != TEAR_CLASS_NAME:
                continue
            conf = float(b.conf.item())
            if best is None or conf > best["conf"]:
                cx0, cy0, cx1, cy1 = [float(v) for v in b.xyxy[0].tolist()]
                best = {"cls": TEAR_CLASS_NAME, "conf": conf,
                        "box": (cx0 + ox, cy0 + oy, cx1 + ox, cy1 + oy), "stage": 2}
        if best is not None and best["conf"] > det["conf"]:
            refined.append({"replace": det, "with": best})
    return refined


def draw_detections(img, boxes):
    out = img.copy()
    colors = {"shoe": (0, 255, 0), TEAR_CLASS_NAME: (0, 0, 255)}
    for det in boxes:
        x0, y0, x1, y1 = [int(v) for v in det["box"]]
        color = colors.get(det["cls"], (0, 255, 255))
        thickness = 3 if det["stage"] == 2 else 1
        cv2.rectangle(out, (x0, y0), (x1, y1), color, thickness)
        label = f"{det['cls']} {det['conf']:.2f}" + ("*" if det["stage"] == 2 else "")
        cv2.putText(out, label, (x0, max(0, y0 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return out


def process_image(model, img_path, out_dir, imgsz1, imgsz2, low_conf, pad_ratio):
    img = cv2.imread(str(img_path))
    stage1_boxes = run_stage1(model, img, imgsz1)
    refinements = refine_low_conf_tears(model, img, stage1_boxes, low_conf, pad_ratio, imgsz2)

    final_boxes = list(stage1_boxes)
    for r in refinements:
        final_boxes.remove(r["replace"])
        final_boxes.append(r["with"])
        print(f"[{img_path.name}] tear conf {r['replace']['conf']:.2f} -> {r['with']['conf']:.2f} (crop 재검출)")

    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / img_path.name), draw_detections(img, final_boxes))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="학습된 shoe/tear 모델 가중치(best.pt)")
    ap.add_argument("--source", required=True, help="이미지 파일 또는 이미지가 든 디렉터리")
    ap.add_argument("--out_dir", default="cascade_out")
    ap.add_argument("--imgsz1", type=int, default=960, help="1단계(전체 이미지) 추론 해상도")
    ap.add_argument("--imgsz2", type=int, default=640, help="2단계(크롭) 추론 해상도")
    ap.add_argument("--low_conf", type=float, default=0.5, help="이 값보다 낮은 tear conf만 크롭 재검출")
    ap.add_argument("--pad_ratio", type=float, default=1.5, help="크롭 시 박스 크기 대비 여유 비율")
    a = ap.parse_args()

    model = YOLO(a.weights)
    source = Path(a.source)
    out_dir = Path(a.out_dir)

    img_paths = [source] if source.is_file() else sorted(source.glob("*.png")) + sorted(source.glob("*.jpg"))
    for p in img_paths:
        process_image(model, p, out_dir, a.imgsz1, a.imgsz2, a.low_conf, a.pad_ratio)

    print(f"[done] {len(img_paths)}장 처리 -> {out_dir}/")


if __name__ == "__main__":
    main()
