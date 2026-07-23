"""
Isaac Sim Replicator(BasicWriter) 출력 -> YOLO-seg 학습셋 변환

전제 파일 구조 (isaac_shoe_sdg.py 기본 출력):
    <src>/rgb_XXXX.png
    <src>/instance_segmentation_XXXX.png
    <src>/instance_segmentation_mapping_XXXX.json

⚠ Isaac Sim 버전에 따라 mapping json의 키/값 구조가 다를 수 있음.
   먼저 소량(num_frames=2~3)으로 생성 후 아래처럼 json을 한번 찍어보고,
   `instance_label_from_mapping()` 파싱 부분만 실제 구조에 맞춰 조정하면 됨:
       python -c "import json; print(json.dumps(json.load(open('_out_shoe_sdg/instance_segmentation_mapping_0000.json')), indent=2))"

사용:
    pip install opencv-python-headless numpy
    python convert_sdg_to_yolo.py --src _out_shoe_sdg --dst dataset_shoe_v1 --val_ratio 0.15
"""

import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

CLASS_TO_ID = {"normal": 0, "defect": 1}


def find_frame_indices(src_dir):
    rgb_files = sorted(Path(src_dir).glob("rgb_*.png"))
    return [f.stem.split("_")[-1] for f in rgb_files]


def load_mapping(mapping_path):
    with open(mapping_path, "r") as f:
        return json.load(f)


def instance_label_from_mapping(mapping, key):
    """mapping json 항목에서 'class' 시맨틱 라벨 문자열을 뽑는다.
    Isaac Sim 버전에 따라 {"class": "defect"} 또는 {"class": ["defect"]} 형태일 수 있어 둘 다 처리."""
    entry = mapping.get(key)
    if entry is None:
        return None
    cls = entry.get("class")
    if isinstance(cls, list):
        cls = cls[0] if cls else None
    return cls


def mask_to_yolo_polygons(binary_mask, img_w, img_h, min_area=20):
    contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        pts = c.reshape(-1, 2).astype(np.float32)
        pts[:, 0] /= img_w
        pts[:, 1] /= img_h
        polygons.append(pts.flatten().tolist())
    return polygons


def process_frame(src_dir, idx, out_img_dir, out_lbl_dir):
    src_dir = Path(src_dir)
    rgb_path = src_dir / f"rgb_{idx}.png"
    seg_path = src_dir / f"instance_segmentation_{idx}.png"
    map_path = src_dir / f"instance_segmentation_mapping_{idx}.json"
    if not (rgb_path.exists() and seg_path.exists() and map_path.exists()):
        print(f"[skip] frame {idx}: 파일 누락 ({rgb_path.name}/{seg_path.name}/{map_path.name})")
        return False

    seg_img = cv2.imread(str(seg_path), cv2.IMREAD_UNCHANGED)
    if seg_img is None:
        print(f"[skip] frame {idx}: 세그멘테이션 이미지 로드 실패")
        return False
    h, w = seg_img.shape[:2]
    mapping = load_mapping(map_path)

    if seg_img.ndim == 2:
        unique_vals = np.unique(seg_img)
        get_key = lambda v: str(int(v))
        get_mask = lambda v: seg_img == v
    else:
        flat = seg_img.reshape(-1, seg_img.shape[-1])
        unique_vals = np.unique(flat, axis=0)
        get_key = lambda v: str(tuple(int(c) for c in v))
        get_mask = lambda v: np.all(seg_img == v, axis=-1)

    lines = []
    for val in unique_vals:
        key = get_key(val)
        label = instance_label_from_mapping(mapping, key)
        if label not in CLASS_TO_ID:
            continue  # 배경 / 미분류 인스턴스는 스킵
        mask = get_mask(val)
        for poly in mask_to_yolo_polygons(mask, w, h):
            coords = " ".join(f"{v:.6f}" for v in poly)
            lines.append(f"{CLASS_TO_ID[label]} {coords}")

    shutil.copy(rgb_path, out_img_dir / f"{idx}.png")
    with open(out_lbl_dir / f"{idx}.txt", "w") as f:
        f.write("\n".join(lines))
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    random.seed(a.seed)
    indices = find_frame_indices(a.src)
    if not indices:
        print(f"[error] {a.src} 에서 rgb_*.png 를 찾지 못함. 경로/생성 여부 확인 필요.")
        return
    random.shuffle(indices)
    n_val = max(1, int(len(indices) * a.val_ratio))
    val_idx = set(indices[:n_val])

    dst = Path(a.dst)
    for split in ("train", "val"):
        (dst / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst / "labels" / split).mkdir(parents=True, exist_ok=True)

    n_train_ok = n_val_ok = 0
    for idx in indices:
        split = "val" if idx in val_idx else "train"
        ok = process_frame(a.src, idx, dst / "images" / split, dst / "labels" / split)
        if ok:
            n_val_ok += split == "val"
            n_train_ok += split == "train"

    data_yaml = (
        f"path: {dst.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: {len(CLASS_TO_ID)}\n"
        f"names: {list(CLASS_TO_ID.keys())}\n"
    )
    with open(dst / "data.yaml", "w") as f:
        f.write(data_yaml)

    print(f"[done] train={n_train_ok} val={n_val_ok} -> {dst/'data.yaml'}")


if __name__ == "__main__":
    main()