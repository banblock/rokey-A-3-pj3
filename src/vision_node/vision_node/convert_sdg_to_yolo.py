"""
Isaac Sim Replicator(BasicWriter) 출력 -> YOLO-seg 학습셋 변환

전제 파일 구조:
  - 카메라 1대만 쓸 때 (isaac_shoe_sdg.py 초기 버전):
        <src>/rgb_XXXX.png
        <src>/instance_segmentation_XXXX.png
        <src>/instance_segmentation_semantics_mapping_XXXX.json
  - 카메라 2대 이상 쓸 때 (Replicator가 render_product별로 서브폴더를 만듦):
        <src>/<cam_name>/rgb/rgb_XXXX.png
        <src>/<cam_name>/instance_segmentation/instance_segmentation_XXXX.png
        <src>/<cam_name>/instance_segmentation/instance_segmentation_semantics_mapping_XXXX.json
  둘 다 자동으로 인식해서 처리한다. 카메라별 서브폴더가 있으면 각각 순회하고,
  출력 파일명은 "<cam_name>_<idx>"로 저장해서 카메라끼리 섞여도 안 겹치게 한다.

  (참고: instance_segmentation_mapping_*.json은 인스턴스ID->prim 경로 문자열이고,
  클래스 라벨은 instance_segmentation_semantics_mapping_*.json에 {"class": ...}로 있음)

⚠ Isaac Sim 버전에 따라 mapping json의 키/값 구조가 다를 수 있음.
   먼저 소량(num_frames=2~3)으로 생성 후 아래처럼 json을 한번 찍어보고,
   `instance_label_from_mapping()` 파싱 부분만 실제 구조에 맞춰 조정하면 됨:
       python -c "import json; print(json.dumps(json.load(open('_out_shoe_sdg/D455_1/instance_segmentation/instance_segmentation_semantics_mapping_0000.json')), indent=2))"

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

CLASS_TO_ID = {"shoe": 0, "tear": 1}
# scratch도 tear와 같은 손상으로 취급해서 bbox 분류를 2클래스로 합친다.
LABEL_ALIAS = {"scratch": "tear"}


def find_camera_dirs(src_dir):
    """카메라별 서브폴더(<src>/<cam_name>/rgb/)가 있으면 그 이름들을 반환하고,
    없으면(카메라 1대짜리 평평한 구조) None을 반환한다."""
    src_dir = Path(src_dir)
    cam_dirs = [p.name for p in src_dir.iterdir() if p.is_dir() and (p / "rgb").is_dir()]
    return sorted(cam_dirs) if cam_dirs else None


def find_frame_indices(rgb_dir):
    rgb_files = sorted(Path(rgb_dir).glob("rgb_*.png"))
    return [f.stem.split("_")[-1] for f in rgb_files]


def load_mapping(mapping_path):
    with open(mapping_path, "r") as f:
        return json.load(f)


def instance_label_from_mapping(mapping, key):
    """mapping json 항목에서 'class' 시맨틱 라벨 문자열을 뽑는다.
    Isaac Sim 버전에 따라 {"class": "defect"} 또는 {"class": ["defect"]} 형태일 수 있어 둘 다 처리.

    tear/scratch 패치는 신발(조상 프림)의 "shoe" 라벨을 물려받은 채로 자기 라벨도
    가지고 있어서, mapping에는 "shoe,tear"처럼 콤마로 합쳐진 문자열로 나온다.
    이 경우 더 구체적인(조상이 아닌) 라벨을 우선한다."""
    entry = mapping.get(key)
    if entry is None:
        return None
    cls = entry.get("class")
    if isinstance(cls, list):
        cls = cls[0] if cls else None
    if isinstance(cls, str) and "," in cls:
        parts = [p.strip() for p in cls.split(",")]
        specific = [p for p in parts if p in ("tear", "scratch")]
        cls = specific[0] if specific else parts[0]
    return LABEL_ALIAS.get(cls, cls)


def mask_to_yolo_polygon(binary_mask, img_w, img_h, min_area=3):
    """인스턴스 하나(=신발 한 켤레, tear/scratch 패치 하나)의 마스크를 폴리곤 1개로 반환한다.

    한 인스턴스의 실루엣이 카메라 각도상 여러 조각(예: 왼발/오른발이 픽셀상 안 붙어있는
    경우)으로 나뉘어도, 조각별로 별도 bbox를 쓰면 같은 물체가 여러 개로 쪼개져 나온다
    (실제로 발견된 버그). 그래서 조각들을 모두 모아 convex hull 하나로 합쳐 인스턴스당
    폴리곤/bbox를 정확히 1개만 만든다. min_area도 20 -> 3으로 낮춰서, 많이 가려졌지만
    실제로 보이는 작은 tear까지 살아남게 한다(완전히 사라진 노이즈 수준만 제외)."""
    contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pts_all = [c.reshape(-1, 2) for c in contours if cv2.contourArea(c) >= min_area]
    if not pts_all:
        return None
    merged = np.concatenate(pts_all, axis=0).astype(np.int32)
    hull = cv2.convexHull(merged).reshape(-1, 2).astype(np.float32)
    hull[:, 0] /= img_w
    hull[:, 1] /= img_h
    return hull.flatten().tolist()


def process_frame(rgb_dir, seg_dir, idx, out_name, out_img_dir, out_lbl_dir):
    rgb_path = Path(rgb_dir) / f"rgb_{idx}.png"
    seg_path = Path(seg_dir) / f"instance_segmentation_{idx}.png"
    map_path = Path(seg_dir) / f"instance_segmentation_semantics_mapping_{idx}.json"
    if not (rgb_path.exists() and seg_path.exists() and map_path.exists()):
        print(f"[skip] {out_name}: 파일 누락 ({rgb_path.name}/{seg_path.name}/{map_path.name})")
        return False

    seg_img = cv2.imread(str(seg_path), cv2.IMREAD_UNCHANGED)
    if seg_img is None:
        print(f"[skip] {out_name}: 세그멘테이션 이미지 로드 실패")
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
        poly = mask_to_yolo_polygon(mask, w, h)
        if poly is None:
            continue
        coords = " ".join(f"{v:.6f}" for v in poly)
        lines.append(f"{CLASS_TO_ID[label]} {coords}")

    shutil.copy(rgb_path, out_img_dir / f"{out_name}.png")
    with open(out_lbl_dir / f"{out_name}.txt", "w") as f:
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

    # 카메라별 서브폴더가 있는지 먼저 확인. 없으면 카메라 1대짜리 평평한 구조로 처리.
    cam_names = find_camera_dirs(a.src)
    if cam_names:
        print(f"[info] 카메라 {len(cam_names)}개 감지: {cam_names}")
        entries = []  # (out_name, rgb_dir, seg_dir, idx)
        for cam in cam_names:
            rgb_dir = Path(a.src) / cam / "rgb"
            seg_dir = Path(a.src) / cam / "instance_segmentation"
            for idx in find_frame_indices(rgb_dir):
                entries.append((f"{cam}_{idx}", rgb_dir, seg_dir, idx))
    else:
        rgb_dir = seg_dir = Path(a.src)
        entries = [(idx, rgb_dir, seg_dir, idx) for idx in find_frame_indices(rgb_dir)]

    if not entries:
        print(f"[error] {a.src} 에서 rgb_*.png 를 찾지 못함. 경로/생성 여부 확인 필요.")
        return

    random.shuffle(entries)
    n_val = max(1, int(len(entries) * a.val_ratio))
    val_entries = set(e[0] for e in entries[:n_val])

    dst = Path(a.dst)
    for split in ("train", "val"):
        (dst / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst / "labels" / split).mkdir(parents=True, exist_ok=True)

    n_train_ok = n_val_ok = 0
    for out_name, rgb_dir, seg_dir, idx in entries:
        split = "val" if out_name in val_entries else "train"
        ok = process_frame(rgb_dir, seg_dir, idx, out_name, dst / "images" / split, dst / "labels" / split)
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
