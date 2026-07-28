"""
2단계(캐스케이드) 전용 크롭 데이터셋 생성.

dataset_shoe_v5(convert_sdg_to_yolo.py 결과, shoe/tear 2클래스)에서, tear 자체가 아니라
**shoe(신발 페어) bbox 기준**으로 크롭한다. 실제 추론(vision_node.py/infer_cascade.py)이
"1단계에서 찾은 shoe 박스를 그대로 크롭해서 2단계에 넣는" 방식이라, 학습 크롭도 정확히
같은 기준(같은 pad_ratio)으로 만들어야 배율이 맞는다.

(이전 버전은 tear의 작은 bbox를 기준으로 딱 붙여 크롭해서, 학습 때는 tear가 크롭 안에
크게 보였는데 실제 추론 때는 shoe 전체를 크롭하니 tear가 상대적으로 작아져서 모델이 거의
못 잡는 문제가 있었다 — 학습/추론 배율 불일치.)

shoe 하나당 크롭 1장을 만들고, 그 크롭 범위 안에 중심이 들어오는 tear들만 라벨링한다.
tear가 없는 shoe 크롭은 그대로 negative 샘플이 된다 (실제 추론에서도 손상 없는 신발이
크롭돼 들어오는 경우가 있으니 별도로 랜덤 negative를 안 만들어도 자연스럽게 섞인다).

사용:
    python build_crop_dataset.py --src dataset_shoe_v5 --dst dataset_shoe_v5_crop
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

CLASS_NAMES = ["shoe", "tear"]
SHOE_ID, TEAR_ID = 0, 1
OUT_CLASS_NAMES = ["tear"]
CROP_OUT_SIZE = 480
# vision_node.py의 stage2_pad_ratio 기본값과 반드시 같아야 한다 (infer_cascade.py의
# --pad_ratio 기본값도 마찬가지). 여기서 값이 달라지면 다시 학습/추론 배율이 어긋난다.
STAGE2_PAD_RATIO = 0.3


def load_polygons(lbl_path, img_w, img_h):
    """[(cls_id, pts_abs(Nx2))] 반환. pts_abs는 원본 이미지 픽셀 좌표."""
    polys = []
    for line in Path(lbl_path).read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split()
        cls_id = int(parts[0])
        coords = np.array(list(map(float, parts[1:])), dtype=np.float32).reshape(-1, 2)
        coords[:, 0] *= img_w
        coords[:, 1] *= img_h
        polys.append((cls_id, coords))
    return polys


def crop_with_padding(x0, y0, x1, y1, img_w, img_h, pad_ratio):
    """vision_node.py의 _crop_with_padding / infer_cascade.py의 crop_with_padding과
    동일한 계산 — 박스 크기에 비례해서 패딩하고, 이미지 경계에서 잘라낸다(반대쪽으로
    밀어서 크기를 맞추지 않음 — 실제 추론 코드가 그렇게 안 하기 때문에 여기서도 안 한다)."""
    bw, bh = x1 - x0, y1 - y0
    pad_x, pad_y = bw * pad_ratio, bh * pad_ratio
    cx0 = max(0, int(x0 - pad_x))
    cy0 = max(0, int(y0 - pad_y))
    cx1 = min(img_w, int(x1 + pad_x))
    cy1 = min(img_h, int(y1 + pad_y))
    return cx0, cy0, cx1, cy1


def clip_polygon_to_box(pts, cx0, cy0, cx1, cy1):
    """polygon의 bbox가 크롭 박스와 겹치면, 각 점을 크롭 박스 안으로 clamp해서 반환.
    (정밀한 폴리곤 클리핑 대신 각 점을 clamp하는 근사 방식 — tear 패치는 원래
    작고 볼록한 모양이라 이 근사로도 학습용 라벨로는 충분하다.)"""
    bx0, by0 = pts[:, 0].min(), pts[:, 1].min()
    bx1, by1 = pts[:, 0].max(), pts[:, 1].max()
    if bx1 < cx0 or bx0 > cx1 or by1 < cy0 or by0 > cy1:
        return None
    clamped = pts.copy()
    clamped[:, 0] = np.clip(clamped[:, 0], cx0, cx1)
    clamped[:, 1] = np.clip(clamped[:, 1], cy0, cy1)
    return clamped


def process_split(src, dst, split):
    img_dir = src / "images" / split
    lbl_dir = src / "labels" / split
    out_img_dir = dst / "images" / split
    out_lbl_dir = dst / "labels" / split
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    n_pos = n_neg = 0

    for lbl_path in sorted(lbl_dir.glob("*.txt")):
        img_path = img_dir / f"{lbl_path.stem}.png"
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        polys = load_polygons(lbl_path, w, h)
        shoe_polys = [p for c, p in polys if c == SHOE_ID]
        tear_polys = [p for c, p in polys if c == TEAR_ID]

        for i, sp in enumerate(shoe_polys):
            x0, y0 = sp[:, 0].min(), sp[:, 1].min()
            x1, y1 = sp[:, 0].max(), sp[:, 1].max()
            cx0, cy0, cx1, cy1 = crop_with_padding(x0, y0, x1, y1, w, h, STAGE2_PAD_RATIO)
            crop = img[cy0:cy1, cx0:cx1]
            if crop.size == 0:
                continue
            crop_resized = cv2.resize(crop, (CROP_OUT_SIZE, CROP_OUT_SIZE))
            cw, ch = cx1 - cx0, cy1 - cy0

            lines = []
            for tp in tear_polys:
                tcx, tcy = tp[:, 0].mean(), tp[:, 1].mean()
                if not (cx0 <= tcx <= cx1 and cy0 <= tcy <= cy1):
                    continue  # 이 shoe 크롭에 안 속하는(다른 신발의) tear는 제외
                clipped = clip_polygon_to_box(tp, cx0, cy0, cx1, cy1)
                if clipped is None:
                    continue
                norm = clipped.copy()
                norm[:, 0] = (norm[:, 0] - cx0) / cw
                norm[:, 1] = (norm[:, 1] - cy0) / ch
                coords = " ".join(f"{v:.6f}" for v in norm.flatten())
                lines.append(f"0 {coords}")

            out_name = f"{lbl_path.stem}_s{i}"
            cv2.imwrite(str(out_img_dir / f"{out_name}.png"), crop_resized)
            (out_lbl_dir / f"{out_name}.txt").write_text("\n".join(lines))
            if lines:
                n_pos += 1
            else:
                n_neg += 1

    return n_pos, n_neg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="원본 YOLO 데이터셋 (dataset_shoe_v5 등)")
    ap.add_argument("--dst", required=True, help="크롭 데이터셋 출력 경로")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    src, dst = Path(a.src), Path(a.dst)
    if dst.exists():
        print(f"[error] {dst} 이미 존재함 — 다른 --dst를 쓰거나 지우고 다시 실행하세요.")
        return

    random.seed(a.seed)
    total_pos = total_neg = 0
    for split in ("train", "val"):
        n_pos, n_neg = process_split(src, dst, split)
        print(f"[{split}] tear 있는 shoe crop={n_pos} tear 없는 shoe crop={n_neg}")
        total_pos += n_pos
        total_neg += n_neg

    data_yaml = (
        f"path: {dst.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: {len(OUT_CLASS_NAMES)}\n"
        f"names: {OUT_CLASS_NAMES}\n"
    )
    (dst / "data.yaml").write_text(data_yaml)
    print(f"[done] total positive={total_pos} negative={total_neg} -> {dst/'data.yaml'}")


if __name__ == "__main__":
    main()
