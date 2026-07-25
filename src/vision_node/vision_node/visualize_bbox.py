"""
convert_sdg_to_yolo.py가 만든 YOLO-seg 데이터셋(images/, labels/)에서 라벨 파일을 그대로
읽어 bbox를 그려 몇 장 저장해본다 — 학습에 실제로 들어가는 형태 그대로 확인하기 위한
용도 (Isaac이 준 bounding_box_2d_tight가 아니라, 변환된 최종 라벨 기준).

사용:
    python visualize_bbox.py --dataset dataset_shoe_v5 --n 6 --out bbox_check
"""

import argparse
import ast
import random
from pathlib import Path

import cv2
import numpy as np

DEFAULT_COLORS = [(0, 255, 0), (0, 0, 255), (255, 0, 255), (0, 255, 255), (255, 255, 0)]


def load_class_names(dataset_dir):
    """data.yaml의 names: [...] 줄에서 클래스 이름을 읽어온다 (하드코딩 대신)."""
    yaml_path = Path(dataset_dir) / "data.yaml"
    for line in yaml_path.read_text().splitlines():
        line = line.strip()
        if line.startswith("names:"):
            return ast.literal_eval(line.split(":", 1)[1].strip())
    raise ValueError(f"{yaml_path}에서 names:를 못 찾음")


def draw_one(img_path, lbl_path, out_path, class_names, class_colors, obb=False):
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    for line in Path(lbl_path).read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split()
        cls = class_names[int(parts[0])]
        coords = list(map(float, parts[1:]))
        pts = np.array(coords, dtype=np.float32).reshape(-1, 2)
        pts[:, 0] *= w
        pts[:, 1] *= h
        color = class_colors.get(cls, (0, 255, 255))
        if obb:
            # OBB(회전된 4점 폴리곤) 라벨은 min/max로 그리면(축 정렬 사각형) 회전된
            # 만큼 여백이 생겨서 실제로 안 맞는 것처럼 보인다 — 폴리곤 그대로 그려야 한다.
            cv2.polylines(img, [pts.astype(np.int32)], True, color, 2)
            x0, y0 = pts.min(axis=0)
        else:
            x0, y0 = pts.min(axis=0)
            x1, y1 = pts.max(axis=0)
            cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), color, 2)
        cv2.putText(img, cls, (int(x0), max(0, int(y0) - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.imwrite(str(out_path), img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="convert_sdg_to_yolo.py 출력 디렉터리 (dataset_shoe_v5 등)")
    ap.add_argument("--n", type=int, default=6, help="무작위로 뽑아 그릴 장 수")
    ap.add_argument("--out", default="bbox_check", help="결과 저장 디렉터리")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only_with_tear", action="store_true", help="tear 라벨이 있는 이미지만 대상으로 뽑기")
    ap.add_argument("--obb", action="store_true", help="OBB(회전 박스) 데이터셋용 — min/max 사각형 대신 폴리곤 그대로 그림")
    a = ap.parse_args()

    random.seed(a.seed)
    dataset = Path(a.dataset)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    class_names = load_class_names(dataset)
    class_colors = {name: DEFAULT_COLORS[i % len(DEFAULT_COLORS)] for i, name in enumerate(class_names)}
    tear_id = class_names.index("tear") if "tear" in class_names else None

    candidates = []
    for split in ("train", "val"):
        for lbl_path in (dataset / "labels" / split).glob("*.txt"):
            if a.only_with_tear and tear_id is not None:
                lines = lbl_path.read_text().splitlines()
                if not any(l.split()[0] == str(tear_id) for l in lines if l.strip()):
                    continue
            img_path = dataset / "images" / split / f"{lbl_path.stem}.png"
            if img_path.exists():
                candidates.append((img_path, lbl_path))

    if not candidates:
        print(f"[error] {dataset} 에서 라벨/이미지 쌍을 찾지 못함")
        return

    sample = random.sample(candidates, min(a.n, len(candidates)))
    for img_path, lbl_path in sample:
        out_path = out_dir / img_path.name
        draw_one(img_path, lbl_path, out_path, class_names, class_colors, obb=a.obb)
        print("saved", out_path)


if __name__ == "__main__":
    main()
