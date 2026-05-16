from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import List, Tuple

from src.pipeline import CircuitPolarityPipeline

BBox = Tuple[int, int, int, int]


def iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    aa = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    bb = max(0, bx2 - bx1) * max(0, by2 - by1)
    return inter / (aa + bb - inter + 1e-6)


def load_gt(annotation_path: Path) -> List[BBox]:
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    boxes = []
    for f in data.get("faults", []):
        boxes.append(tuple(map(int, f.get("fault_bbox", []))))
    return boxes


def main() -> None:
    root = Path("data/circuit_polarity_fault_testset")
    image_dir = root / "images/fault"
    ann_dir = root / "annotations/fault"
    pipe = CircuitPolarityPipeline("config.yaml")
    out_dir = Path("results/batch_fault_test")
    rows = []
    for image_path in sorted(image_dir.glob("*.png")):
        res = pipe.run(str(image_path), str(out_dir))
        pred_boxes = [tuple(map(int, f["bbox"])) for f in res["faults"] if f.get("level") == "danger"]
        ann_path = ann_dir / f"{image_path.stem}.json"
        gt_boxes = load_gt(ann_path) if ann_path.exists() else []
        best = 0.0
        for pb in pred_boxes:
            for gb in gt_boxes:
                best = max(best, iou(pb, gb))
        rows.append({
            "image": image_path.name,
            "gt_count": len(gt_boxes),
            "pred_count": len(pred_boxes),
            "best_iou": round(best, 4),
            "result_image": res["result_image"],
        })
    csv_path = out_dir / "evaluation.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["image"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"批量测试完成：{csv_path}")


if __name__ == "__main__":
    main()
