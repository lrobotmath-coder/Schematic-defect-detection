from __future__ import annotations

from pathlib import Path
from typing import List

import cv2
import numpy as np

from .topology import net_name
from .types import Component, Fault, OCRToken, WireGraph


def draw_result(
    image_path: str,
    components: List[Component],
    tokens: List[OCRToken],
    wire_graph: WireGraph,
    faults: List[Fault],
    out_path: str,
) -> str:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)

    # 导线线段可视化
    for (x1, y1), (x2, y2) in wire_graph.segments[:500]:
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 180, 0), 1)

    # 元件框与关键点
    for c in components:
        x1, y1, x2, y2 = c.bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 120, 0), 2)
        label = f"{c.ref or c.name} {c.conf:.2f}"
        cv2.putText(img, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 120, 0), 1, cv2.LINE_AA)
        for kp_name, color in [("anode", (0, 0, 255)), ("cathode", (255, 0, 0))]:
            pt = c.keypoints.get(kp_name)
            if pt is None:
                continue
            px, py = int(round(pt[0])), int(round(pt[1]))
            cv2.circle(img, (px, py), 5, color, -1)
            nid = c.nets.get(kp_name)
            txt = "A" if kp_name == "anode" else "K"
            txt += f":{net_name(nid, wire_graph.net_names)}"
            cv2.putText(img, txt, (px + 6, py - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    # OCR文字框
    for t in tokens:
        pts = np.array([[int(x), int(y)] for x, y in t.bbox], dtype=np.int32)
        cv2.polylines(img, [pts], True, (180, 0, 180), 1)

    # 故障框最后画，最醒目
    for f in faults:
        x1, y1, x2, y2 = f.bbox
        color = (0, 0, 255) if f.level == "danger" else (0, 165, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.putText(img, f.fault_type, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out_path, img)
    return out_path
