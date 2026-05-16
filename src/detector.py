from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from .types import Component


# 需要保留 A/K 关键点的极性器件类别
DIODE_CLASSES = {"diode", "zener_diode", "schottky_diode", "led"}

# 新增的非二极管类别：用于识别和显示，但不强制要求 A/K 关键点
NON_POLARITY_CLASSES = {
    "power",
    "gnd",
    "rectifier_bridge",
    "npn_transistor",
    "pnp_transistor",
    "nmos",
    "pmos",
    "inductor",
    "capacitor",
    "electrolytic_capacitor",
}


class YOLOPoseDetector:
    """YOLOv8-Pose 元器件与 A/K 关键点检测封装。

    修改重点：
    1. 不再强制 device=0，而是 CUDA 可用时用 GPU，否则自动回退 CPU。
    2. 增加异常大框过滤，避免电感/电容误框整张图。
    3. 增加类别置信度过滤，二极管类允许较低阈值，非极性元件要求更高阈值。
    4. 增加重复框过滤，减少同一区域重复检测。

    注意：Q1/NMOS 被误识别成二极管时，仅靠 detector.py 不能完全过滤，
    因为 ref=Q1 通常是在 topology/OCR 阶段才绑定的。
    这类误检应在 rules.py 中结合 OCR 文字继续过滤。
    """

    def __init__(
        self,
        weights: str,
        imgsz: int = 960,
        conf: float = 0.25,
        device: Optional[Union[str, int]] = None,
        max_box_area_ratio: float = 0.25,
        duplicate_iou: float = 0.70,
    ):
        self.weights = Path(weights)
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.device = device
        self.max_box_area_ratio = float(max_box_area_ratio)
        self.duplicate_iou = float(duplicate_iou)

        self.model = None
        self.available = False
        self.warning = ""
        self.names: Dict[int, str] = {}

        self._load_model()

    def _load_model(self) -> None:
        if not self.weights.exists():
            self.warning = f"未找到 YOLO-Pose 权重：{self.weights}。请先运行 train_pose.py 训练。"
            return

        try:
            from ultralytics import YOLO

            self.model = YOLO(str(self.weights))
            self.names = dict(self.model.names)
            self.available = True
        except Exception as exc:  # pragma: no cover
            self.warning = f"YOLO 模型加载失败：{exc}"

    def _select_device(self) -> Union[str, int]:
        """选择推理设备。

        如果用户指定了 device=0 但当前 PyTorch 看不到 CUDA，自动回退 CPU，避免界面报错。
        """
        if self.device is None or str(self.device).lower() in {"auto", ""}:
            try:
                import torch

                return 0 if torch.cuda.is_available() else "cpu"
            except Exception:
                return "cpu"

        dev = self.device
        dev_str = str(dev).lower()

        if dev_str in {"0", "cuda", "cuda:0"}:
            try:
                import torch

                if not torch.cuda.is_available():
                    self.warning = "当前 PyTorch 未检测到 CUDA，已自动切换为 CPU 推理。"
                    return "cpu"
            except Exception:
                self.warning = "CUDA 检测失败，已自动切换为 CPU 推理。"
                return "cpu"

            return 0

        if dev_str == "cpu":
            return "cpu"

        return dev

    @staticmethod
    def _box_area(bbox: Tuple[int, int, int, int]) -> int:
        x1, y1, x2, y2 = bbox
        return max(0, x2 - x1) * max(0, y2 - y1)

    @staticmethod
    def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b

        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)

        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0

        area_a = YOLOPoseDetector._box_area(a)
        area_b = YOLOPoseDetector._box_area(b)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def _class_min_conf(self, name: str) -> float:
        """按类别设置最低置信度。

        二极管类元件在完整原理图中通常较小，先允许低阈值，避免漏检 D1。
        新增的 MOS/电感/电容类如果置信度过低，很容易产生大框误检，因此阈值稍高。
        """
        name = name.lower()

        if name in DIODE_CLASSES:
            return min(self.conf, 0.05)

        if name in {"inductor", "capacitor", "electrolytic_capacitor"}:
            return max(self.conf, 0.20)

        if name in {"nmos", "pmos", "npn_transistor", "pnp_transistor"}:
            return max(self.conf, 0.15)

        return self.conf

    def _should_keep_detection(
        self,
        name: str,
        conf: float,
        bbox: Tuple[int, int, int, int],
        img_area: int,
    ) -> bool:
        if conf < self._class_min_conf(name):
            return False

        box_area = self._box_area(bbox)
        if box_area <= 4:
            return False

        # 过滤覆盖大半张图的异常框。完整原理图中的单个元件通常不会超过整图 25%。
        if img_area > 0 and (box_area / img_area) > self.max_box_area_ratio:
            return False

        return True

    def _remove_duplicates(self, comps: List[Component]) -> List[Component]:
        """去掉同类别高度重叠的重复框，保留置信度高的。"""
        kept: List[Component] = []

        for comp in sorted(comps, key=lambda c: c.conf, reverse=True):
            duplicate = False
            for old in kept:
                if comp.name == old.name and self._iou(comp.bbox, old.bbox) > self.duplicate_iou:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(comp)

        # 恢复大致从左到右、从上到下的顺序，方便后续阅读日志
        kept.sort(key=lambda c: (c.bbox[1], c.bbox[0]))
        return kept

    def predict(self, image_path: str) -> List[Component]:
        if not self.available or self.model is None:
            return []

        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"无法读取图片：{image_path}")

        h, w = img.shape[:2]
        img_area = h * w
        device = self._select_device()

        results = self.model.predict(
            source=image_path,
            imgsz=self.imgsz,
            conf=self.conf,
            device=device,
            verbose=False,
        )

        components: List[Component] = []
        if not results:
            return components

        r = results[0]
        if r.boxes is None:
            return components

        boxes = r.boxes.xyxy.cpu().numpy()
        clss = r.boxes.cls.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy() if r.boxes.conf is not None else np.ones(len(boxes))

        kpts = None
        if getattr(r, "keypoints", None) is not None and r.keypoints is not None:
            try:
                kpts = r.keypoints.data.cpu().numpy()  # [n, 2, 3]
            except Exception:
                kpts = None

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.astype(int).tolist()
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            cls_id = int(clss[i])
            name = str(self.names.get(cls_id, str(cls_id))).lower()
            conf_value = float(confs[i])
            bbox = (x1, y1, x2, y2)

            if not self._should_keep_detection(name, conf_value, bbox, img_area):
                continue

            kp_dict = {"anode": None, "cathode": None}

            # 只有二极管类才保留 A/K 关键点。非极性器件不参与极性判断。
            if name in DIODE_CLASSES and kpts is not None and i < len(kpts) and kpts.shape[1] >= 2:
                # 关键点顺序：0 anode, 1 cathode，visibility/conf > 0.1 才认为有效
                for label, idx in [("anode", 0), ("cathode", 1)]:
                    try:
                        x, y, v = kpts[i, idx].tolist()
                    except Exception:
                        continue

                    if v > 0.1 and 0 <= x <= w and 0 <= y <= h:
                        kp_dict[label] = (float(x), float(y))

            components.append(
                Component(
                    cls_id=cls_id,
                    name=name,
                    bbox=bbox,
                    conf=conf_value,
                    keypoints=kp_dict,
                )
            )

        return self._remove_duplicates(components)
