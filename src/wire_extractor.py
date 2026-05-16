from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .types import Point, WireGraph


class WireExtractor:
    """从原理图图片中提取导线网络。

    当前默认针对 KiCad/EasyEDA 风格的绿色/蓝色导线。核心思想：
    1. HSV 阈值提取绿色和蓝色导线；
    2. 形态学闭运算连接断点；
    3. 连通域编号，每个连通域近似为一个 net；
    4. HoughLinesP 提取线段用于可视化。
    """

    def __init__(self, min_area: int = 20):
        self.min_area = min_area

    def build_wire_mask(self, bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        # 绿色导线：KiCad / LTspice 等常见，Hue 约 35~95。
        # 适当降低 S/V 门限，兼容截图、缩放、压缩后的浅绿色导线。
        lower_green = np.array([35, 20, 35], dtype=np.uint8)
        upper_green = np.array([95, 255, 255], dtype=np.uint8)
        mask_green = cv2.inRange(hsv, lower_green, upper_green)

        # 蓝色导线：EasyEDA / Altium / 部分截图常见，Hue 约 90~135。
        # 这里只新增蓝色导线，不把黑色文字/元件纳入网络，避免产生大量伪网络。
        lower_blue = np.array([90, 20, 35], dtype=np.uint8)
        upper_blue = np.array([135, 255, 255], dtype=np.uint8)
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

        # 合并绿色和蓝色导线。
        mask_color = cv2.bitwise_or(mask_green, mask_blue)

        # 通过横/竖长线形态学开运算过滤文字，只保留原理图导线主体。
        # 蓝色图中有些导线较细，所以核不要太大。
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 1))
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 13))
        h_lines = cv2.morphologyEx(mask_color, cv2.MORPH_OPEN, kernel_h)
        v_lines = cv2.morphologyEx(mask_color, cv2.MORPH_OPEN, kernel_v)
        mask = cv2.bitwise_or(h_lines, v_lines)

        mask = cv2.medianBlur(mask, 3)
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)
        return mask

    def _erase_component_bboxes(
        self,
        mask: np.ndarray,
        exclude_bboxes: Optional[Sequence[Tuple[int, int, int, int]]] = None,
    ) -> np.ndarray:
        """从导线 mask 中扣掉元件框区域。

        蓝色原理图里，导线、二极管符号、电容符号等可能是同一种蓝色。
        如果不扣掉元件主体，二极管/电容等符号会把两侧导线连成同一个连通域，
        导致 A_net 和 K_net 被错误匹配为同一个 NET。

        注意：这里只轻微向内收缩 bbox 后擦除，避免把元件两端外侧的真实导线全部擦掉。
        """
        if not exclude_bboxes:
            return mask

        h, w = mask.shape[:2]
        cleaned = mask.copy()
        image_area = float(h * w)

        for bbox in exclude_bboxes:
            if bbox is None:
                continue

            x1, y1, x2, y2 = [int(round(v)) for v in bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            box_area = float((x2 - x1) * (y2 - y1))
            # 防止极少数超大误检框把大半张导线擦掉。
            if image_area > 0 and box_area / image_area > 0.20:
                continue

            # 向内收缩，保留 bbox 边界外侧的导线端点，方便后续 nearest_net_id 查找。
            inset = 2
            ex1 = max(0, x1 + inset)
            ey1 = max(0, y1 + inset)
            ex2 = min(w - 1, x2 - inset)
            ey2 = min(h - 1, y2 - inset)

            if ex2 > ex1 and ey2 > ey1:
                cleaned[ey1:ey2 + 1, ex1:ex2 + 1] = 0

        return cleaned

    def extract(
        self,
        image_path: str,
        exclude_bboxes: Optional[Sequence[Tuple[int, int, int, int]]] = None,
    ) -> WireGraph:
        bgr = cv2.imread(image_path)
        if bgr is None:
            raise FileNotFoundError(f"无法读取图片：{image_path}")

        mask = self.build_wire_mask(bgr)
        mask = self._erase_component_bboxes(mask, exclude_bboxes)

        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)

        # 删除太小的连通域，重新编号
        cleaned = np.zeros_like(mask)
        for i in range(1, num):
            if stats[i, cv2.CC_STAT_AREA] >= self.min_area:
                cleaned[labels == i] = 255
        num2, labels2, stats2, _ = cv2.connectedComponentsWithStats(cleaned, 8)

        # 提取线段用于显示
        edges = cv2.Canny(cleaned, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=25, minLineLength=20, maxLineGap=8)
        segments: List[Tuple[Point, Point]] = []
        if lines is not None:
            for l in lines[:, 0, :]:
                x1, y1, x2, y2 = [int(v) for v in l]
                segments.append(((x1, y1), (x2, y2)))
        return WireGraph(label_image=labels2, num_nets=max(0, num2 - 1), segments=segments)

    @staticmethod
    def nearest_net_id(label_image: np.ndarray, point: Point, radius: int = 25) -> Optional[int]:
        x, y = int(round(point[0])), int(round(point[1]))
        h, w = label_image.shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            return None
        if label_image[y, x] > 0:
            return int(label_image[y, x])
        x1, x2 = max(0, x - radius), min(w - 1, x + radius)
        y1, y2 = max(0, y - radius), min(h - 1, y + radius)
        crop = label_image[y1:y2 + 1, x1:x2 + 1]
        ys, xs = np.where(crop > 0)
        if len(xs) == 0:
            return None
        # 距离最近的非零连通域标签
        abs_xs = xs + x1
        abs_ys = ys + y1
        d2 = (abs_xs - x) ** 2 + (abs_ys - y) ** 2
        idx = int(np.argmin(d2))
        return int(label_image[abs_ys[idx], abs_xs[idx]])
