from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

Point = Tuple[float, float]
BBox = Tuple[int, int, int, int]


@dataclass
class Component:
    cls_id: int
    name: str
    bbox: BBox
    conf: float = 1.0
    keypoints: Dict[str, Optional[Point]] = field(default_factory=dict)
    nets: Dict[str, Optional[int]] = field(default_factory=dict)
    ref: Optional[str] = None

    @property
    def center(self) -> Point:
        x1, y1, x2, y2 = self.bbox
        return (float((x1 + x2) / 2), float((y1 + y2) / 2))

    def has_polarity(self) -> bool:
        return self.keypoints.get("anode") is not None and self.keypoints.get("cathode") is not None


@dataclass
class OCRToken:
    text: str
    bbox: List[Point]
    conf: float = 0.0
    # net_label / refdes / value / text / noise。旧代码不使用该字段也不会受影响。
    kind: str = "text"

    @property
    def center(self) -> Point:
        xs = [p[0] for p in self.bbox]
        ys = [p[1] for p in self.bbox]
        return (sum(xs) / len(xs), sum(ys) / len(ys))


@dataclass
class WireGraph:
    label_image: object
    num_nets: int
    # 每个网络最终用于显示的主名称，例如 +5V / GND / NET_12
    net_names: Dict[int, str] = field(default_factory=dict)
    # 每个网络绑定到的全部 OCR/符号别名，例如同一根线上同时有 ACL 和 +5V。
    # 场景判断时不要只看 net_names，应该同时看 net_aliases。
    net_aliases: Dict[int, List[str]] = field(default_factory=dict)
    # 可选：如果后续做网络合并，可在这里记录 old_net -> root_net。
    net_roots: Dict[int, int] = field(default_factory=dict)
    segments: List[Tuple[Point, Point]] = field(default_factory=list)


@dataclass
class Fault:
    level: str
    fault_type: str
    target: str
    message: str
    bbox: BBox
    evidence: Dict[str, object] = field(default_factory=dict)
