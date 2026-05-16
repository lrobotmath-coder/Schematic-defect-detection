from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import yaml

from .detector import YOLOPoseDetector
from .ocr_reader import OCRReader
from .rules import RuleChecker
from .topology import assign_component_refs, assign_nets, build_net_names
from .types import Fault
from .visualize import draw_result
from .wire_extractor import WireExtractor


class CircuitPolarityPipeline:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)
        m = self.cfg.get("model", {})
        o = self.cfg.get("ocr", {})
        w = self.cfg.get("wire", {})
        r = self.cfg.get("rules", {})
        self.detector = YOLOPoseDetector(
            weights=m.get("pose_weights", "runs/pose/circuit_polarity_pose/weights/best.pt"),
            imgsz=int(m.get("imgsz", 960)),
            conf=float(m.get("conf", 0.25)),
        )
        self.ocr = OCRReader(backend=o.get("backend", "none"))
        self.wire_extractor = WireExtractor(min_area=int(w.get("min_area", 20)))
        self.pin_search_radius = int(w.get("pin_search_radius", 28))
        self.ocr_search_radius = int(w.get("ocr_search_radius", 55))
        self.rule_checker = RuleChecker(only_danger=bool(r.get("only_danger", False)))

    def run(self, image_path: str, out_dir: str = "results") -> Dict[str, object]:
        image_path = str(image_path)
        out_dir_p = Path(out_dir)
        out_dir_p.mkdir(parents=True, exist_ok=True)

        components = self.detector.predict(image_path)
        tokens = self.ocr.read(image_path)
        assign_component_refs(components, tokens)

        # 蓝色/绿色原理图中，元件符号可能和导线颜色一致。
        # 先用 YOLO 得到元件框，再在提取导线网络时扣掉这些元件主体区域，
        # 避免二极管/电容/MOS 符号把两端导线错误连成同一个 NET。
        exclude_bboxes = [c.bbox for c in components]

        wire_graph = self.wire_extractor.extract(image_path, exclude_bboxes=exclude_bboxes)
        assign_nets(components, wire_graph, pin_search_radius=self.pin_search_radius)
        build_net_names(wire_graph, tokens, components, ocr_search_radius=self.ocr_search_radius)
        faults: List[Fault] = self.rule_checker.check(components, wire_graph, tokens)

        out_img = out_dir_p / (Path(image_path).stem + "_result.png")
        draw_result(image_path, components, tokens, wire_graph, faults, str(out_img))

        warnings = []
        if self.detector.warning:
            warnings.append(self.detector.warning)
        if self.ocr.warning:
            warnings.append(self.ocr.warning)

        return {
            "image": image_path,
            "result_image": str(out_img),
            "components": [c.__dict__ for c in components],
            "ocr_tokens": [t.__dict__ for t in tokens],
            "net_names": wire_graph.net_names,
            "net_aliases": getattr(wire_graph, "net_aliases", {}),
            "diode_reports": getattr(self.rule_checker, "last_diode_reports", []),
            "faults": [f.__dict__ for f in faults],
            "warnings": warnings,
        }
