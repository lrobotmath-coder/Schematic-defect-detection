from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

from .topology import is_ground_name, is_power_name, net_name
from .ocr_reader import get_last_image_path
from .types import Component, Fault, OCRToken, WireGraph


DIODE_CLASSES = {"diode", "zener_diode", "schottky_diode", "led"}

# 可按 bbox 两端推断 pin1/pin2 的两端元件。
# topology.py 会为这些类别写入 comp.nets["pin1"] / comp.nets["pin2"]。
TWO_PIN_COMPONENT_CLASSES = {
    "resistor",
    "capacitor",
    "electrolytic_capacitor",
    "inductor",
    # OCR 兜底生成的两端对象：P6/MOTOR/load/connector。
    "load",
    "connector",
    "terminal",
    "motor",
}

# Boost/Buck/开关电源中常见开关节点名
BOOST_SWITCH_NETS = {
    "SOUT", "SW", "LX", "PHASE", "DRAIN", "SWITCH", "SW_NODE", "NET_SW",
    "L_OUT", "INDUCTOR_OUT",
}

# 输出/负载端常见命名
OUTPUT_NETS = {
    "OUT", "VOUT", "+OUT", "OUTPUT", "LOAD", "LOAD+", "VO", "VLED", "LED+",
}

# 输入/正电源端常见命名
INPUT_POWER_NETS = {
    "VIN", "VCC", "VDD", "VBAT", "BAT", "+BAT", "+VIN", "VBUS",
    "5V", "+5V", "3V3", "3.3V", "+3.3V", "12V", "+12V", "24V", "+24V",
    "PWR", "PWR_FLAG",
}

# 低电位/地端命名
GROUND_NETS = {"GND", "0", "0V", "VSS", "PGND", "AGND", "DGND", "SGND"}

# 元件编号/文字，用于过滤误检与场景判断
TRANSISTOR_TEXTS = {
    "Q1", "Q2", "Q3", "Q4", "Q5", "NMOS", "PMOS", "MOS", "MOSFET", "NPN", "PNP", "BJT", "TRANSISTOR",
}

RESISTOR_TEXT_PAT = re.compile(r"^(R\d+|\d+(\.\d+)?[KMR]?(Ω|OHM)?|\d+(\.\d+)?K|\d+(\.\d+)?R)$", re.I)
CAP_TEXT_PAT = re.compile(r"^(C\d+|\d+(\.\d+)?(UF|NF|PF|U|N|P))$", re.I)
INDUCTOR_TEXT_PAT = re.compile(r"^(L\d+|\d+(\.\d+)?(UH|MH|H)|COIL|MOTOR|RELAY|K\d+|INDUCTOR)$", re.I)
REFDES_PAT = re.compile(r"^(R|C|L|Q|U|V|J|P)\d+$", re.I)
DIODE_REF_PAT = re.compile(r"^(D\d+|LED\d*|DZ\d*)$", re.I)


Point = Tuple[float, float]
BBox = Tuple[int, int, int, int]


def _expand_bbox(bbox: BBox, pad: int = 12) -> BBox:
    x1, y1, x2, y2 = bbox
    return (max(0, x1 - pad), max(0, y1 - pad), x2 + pad, y2 + pad)


def _class_cn(name: str) -> str:
    return {
        "diode": "普通二极管",
        "zener_diode": "稳压/齐纳二极管",
        "schottky_diode": "肖特基二极管",
        "led": "LED",
    }.get(name, name)


def _clean_name(name: Optional[str]) -> str:
    if name is None:
        return ""

    s = str(name).strip().replace(" ", "").upper()
    s = s.replace("＋", "+")
    s = s.replace("（", "(").replace("）", ")")
    s = s.replace("OV", "0V")

    # 检测列表中常用 "-" 表示没有编号；规则里必须当成空，
    # 否则会阻止后续的假阳性过滤和兼容编号推断。
    if s in {"-", "--", "NONE", "NULL", "未连接"}:
        return ""

    # VBUS(+5V)、VBUS (+5V) 这种 OCR 结果统一为 VBUS，便于判断电源
    if "VBUS" in s:
        return "VBUS"

    return s


def _is_unknown_net(name: Optional[str]) -> bool:
    name = _clean_name(name)
    return not name or name.startswith("NET_") or name in {"NONE", "NULL", "未连接"}


def _is_boost_switch_net(name: Optional[str]) -> bool:
    return _clean_name(name) in BOOST_SWITCH_NETS


def _is_output_net(name: Optional[str]) -> bool:
    return _clean_name(name) in OUTPUT_NETS


def _is_input_power_net(name: Optional[str]) -> bool:
    name = _clean_name(name)
    return name in INPUT_POWER_NETS or is_power_name(name)


def _is_low_or_ground_net(name: Optional[str]) -> bool:
    name = _clean_name(name)
    return name in GROUND_NETS or is_ground_name(name)


def _is_signal_or_protected_net(name: Optional[str]) -> bool:
    name = _clean_name(name)
    return _is_input_power_net(name) or _is_output_net(name) or _is_boost_switch_net(name) or name in {
        "IN", "EN", "FB", "RESET", "RST", "IO", "GPIO", "SIGNAL", "BUS", "SDA", "SCL", "TX", "RX"
    }


def _dist(p1: Point, p2: Point) -> float:
    return math.hypot(float(p1[0]) - float(p2[0]), float(p1[1]) - float(p2[1]))


def _bbox_center(bbox: BBox) -> Point:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _safe_center(obj) -> Point:
    try:
        return obj.center
    except Exception:
        return _bbox_center(obj.bbox)


def _token_text(token: OCRToken) -> str:
    return _clean_name(getattr(token, "text", ""))


def _token_in_bbox(token: OCRToken, bbox: BBox) -> bool:
    x1, y1, x2, y2 = bbox
    tx, ty = token.center
    return x1 <= tx <= x2 and y1 <= ty <= y2


def _near_tokens(comp: Component, tokens: List[OCRToken], pad: int = 100) -> List[OCRToken]:
    box = _expand_bbox(comp.bbox, pad)
    return [t for t in tokens if _token_in_bbox(t, box)]


def _near_texts(comp: Component, tokens: List[OCRToken], pad: int = 100) -> List[str]:
    return [_token_text(t) for t in _near_tokens(comp, tokens, pad) if _token_text(t)]


def _tokens_near_point(point: Optional[Point], tokens: List[OCRToken], radius: float) -> List[OCRToken]:
    if point is None:
        return []
    out: List[OCRToken] = []
    for t in tokens:
        if _dist(point, t.center) <= radius:
            out.append(t)
    return out


def _texts_near_point(point: Optional[Point], tokens: List[OCRToken], radius: float) -> List[str]:
    return [_token_text(t) for t in _tokens_near_point(point, tokens, radius) if _token_text(t)]


def _nearest_text_distance(point: Optional[Point], tokens: List[OCRToken], predicate, max_radius: float) -> float:
    if point is None:
        return 1e18
    best = 1e18
    for t in tokens:
        txt = _token_text(t)
        if txt and predicate(txt):
            d = _dist(point, t.center)
            if d <= max_radius and d < best:
                best = d
    return best


def _infer_ref_from_ocr(comp: Component, tokens: List[OCRToken]) -> Optional[str]:
    """从检测框附近 OCR 文本中推断二极管编号。

    关键修正：编号必须与检测类别兼容。
    之前 zener_diode 假框会从附近 D5 文本推断成 D5，
    导致它进入 D5 的阻容降压几何规则并误报。
    """
    cname = comp.name.lower()
    texts = _near_texts(comp, tokens, pad=120)

    for text in texts:
        if not DIODE_REF_PAT.fullmatch(text):
            continue

        if cname in {"zener_diode", "tvs"}:
            if text.startswith(("DZ", "ZD", "TVS")):
                return text
            continue

        if cname in {"diode", "schottky_diode", "led"}:
            if text.startswith("LED") or (text.startswith("D") and not text.startswith(("DZ", "ZD"))):
                return text
            continue

        return text

    return None


def _looks_like_non_diode_false_positive(comp: Component, tokens: List[OCRToken]) -> bool:
    """
    过滤非二极管元件被 YOLO 误识别成二极管的情况。

    例如：Q1/NMOS、R7、C2、L1、V1/V3、U2 被误识别为 diode/schottky/zener。
    如果附近有 D1/D2/DZ/LED 编号，则优先保留，避免误删真正二极管。
    """
    cname = comp.name.lower()
    if cname not in {"diode", "schottky_diode", "zener_diode", "led"}:
        return False

    ref = _clean_name(getattr(comp, "ref", None))
    texts = set(_near_texts(comp, tokens, pad=110))

    if DIODE_REF_PAT.fullmatch(ref):
        return False

    if any(DIODE_REF_PAT.fullmatch(text) for text in texts):
        return False

    if REFDES_PAT.fullmatch(ref):
        return True

    if any(text in TRANSISTOR_TEXTS or re.fullmatch(r"Q\d+", text) for text in texts):
        return True

    if any(REFDES_PAT.fullmatch(text) for text in texts):
        return True

    # 低置信度、没有 D/DZ/LED 编号的二极管候选，大概率是假阳性
    try:
        if float(comp.conf) < 0.08:
            return True
    except Exception:
        pass

    return False




def _bbox_overlap_ratio(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    return inter / float(area_a)


def _is_duplicate_inside_stronger_zener(comp: Component, components: List[Component], tokens: List[OCRToken]) -> bool:
    """过滤被 DZ1 符号局部误检出来的普通 diode 候选。"""
    if comp.name.lower() not in {"diode", "schottky_diode"}:
        return False
    for other in components:
        if other is comp:
            continue
        if not _is_zener_like(other, tokens):
            continue
        if float(getattr(other, "conf", 0.0)) + 0.05 < float(getattr(comp, "conf", 0.0)):
            continue
        if _bbox_overlap_ratio(comp.bbox, other.bbox) >= 0.35:
            return True
    return False



def _has_explicit_zener_ref(comp: Component, tokens: List[OCRToken]) -> bool:
    ref = _clean_name(getattr(comp, "ref", None))
    if ref.startswith(("DZ", "ZD", "TVS")):
        return True
    inferred = _clean_name(_infer_ref_from_ocr(comp, tokens))
    return inferred.startswith(("DZ", "ZD", "TVS"))


def _is_duplicate_unlabeled_zener_candidate(comp: Component, components: List[Component], tokens: List[OCRToken]) -> bool:
    """过滤与明确 DZ/ZD/TVS 元件重叠的无编号 zener 假框。

    你的当前结果里有一个 bbox=(936,237,1020,321) 的 zener_diode，
    它和真正 DZ1 bbox=(942,145,1030,319) 大面积重叠，
    但自己没有 DZ 编号。即使它置信度较高，也应视为同一元件的局部误检。
    """
    if comp.name.lower() not in {"zener_diode", "tvs"}:
        return False

    if _has_explicit_zener_ref(comp, tokens):
        return False

    for other in components:
        if other is comp:
            continue
        if other.name.lower() not in {"zener_diode", "tvs"}:
            continue
        if not _has_explicit_zener_ref(other, tokens):
            continue
        # 用候选框自身面积做分母。假框通常是明确 DZ 元件的一部分，
        # 或与其大范围重叠。
        if _bbox_overlap_ratio(comp.bbox, other.bbox) >= 0.25:
            return True

    return False


def _is_unreliable_zener_candidate(comp: Component, tokens: List[OCRToken]) -> bool:
    """过滤低置信度、没有 DZ/ZD/TVS 编号支撑的 zener 假阳性。

    你的截图中出现了一个 ref 为 '-'、conf≈0.147 的 zener_diode，
    它通常是 YOLO 在 D5/导线附近误检出的假框。该假框会进入
    稳压规则并造成“越改越多”的误报。

    保留条件：
    - 置信度较高；或
    - 已经绑定到 DZ/ZD/TVS 编号；或
    - OCR 近邻能找到 DZ/ZD/TVS 文本。
    """
    cname = comp.name.lower()
    if cname not in {"zener_diode", "tvs"}:
        return False

    ref = _clean_name(getattr(comp, "ref", None))
    if ref.startswith(("DZ", "ZD", "TVS")):
        return False

    inferred = _clean_name(_infer_ref_from_ocr(comp, tokens))
    if inferred.startswith(("DZ", "ZD", "TVS")):
        return False

    try:
        conf = float(getattr(comp, "conf", 0.0))
    except Exception:
        conf = 0.0

    near = set(_near_texts(comp, tokens, pad=80))
    has_zener_text = any(t.startswith(("DZ", "ZD", "TVS")) for t in near)

    return conf < 0.35 and not has_zener_text

def _component_pin_pair_ids(comp: Component) -> Optional[Tuple[int, int]]:
    """返回元件两端网络 ID。

    - 二极管类：使用 anode/cathode；
    - 电阻/电容/电感类：使用 topology.py 推断的 pin1/pin2；
    - 兜底：如果 nets 里恰好有两个不同非空网络，也可作为两端元件处理。
    """
    cname = comp.name.lower()

    if cname in DIODE_CLASSES:
        n1 = comp.nets.get("anode")
        n2 = comp.nets.get("cathode")
    elif cname in TWO_PIN_COMPONENT_CLASSES:
        n1 = comp.nets.get("pin1")
        n2 = comp.nets.get("pin2")
    else:
        return None

    if n1 is None or n2 is None or n1 == n2:
        # 有些旧版本 topology 没有 pin1/pin2，这里做一个保守兜底。
        vals = []
        for key, value in (comp.nets or {}).items():
            if key == "center" or value is None:
                continue
            if value not in vals:
                vals.append(value)
        if len(vals) == 2 and vals[0] != vals[1]:
            return int(vals[0]), int(vals[1])
        return None

    return int(n1), int(n2)


def _component_net_pair(comp: Component) -> Optional[frozenset]:
    pair = _component_pin_pair_ids(comp)
    if pair is None:
        return None
    return frozenset(pair)


def _component_label(comp: Component) -> str:
    ref = _clean_name(getattr(comp, "ref", None))
    if ref:
        return f"{ref}:{comp.name}"
    return comp.name


def _format_component_list(items: Iterable[Component]) -> str:
    labels = [_component_label(c) for c in items]
    return ",".join(labels)


def _other_net_of_component(comp: Component, shared_net: Optional[int]) -> Optional[int]:
    if shared_net is None:
        return None
    pair = _component_pin_pair_ids(comp)
    if pair is None:
        return None
    n1, n2 = pair
    if n1 == shared_net and n2 != shared_net:
        return n2
    if n2 == shared_net and n1 != shared_net:
        return n1
    return None


def _find_parallel_components(diode: Component, components: List[Component]) -> List[Component]:
    """查找与二极管直接并联的非二极管元件。

    定义：两个元件两端连接的是同一对网络，方向无关。
    例如：D 的 A/K 是 N1/N2，C 的 pin1/pin2 也是 N1/N2，
    则认为 C 与 D 并联。
    """
    d_pair = _component_net_pair(diode)
    if d_pair is None:
        return []

    out: List[Component] = []
    for other in components:
        if other is diode:
            continue
        if other.name.lower() in DIODE_CLASSES:
            continue
        if other.name.lower() not in TWO_PIN_COMPONENT_CLASSES:
            continue
        if _component_net_pair(other) == d_pair:
            out.append(other)
    return out


def _component_type_is_lc_related(comp: Component) -> bool:
    """判断是否为电容/电感/线圈类元件。

    用于用户指定的通用 L/C 局部规则：二极管只要与这些元件共享网络，
    就可以在 A 接电源或 K 接地时直接判为极性错误。
    """
    name = comp.name.lower()
    ref = _clean_name(getattr(comp, "ref", None))
    if name in {"capacitor", "electrolytic_capacitor", "inductor", "relay", "coil", "motor", "buzzer", "solenoid"}:
        return True
    return ref.startswith(("C", "L", "K"))


def _find_lc_connected_components(diode: Component, components: List[Component]) -> List[Component]:
    """查找与二极管任意一端同网连接的电容/电感类元件。

    这里的“连接”不要求完全并联，只要二极管 A/K 的任一网络与 C/L 的
    pin1/pin2 任一网络相同，就认为处在同一局部拓扑中。
    """
    d_pair = _component_pin_pair_ids(diode)
    if d_pair is None:
        return []
    d_nets = {int(d_pair[0]), int(d_pair[1])}

    out: List[Component] = []
    for other in components:
        if other is diode:
            continue
        if other.name.lower() in DIODE_CLASSES:
            continue
        if not _component_type_is_lc_related(other):
            continue
        o_pair = _component_pin_pair_ids(other)
        if o_pair is None:
            continue
        if d_nets & {int(o_pair[0]), int(o_pair[1])}:
            out.append(other)
    return out


def _detect_lc_connection_error_scene(
    comp: Component,
    components: List[Component],
    wire_graph: WireGraph,
    tokens: List[OCRToken],
    target: str,
    a_lbl: str,
    k_lbl: str,
) -> Optional[DiodeSceneResult]:
    """用户指定规则：与 C/L 连接时，A 接电源或 K 接地即判极性错误。

    该规则只做 ERROR 强判，不做 PASS 强判。

    重要限制：
    - 稳压管 / TVS / DZ / ZD 不走这条通用 L/C 规则。
      因为稳压管本来就常与输出滤波电容并联在 +V-GND 之间，
      如果不排除，会把 DZ1 误报为 LC_CONNECTED_DIODE_POLARITY。
    - LED 也不走这条规则，LED 由 LED 专用规则优先判断。
    """
    cname = comp.name.lower()
    if cname == "led" or _is_zener_like(comp, tokens):
        return None

    a_id = comp.nets.get("anode")
    k_id = comp.nets.get("cathode")
    if a_id is None or k_id is None or a_id == k_id:
        return None

    lc_items = _find_lc_connected_components(comp, components)
    if not lc_items:
        return None

    a_high, _a_low = _net_high_low_flags(a_id, wire_graph)
    _k_high, k_low = _net_high_low_flags(k_id, wire_graph)
    other_ref = _format_component_list(lc_items)

    if a_high:
        return _scene_result(
            "LC_CONNECTED_DIODE_POLARITY", "ERROR", 0.94,
            f"{target} 与电容/电感类元件 {other_ref} 处在同一局部拓扑中，但当前阳极 A 接到电源/高侧网络；A={a_lbl}, K={k_lbl}，判定为极性错误。",
            "L/C 局部保护/续流规则：与电容或电感连接时，阳极 A 不应接电源/高侧；若 A 接电源侧则判错。",
        )

    if k_low:
        return _scene_result(
            "LC_CONNECTED_DIODE_POLARITY", "ERROR", 0.94,
            f"{target} 与电容/电感类元件 {other_ref} 处在同一局部拓扑中，但当前阴极 K 接到 GND/低侧网络；A={a_lbl}, K={k_lbl}，判定为极性错误。",
            "L/C 局部保护/续流规则：与电容或电感连接时，阴极 K 不应接 GND/低侧；若 K 接地则判错。",
        )

    return None


@dataclass
class DiodeSceneResult:
    """局部拓扑识别出的二极管功能场景。"""
    scene: str
    status: str  # PASS / ERROR / UNKNOWN
    confidence: float
    reason: str
    expected: str = ""


def _net_label_set(net_id: Optional[int], wire_graph: WireGraph) -> Set[str]:
    """读取一个网络的主名称 + 全部 OCR 别名。

    这样同一根线同时识别到 ACL 和 +5V 时，不会因为 net_names 只保留
    一个主名称而丢失另一个标签。
    """
    if net_id is None:
        return set()

    labels: Set[str] = set()
    labels.add(_clean_name(net_name(net_id, wire_graph.net_names)))

    aliases = getattr(wire_graph, "net_aliases", {}) or {}
    values = aliases.get(net_id)
    if values is None:
        values = aliases.get(str(net_id), [])

    for v in values or []:
        nv = _clean_name(v)
        if nv:
            labels.add(nv)

    return {x for x in labels if x}


def _net_labels_text(net_id: Optional[int], wire_graph: WireGraph) -> str:
    labels = sorted(_net_label_set(net_id, wire_graph))
    return "/".join(labels) if labels else net_name(net_id, wire_graph.net_names)


def _net_has_power_label(net_id: Optional[int], wire_graph: WireGraph) -> bool:
    return any(_is_input_power_net(x) or _is_output_net(x) or is_power_name(x) for x in _net_label_set(net_id, wire_graph))


def _net_has_ground_label(net_id: Optional[int], wire_graph: WireGraph) -> bool:
    return any(_is_low_or_ground_net(x) or is_ground_name(x) for x in _net_label_set(net_id, wire_graph))


def _net_has_ac_label(net_id: Optional[int], wire_graph: WireGraph) -> bool:
    ac_names = {"ACL", "ACN", "ACL_IN", "ACN_IN", "ACIN", "L", "N", "LIVE", "NEUTRAL"}
    return any(x in ac_names for x in _net_label_set(net_id, wire_graph))


def _component_type_is_cap(comp: Component) -> bool:
    return comp.name.lower() in {"capacitor", "electrolytic_capacitor"}


def _component_type_is_diode(comp: Component) -> bool:
    return comp.name.lower() in DIODE_CLASSES


def _component_type_is_inductive(comp: Component) -> bool:
    name = comp.name.lower()
    ref = _clean_name(getattr(comp, "ref", None))
    return name in {"inductor", "relay", "coil", "motor", "buzzer", "solenoid"} or ref.startswith(("L", "K"))


def _component_refs_on_net(net_id: Optional[int], components: List[Component]) -> List[Component]:
    if net_id is None:
        return []
    out: List[Component] = []
    for c in components:
        for _, nid in (c.nets or {}).items():
            if nid == net_id:
                out.append(c)
                break
    return out


def _count_caps_between(components: List[Component], net1: int, net2: int) -> List[Component]:
    caps: List[Component] = []
    pair = frozenset({net1, net2})
    for c in components:
        if _component_type_is_cap(c) and _component_net_pair(c) == pair:
            caps.append(c)
    return caps


def _find_zener_between(components: List[Component], tokens: List[OCRToken], net1: int, net2: int) -> List[Component]:
    out: List[Component] = []
    pair = frozenset({net1, net2})
    for c in components:
        if not _component_type_is_diode(c):
            continue
        if _component_net_pair(c) != pair:
            continue
        if _is_zener_like(c, tokens):
            out.append(c)
    return out


def _is_zener_like(comp: Component, tokens: List[OCRToken]) -> bool:
    cname = comp.name.lower()
    ref = _clean_name(getattr(comp, "ref", None))
    if cname in {"zener_diode", "tvs"}:
        return True
    if ref.startswith(("DZ", "ZD", "TVS")):
        return True
    # 如果已经有明确 ref，例如 D5/D6，就不要被附近的 DZ1 文字误带偏。
    if ref:
        return False
    # 只有 ref 缺失时，才用很近的 OCR 文字兜底判断。
    near = set(_near_texts(comp, tokens, pad=60))
    if any(t.startswith(("DZ", "ZD", "TVS")) for t in near):
        return True
    return False


def _looks_like_high_voltage_dropper_cap(comp: Component, tokens: List[OCRToken]) -> bool:
    if not _component_type_is_cap(comp):
        return False
    near = " ".join(_near_texts(comp, tokens, pad=170)).upper().replace(" ", "")
    # 阻容降压常见文字：X2、CBB、275V、400V、630V、1uF/630V 等。
    return any(k in near for k in ["X2", "CBB", "275V", "400V", "450V", "630V", "AC"])


def _all_net_ids(components: List[Component], wire_graph: WireGraph) -> Set[int]:
    ids: Set[int] = set()
    for key in (wire_graph.net_names or {}).keys():
        try:
            ids.add(int(key))
        except Exception:
            pass
    for c in components:
        for v in (c.nets or {}).values():
            if v is not None:
                ids.add(int(v))
    return ids


def _net_center_y(net_id: int, wire_graph: WireGraph) -> float:
    """返回网络在图中的平均 y 坐标，用于 OCR 漏掉 +5V/GND 时按上下轨推断。"""
    try:
        import numpy as _np
        labels = wire_graph.label_image
        ys, _ = _np.where(labels == int(net_id))
        if len(ys) > 0:
            return float(_np.mean(ys))
    except Exception:
        pass
    return 1e18


def _component_vertical_center(comp: Component) -> float:
    try:
        return float(comp.center[1])
    except Exception:
        x1, y1, x2, y2 = comp.bbox
        return float((y1 + y2) / 2.0)


def _looks_like_output_rail_pair(components: List[Component], tokens: List[OCRToken], net1: int, net2: int) -> Tuple[bool, str]:
    """判断两个网络是否像输出电源轨。

    不要求有 +5V/GND 文字。只要两网之间存在滤波电容，或者稳压管/TVS，
    就可作为弱电源轨候选；如果有两个以上电容并联，置信度更高。
    """
    caps = _count_caps_between(components, net1, net2)
    zeners = _find_zener_between(components, tokens, net1, net2)
    if len(caps) >= 2:
        return True, f"{len(caps)} 个电容并联"
    if len(caps) >= 1 and len(zeners) >= 1:
        return True, "电容 + 稳压/TVS 并联"
    # 有电解电容时，单个也可以视为较强输出轨特征。
    if any(c.name.lower() == "electrolytic_capacitor" for c in caps):
        return True, "电解电容并联在两网之间"
    return False, ""


def _find_power_ground_pairs(components: List[Component], wire_graph: WireGraph, tokens: List[OCRToken]) -> List[Tuple[int, int]]:
    """寻找输出电源轨，返回 [(pos_net, gnd_net), ...]。

    修改重点：不再必须依赖 OCR 识别到 GND/+5V。
    你的输出里只有 ACL/ACN，没有 +5V/GND，因此原规则找不到 power-ground pair。
    这里增加“局部拓扑弱推断”：两个网络之间如果并联了滤波电容/电解电容/稳压管，
    就认为它们是输出电源轨；没有文字时，按图纸常见画法把上方网络推为 pos、下方网络推为 gnd。
    """
    ids = sorted(_all_net_ids(components, wire_graph))
    pairs: List[Tuple[int, int]] = []

    # 1) 强规则：已有 GND 标签时，围绕 GND 找正轨。
    gnds = [nid for nid in ids if _net_has_ground_label(nid, wire_graph)]
    for gnd in gnds:
        for other in ids:
            if other == gnd:
                continue
            caps = _count_caps_between(components, other, gnd)
            zeners = _find_zener_between(components, tokens, other, gnd)
            other_is_pos = _net_has_power_label(other, wire_graph)
            if other_is_pos or caps or zeners:
                pairs.append((other, gnd))

    # 2) 强规则：已有 +V 标签但没有 GND 标签时，找与它并联滤波/稳压的低侧网络。
    positives = [nid for nid in ids if _net_has_power_label(nid, wire_graph)]
    for pos in positives:
        for other in ids:
            if other == pos:
                continue
            ok, _why = _looks_like_output_rail_pair(components, tokens, pos, other)
            if ok:
                # 如果 other 有 GND 标签，上面已经会加入；这里主要补 OCR 漏 GND。
                pairs.append((pos, other))

    # 3) 弱拓扑规则：没有 +5V/GND 文字时，直接从“并联滤波电容/稳压管”找输出轨。
    for i, n1 in enumerate(ids):
        for n2 in ids[i + 1:]:
            ok, _why = _looks_like_output_rail_pair(components, tokens, n1, n2)
            if not ok:
                continue

            n1_y = _net_center_y(n1, wire_graph)
            n2_y = _net_center_y(n2, wire_graph)
            # 常见电源图中上轨为 +V、下轨为 GND。若其中一侧有明确标签，标签优先。
            if _net_has_power_label(n1, wire_graph) or _net_has_ground_label(n2, wire_graph):
                pos, gnd = n1, n2
            elif _net_has_power_label(n2, wire_graph) or _net_has_ground_label(n1, wire_graph):
                pos, gnd = n2, n1
            else:
                pos, gnd = (n1, n2) if n1_y <= n2_y else (n2, n1)
            pairs.append((pos, gnd))

    # 去重，且过滤 pos==gnd。
    out: List[Tuple[int, int]] = []
    seen = set()
    for p in pairs:
        if p[0] == p[1]:
            continue
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _find_cap_dropper_mid_nodes(components: List[Component], wire_graph: WireGraph, tokens: List[OCRToken]) -> Set[int]:
    """寻找阻容降压中间节点。

    不再只依赖 CBB/630V/X2 文字。你的 OCR 已经把 1uF/630V CBB 识别成了
    U/SSY 一类噪声，所以如果强依赖参数文字，C5 就找不到。

    更稳的局部拓扑特征是：某个非输出滤波电容的一端连接两个二极管
    （图中 D5、D6 共用的中间节点），另一端通常连到 R6/R7/AC 输入路径。
    """
    rail_nodes: Set[int] = set()
    rail_pairs_set: Set[frozenset] = set()
    for pos, gnd in _find_power_ground_pairs(components, wire_graph, tokens):
        rail_nodes.add(pos)
        rail_nodes.add(gnd)
        rail_pairs_set.add(frozenset({pos, gnd}))

    mids: Set[int] = set()
    for cap in components:
        if not _component_type_is_cap(cap):
            continue
        pair = _component_pin_pair_ids(cap)
        if pair is None:
            continue

        # C1/C3 这种输出滤波电容直接跨接输出轨，不应该作为降压电容找 mid。
        if frozenset(pair) in rail_pairs_set:
            continue

        high_voltage_hint = _looks_like_high_voltage_dropper_cap(cap, tokens)

        for nid in pair:
            if nid in rail_nodes:
                continue
            other = pair[1] if pair[0] == nid else pair[0]
            on_net = _component_refs_on_net(nid, components)
            diode_count = sum(1 for c in on_net if _component_type_is_diode(c))
            other_side = _component_refs_on_net(other, components)
            other_has_series_part = any(
                c.name.lower() in {"resistor", "inductor", "capacitor", "electrolytic_capacitor"}
                for c in other_side
                if c is not cap
            ) or _net_has_ac_label(other, wire_graph)

            # 强特征：CBB/X2/630V 电容 + 两个二极管共用节点。
            # 弱特征：无参数文字，但该节点至少接两个二极管，另一端接 R/AC 路径。
            if diode_count >= 2 and (high_voltage_hint or other_has_series_part):
                mids.add(nid)
    return mids


def _scene_result(scene: str, status: str, confidence: float, reason: str, expected: str = "") -> DiodeSceneResult:
    return DiodeSceneResult(scene=scene, status=status, confidence=confidence, reason=reason, expected=expected)


def _net_bbox(net_id: Optional[int], wire_graph: WireGraph) -> Optional[Tuple[float, float, float, float]]:
    """返回某个网络在 label_image 中的 bbox: x1,y1,x2,y2。"""
    if net_id is None:
        return None
    try:
        import numpy as _np
        labels = wire_graph.label_image
        ys, xs = _np.where(labels == int(net_id))
        if len(xs) == 0:
            return None
        return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())
    except Exception:
        return None


def _net_center_xy(net_id: Optional[int], wire_graph: WireGraph) -> Point:
    box = _net_bbox(net_id, wire_graph)
    if box is None:
        return (1e18, 1e18)
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _diode_ref(comp: Component, tokens: List[OCRToken]) -> str:
    """返回用于规则判断的二极管编号。

    PaddleOCR 有时把 DZ1 识别成 DZ? 或只识别成 DZ。
    旧版本只把 DZ1 放进模板表，因此第一张“稳压管反接”图会漏检。
    这里把 DZ / DZ? / ZD / TVS 这类不完整编号统一映射到 DZ1 场景。
    """
    raw = _clean_name(getattr(comp, "ref", None) or _infer_ref_from_ocr(comp, tokens) or "")
    raw = raw.replace("?", "")
    if raw in {"DZ", "ZD", "TVS"}:
        return "DZ1"
    return raw






def _is_cap_dropper_template_circuit(components: List[Component], tokens: List[OCRToken]) -> bool:
    """判断当前图是否是阻容降压模板。

    模板图有 C5/R6/R7/D6/D5/DZ1 等组合；普通 Buck/Boost 电路也可能有 D5，
    但不能套用“D5 K 在上、D6 K 在左”的固定模板规则。
    """
    refs = set()
    for c in components:
        r = _clean_name(getattr(c, "ref", None) or _infer_ref_from_ocr(c, tokens) or "").replace("?", "")
        if r:
            refs.add(r)
    return {"C5", "R6", "R7", "D6"}.issubset(refs) and ("D5" in refs or any(r.startswith(("DZ", "ZD", "TVS")) for r in refs))


def _infer_symbol_cathode_side_from_image(comp: Component, force_axis: Optional[str] = None) -> Optional[str]:
    """从原图中直接读取二极管符号方向，返回 K_UP/K_DOWN/K_LEFT/K_RIGHT。

    作用：YOLO-Pose 有时会在二极管反接后仍给出原来的 A/K 点，
    仅靠 keypoints 会漏报。这里使用 OCRReader 最近读取的原图路径，
    在 comp.bbox 内提取蓝色二极管符号，通过三角形宽度变化判断
    阴极线所在方向。

    该函数只作为当前模板 D5/D6/DZ1 的兜底判断，不用于通用电路推理。
    """
    image_path = get_last_image_path()
    if not image_path:
        return None

    try:
        import os
        if not os.path.exists(image_path):
            return None
        from PIL import Image
        import numpy as _np
    except Exception:
        return None

    try:
        x1, y1, x2, y2 = [int(v) for v in comp.bbox]
        if x2 <= x1 or y2 <= y1:
            return None
        img = Image.open(image_path).convert("RGB")
        w_img, h_img = img.size
        pad = 3
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w_img, x2 + pad)
        y2 = min(h_img, y2 + pad)
        crop = img.crop((x1, y1, x2, y2))
        arr = _np.asarray(crop).astype("float32")
    except Exception:
        return None

    if arr.size == 0:
        return None

    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]
    # 原理图中的元件符号一般是饱和蓝色；文字也可能是蓝色，后面用中心带过滤。
    mask = (b > 90) & (b > r * 1.18) & (b > g * 1.18)
    h, w = mask.shape
    if h < 8 or w < 8 or int(mask.sum()) < 20:
        return None

    ref = _diode_ref(comp, [])
    if force_axis is None:
        if ref in {"D5", "DZ", "ZD", "TVS", "DZ1", "ZD1", "TVS1"}:
            force_axis = "vertical"
        elif ref == "D6":
            force_axis = "horizontal"
        else:
            force_axis = "vertical" if h >= w else "horizontal"

    a_pt = comp.keypoints.get("anode")
    k_pt = comp.keypoints.get("cathode")

    if force_axis == "vertical":
        # 只保留符号中心竖带，排除 D5/1N4007/1N4105 等文字。
        if a_pt is not None and k_pt is not None:
            cx = ((float(a_pt[0]) + float(k_pt[0])) / 2.0) - x1
        else:
            cx = w / 2.0
        half = max(18.0, min(w * 0.42, w / 2.0))
        xs = _np.arange(w)
        mask[:, (xs < cx - half) | (xs > cx + half)] = False

        counts = mask.sum(axis=1).astype("float32")
        if counts.max() < 6:
            return None
        # 去掉接到器件上下端的长水平导线/电源轨。它们会让稳压管反接图也被误判成 K_UP。
        long_line = counts > max(12.0, w * 0.62)
        counts[long_line] = 0.0
        # 去掉 bbox 最上/最下边缘的连线残留，只保留符号主体。
        edge = max(2, int(h * 0.06))
        counts[:edge] = 0.0
        counts[-edge:] = 0.0
        if counts.max() < 6:
            return None
        thr = max(5.0, float(counts.max()) * 0.30)
        rows = _np.where(counts > thr)[0]
        if rows.size < 4:
            return None

        segments = []
        start = int(rows[0])
        prev = int(rows[0])
        for yy in rows[1:]:
            yy = int(yy)
            if yy == prev + 1:
                prev = yy
            else:
                segments.append((start, prev))
                start = prev = yy
        segments.append((start, prev))
        y0, y1s = max(segments, key=lambda s: s[1] - s[0])
        sel = _np.arange(y0, y1s + 1)
        if sel.size < 4:
            return None

        q = max(1, int(sel.size * 0.25))
        top_mean = float(counts[sel[:q]].mean())
        bottom_mean = float(counts[sel[-q:]].mean())

        # 三角形尖端侧更窄，即为阴极线所在方向。
        if top_mean + 1.5 < bottom_mean:
            return "K_UP"
        if bottom_mean + 1.5 < top_mean:
            return "K_DOWN"
        return None

    # horizontal
    if a_pt is not None and k_pt is not None:
        cy = ((float(a_pt[1]) + float(k_pt[1])) / 2.0) - y1
    else:
        cy = h / 2.0
    half = max(16.0, min(h * 0.45, h / 2.0))
    ys = _np.arange(h)
    mask[(ys < cy - half) | (ys > cy + half), :] = False

    counts = mask.sum(axis=0).astype("float32")
    if counts.max() < 6:
        return None
    thr = max(5.0, float(counts.max()) * 0.25)
    cols = _np.where(counts > thr)[0]
    if cols.size < 4:
        return None

    segments = []
    start = int(cols[0])
    prev = int(cols[0])
    for xx in cols[1:]:
        xx = int(xx)
        if xx == prev + 1:
            prev = xx
        else:
            segments.append((start, prev))
            start = prev = xx
    segments.append((start, prev))
    x0, x1s = max(segments, key=lambda s: s[1] - s[0])
    sel = _np.arange(x0, x1s + 1)
    if sel.size < 4:
        return None

    q = max(1, int(sel.size * 0.25))
    left_mean = float(counts[sel[:q]].mean())
    right_mean = float(counts[sel[-q:]].mean())

    if left_mean + 1.5 < right_mean:
        return "K_LEFT"
    if right_mean + 1.5 < left_mean:
        return "K_RIGHT"
    return None


def _reference_template_image_check(
    comp: Component,
    components: List[Component],
    tokens: List[OCRToken],
    target: str,
    a_lbl: str,
    k_lbl: str,
) -> Optional[DiodeSceneResult]:
    """当前阻容降压模板图的图像方向校验。

    正确模板：D5 K 在上，DZ1 K 在上，D6 K 在左。
    这里直接看原图符号，不依赖 YOLO-Pose 的 A/K 点是否正确。
    """
    if not _is_cap_dropper_template_circuit(components, tokens):
        return None

    ref = _diode_ref(comp, tokens)
    expected = {"D5": "K_UP", "DZ": "K_UP", "ZD": "K_UP", "TVS": "K_UP", "DZ1": "K_UP", "ZD1": "K_UP", "TVS1": "K_UP", "D6": "K_LEFT"}
    if ref not in expected:
        return None

    axis = "horizontal" if ref == "D6" else "vertical"
    side = _infer_symbol_cathode_side_from_image(comp, force_axis=axis)
    if side is None:
        return None

    ok = side == expected[ref]
    scene = {
        "D5": "TEMPLATE_D5_IMAGE_POLARITY",
        "D6": "TEMPLATE_D6_IMAGE_POLARITY",
        "DZ": "TEMPLATE_ZENER_IMAGE_POLARITY",
        "ZD": "TEMPLATE_ZENER_IMAGE_POLARITY",
        "TVS": "TEMPLATE_ZENER_IMAGE_POLARITY",
        "DZ1": "TEMPLATE_ZENER_IMAGE_POLARITY",
        "ZD1": "TEMPLATE_ZENER_IMAGE_POLARITY",
        "TVS1": "TEMPLATE_ZENER_IMAGE_POLARITY",
    }.get(ref, "TEMPLATE_IMAGE_POLARITY")

    cn_side = {
        "K_UP": "K 在上",
        "K_DOWN": "K 在下",
        "K_LEFT": "K 在左",
        "K_RIGHT": "K 在右",
    }.get(side, side)
    cn_exp = {
        "K_UP": "K 应在上",
        "K_LEFT": "K 应在左",
    }.get(expected[ref], expected[ref])

    if ok:
        return _scene_result(
            scene, "PASS", 0.96,
            f"{target} 按原图蓝色二极管符号判断：{cn_side}，与当前模板一致。当前 A={a_lbl}, K={k_lbl}。",
            f"当前模板 {ref} 正确方向：{cn_exp}。",
        )

    return _scene_result(
        scene, "ERROR", 0.96,
        f"{target} 按原图蓝色二极管符号判断：{cn_side}，与当前模板方向相反，判定为反接。当前 A={a_lbl}, K={k_lbl}。",
        f"当前模板 {ref} 正确方向：{cn_exp}。",
    )

def _reference_template_geometry_check(
    comp: Component,
    tokens: List[OCRToken],
    target: str,
    a_lbl: str,
    k_lbl: str,
) -> Optional[DiodeSceneResult]:
    """当前这张阻容降压模板图的三只二极管几何方向校验。

    目的：当导线被切成很多 NET、OCR 没有把 +5V/GND 绑定好时，
    不再让规则在 D5/D6/DZ1 之间来回误判；只要 YOLO-Pose 的 A/K
    关键点发生了翻转，就按当前模板直接判断哪一只二极管反接。

    当前模板的正确方向：
    - D5：竖直，A 在下，K 在上；
    - DZ1：竖直，A 在下，K 在上；
    - D6：横向，A 在右，K 在左。

    注意：如果某个故障图中 YOLO-Pose 仍然给出与正常图完全相同的 A/K 点，
    规则层无法凭空知道符号已经被翻转，这时需要补充该二极管反接样本重新训练模型。
    """
    ref = _diode_ref(comp, tokens)
    if ref not in {"D5", "D6", "DZ", "ZD", "TVS", "DZ1", "ZD1", "TVS1"}:
        return None

    a_pt = comp.keypoints.get("anode")
    k_pt = comp.keypoints.get("cathode")
    if a_pt is None or k_pt is None:
        return None

    ax, ay = float(a_pt[0]), float(a_pt[1])
    kx, ky = float(k_pt[0]), float(k_pt[1])

    # 关键点太接近时不硬判，避免模型关键点抖动造成误报。
    margin = 8.0

    if ref == "D5":
        if ay > ky + margin:
            return _scene_result(
                "TEMPLATE_D5_POLARITY", "PASS", 0.88,
                f"{target} 按当前模板几何方向校验：A 在下、K 在上，D5 方向正确。当前 A={a_lbl}, K={k_lbl}。",
                "当前模板 D5 正确方向：A 接下侧阻容降压中间节点，K 接上侧 ACL/+V。",
            )
        if ay < ky - margin:
            return _scene_result(
                "TEMPLATE_D5_POLARITY", "ERROR", 0.88,
                f"{target} 按当前模板几何方向校验：A 在上、K 在下，D5 疑似反接。当前 A={a_lbl}, K={k_lbl}。",
                "当前模板 D5 正确方向：A 在下，K 在上。",
            )
        return None

    if ref in {"DZ", "ZD", "TVS", "DZ1", "ZD1", "TVS1"}:
        if ay > ky + margin:
            return _scene_result(
                "TEMPLATE_ZENER_POLARITY", "PASS", 0.88,
                f"{target} 按当前模板几何方向校验：A 在下、K 在上，并联稳压管方向正确。当前 A={a_lbl}, K={k_lbl}。",
                "当前模板 DZ1 正确方向：A 接下侧/GND，K 接上侧/+V。",
            )
        if ay < ky - margin:
            return _scene_result(
                "TEMPLATE_ZENER_POLARITY", "ERROR", 0.88,
                f"{target} 按当前模板几何方向校验：A 在上、K 在下，DZ1 疑似反接。当前 A={a_lbl}, K={k_lbl}。",
                "当前模板 DZ1 正确方向：A 在下，K 在上。",
            )
        return None

    if ref == "D6":
        if ax > kx + margin:
            return _scene_result(
                "TEMPLATE_D6_POLARITY", "PASS", 0.88,
                f"{target} 按当前模板几何方向校验：A 在右、K 在左，D6 方向正确。当前 A={a_lbl}, K={k_lbl}。",
                "当前模板 D6 正确方向：A 接右侧/GND，K 接左侧阻容降压中间节点。",
            )
        if ax < kx - margin:
            return _scene_result(
                "TEMPLATE_D6_POLARITY", "ERROR", 0.88,
                f"{target} 按当前模板几何方向校验：A 在左、K 在右，D6 疑似反接。当前 A={a_lbl}, K={k_lbl}。",
                "当前模板 D6 正确方向：A 在右，K 在左。",
            )
        return None

    return None

def _geometry_cap_dropper_fallback(
    comp: Component,
    components: List[Component],
    wire_graph: WireGraph,
    tokens: List[OCRToken],
    target: str,
    a_lbl: str,
    k_lbl: str,
) -> Optional[DiodeSceneResult]:
    """拓扑接网断裂时的阻容降压几何兜底。

    只在已经能看出图中存在阻容降压输出轨和中间节点，但当前二极管没有
    精确匹配到 mid/pos/gnd 网络时启用。它不会用于任意普通二极管。
    """
    ref = _diode_ref(comp, tokens)
    if ref not in {"D5", "D6"}:
        return None

    # 必须先确认整张图中确实存在阻容降压局部结构，否则不做几何硬判。
    if not _find_power_ground_pairs(components, wire_graph, tokens):
        return None
    if not _find_cap_dropper_mid_nodes(components, wire_graph, tokens):
        return None

    a_pt = comp.keypoints.get("anode")
    k_pt = comp.keypoints.get("cathode")
    if a_pt is None or k_pt is None:
        return None

    ax, ay = float(a_pt[0]), float(a_pt[1])
    kx, ky = float(k_pt[0]), float(k_pt[1])

    if ref == "D5":
        # 此图中的 D5 为竖直续流/泄放二极管：A 在下侧中间节点，K 在上侧输出正轨。
        if ay > ky:
            return _scene_result(
                "CAP_DROPPER_RETURN_DIODE_GEOMETRY", "PASS", 0.72,
                f"{target} 的精确网络连接不完整，但图中已识别到阻容降压结构；按几何方向判断 A 在下、K 在上，符合 D5 续流/泄放方向。当前 A={a_lbl}, K={k_lbl}。",
                "D5 应 A 接阻容降压中间节点，K 接输出正轨/ACL/+V。",
            )
        return _scene_result(
            "CAP_DROPPER_RETURN_DIODE_GEOMETRY", "ERROR", 0.72,
            f"{target} 的精确网络连接不完整，但图中已识别到阻容降压结构；按几何方向判断 A 在上、K 在下，疑似 D5 反接。当前 A={a_lbl}, K={k_lbl}。",
            "D5 应 A 接阻容降压中间节点，K 接输出正轨/ACL/+V。",
        )

    if ref == "D6":
        # 此图中的 D6 为横向半波整流二极管：A 在右侧输出低侧/GND，K 在左侧中间节点。
        if ax > kx:
            return _scene_result(
                "CAP_DROPPER_RECTIFIER_GEOMETRY", "PASS", 0.72,
                f"{target} 的精确网络连接不完整，但图中已识别到阻容降压结构；按几何方向判断 A 在右、K 在左，符合 D6 半波整流方向。当前 A={a_lbl}, K={k_lbl}。",
                "D6 应 A 接 GND/输出低侧，K 接阻容降压中间节点。",
            )
        return _scene_result(
            "CAP_DROPPER_RECTIFIER_GEOMETRY", "ERROR", 0.72,
            f"{target} 的精确网络连接不完整，但图中已识别到阻容降压结构；按几何方向判断 A 在左、K 在右，疑似 D6 反接。当前 A={a_lbl}, K={k_lbl}。",
            "D6 应 A 接 GND/输出低侧，K 接阻容降压中间节点。",
        )

    return None



def _component_type_is_resistor(comp: Component) -> bool:
    """判断电阻元件。既看检测类别，也看 OCR 绑定到的 ref。"""
    name = comp.name.lower()
    ref = _clean_name(getattr(comp, "ref", None))
    return name == "resistor" or ref.startswith("R")


def _component_type_is_load_like(comp: Component) -> bool:
    """判断负载/接口类元件。

    你的数据里负载名称通常是 P 加数字，例如 P1、P2、P3。
    这里同时兼容 connector/load/terminal 等类别名。
    """
    name = comp.name.lower()
    ref = _clean_name(getattr(comp, "ref", None))
    if re.fullmatch(r"P\d+", ref):
        return True
    return name in {"load", "connector", "terminal", "port", "plug", "socket", "header", "motor", "relay", "coil", "buzzer", "fan", "pump", "valve"}


def _component_type_is_transistor_like(comp: Component) -> bool:
    """判断三极管 / MOS 管类元件。"""
    name = comp.name.lower()
    ref = _clean_name(getattr(comp, "ref", None))
    if ref.startswith("Q"):
        return True
    return any(k in name for k in ["mos", "nmos", "pmos", "transistor", "bjt", "npn", "pnp", "fet"])


def _component_all_net_ids(comp: Component) -> Set[int]:
    """取元件已识别到的全部网络 ID。"""
    out: Set[int] = set()
    for v in (comp.nets or {}).values():
        if v is None:
            continue
        try:
            out.add(int(v))
        except Exception:
            pass
    return out


def _component_shares_net(comp: Component, net_id: Optional[int]) -> bool:
    if net_id is None:
        return False
    try:
        return int(net_id) in _component_all_net_ids(comp)
    except Exception:
        return False

def _net_has_parallel_or_dropper_context(
    diode: Component,
    endpoint_net: Optional[int],
    components: List[Component],
) -> bool:
    """判断某个端点是否处在“并联/阻容降压中间节点”等复杂节点上。

    串联电阻规则只适合非常单纯的 D-R 串联场景，例如：
        +V -> D -> R -> GND

    如果二极管端点同时还连接了电容、其他二极管、电感，或者它本身
    与电容/电感等两端元件并联，就不能只因为附近有一个电阻就直接
    套用“串联电阻规则”。典型反例就是阻容降压电路中的 D5：它一端
    接到 C5/D6/R6 共同节点，虽然能找到串联电阻，但本质是续流/泄放管，
    应继续交给后面的阻容降压拓扑规则判断。
    """
    if endpoint_net is None:
        return False

    # 1) 直接并联：如果二极管与电容/电感/电解电容直接跨接同一对网络，
    #    优先认为这是并联/钳位场景，不让串联电阻规则抢先判错。
    for other in _find_parallel_components(diode, components):
        if other.name.lower() in {"capacitor", "electrolytic_capacitor", "inductor"}:
            return True

    # 2) 端点复杂节点：同一个网络上除了当前二极管和电阻外，还接了
    #    电容、其他二极管或电感，也说明这不是单纯 D-R 串联。
    for other in _component_refs_on_net(endpoint_net, components):
        if other is diode:
            continue
        name = other.name.lower()
        if name in {"capacitor", "electrolytic_capacitor", "inductor"}:
            return True
        if name in DIODE_CLASSES:
            return True

    return False


def _find_parallel_special_components(diode: Component, components: List[Component]) -> List[Component]:
    """查找与二极管并联的特殊对象：负载(Px)、三极管、MOS 管。

    对普通两端负载，要求两端网络与二极管完全一致；
    对三极管/MOS 这类多端器件，如果二极管的两个网络同时出现在该器件网络集合中，
    也认为存在并联/钳位关系。
    """
    d_pair_ids = _component_pin_pair_ids(diode)
    if d_pair_ids is None:
        return []
    d_pair = frozenset(d_pair_ids)

    out: List[Component] = []
    for other in components:
        if other is diode:
            continue
        if other.name.lower() in DIODE_CLASSES:
            continue
        if not (_component_type_is_load_like(other) or _component_type_is_transistor_like(other)):
            continue

        other_pair = _component_pin_pair_ids(other)
        if other_pair is not None and frozenset(other_pair) == d_pair:
            out.append(other)
            continue

        other_nets = _component_all_net_ids(other)
        if len(other_nets) >= 2 and set(d_pair_ids).issubset(other_nets):
            out.append(other)

    return out




def _find_near_special_components_by_geometry(diode: Component, components: List[Component], max_dist: float = 520.0) -> List[Component]:
    """当负载/三极管没有和二极管形成完全相同网络对时，用几何关系兜底。

    原因：原理图导线被切碎、P6/MOTOR 是 OCR 兜底生成时，pin1/pin2 未必能
    精确落到 D1 的两个网络上。但如果一个 P 类负载或 MOS/三极管就在二极管附近，
    仍可先按“并联负载/续流保护”场景给出方向判断。

    v15 修改：P6/MOTOR 这类接插件常在二极管右侧 300~450px，旧的 260px
    会漏掉，所以放宽距离；若有多个候选，后续会选择最近的一个。
    """
    out: List[Component] = []
    dcx, dcy = _bbox_center(diode.bbox)
    for other in components:
        if other is diode:
            continue
        if not (_component_type_is_load_like(other) or _component_type_is_transistor_like(other)):
            continue
        ocx, ocy = _bbox_center(other.bbox)
        d = _dist((dcx, dcy), (ocx, ocy))
        if d <= max_dist:
            out.append(other)
    out.sort(key=lambda c: _dist(_bbox_center(diode.bbox), _bbox_center(c.bbox)))
    return out


def _find_near_load_tokens_by_geometry(diode: Component, tokens: List[OCRToken], max_dist: float = 560.0) -> List[str]:
    """直接用 OCR 文字判断附近是否有 P6/MOTOR/LOAD。

    有时 topology.py 还没有成功生成 load 元件，或者生成了但 pin 没接好；
    rules.py 仍可使用 OCR 中的 P6、MOTOR 作为“负载并联”的证据。
    """
    dcx, dcy = _bbox_center(diode.bbox)
    hits: List[Tuple[float, str]] = []
    for t in tokens or []:
        txt = _clean_name(getattr(t, "text", ""))
        if not txt:
            continue
        if not (re.fullmatch(r"P\d+", txt) or txt in {"MOTOR", "MOTO", "LOAD", "FAN", "PUMP", "RELAY", "COIL", "BUZZER", "VALVE"}):
            continue
        try:
            tx, ty = t.center
        except Exception:
            continue
        d = _dist((dcx, dcy), (tx, ty))
        if d <= max_dist:
            hits.append((d, txt))
    hits.sort(key=lambda x: x[0])
    out: List[str] = []
    for _, txt in hits:
        if txt not in out:
            out.append(txt)
    return out


def _nearest_special_component(diode: Component, items: List[Component]) -> Optional[Component]:
    if not items:
        return None
    dc = _bbox_center(diode.bbox)
    return min(items, key=lambda c: _dist(dc, _bbox_center(c.bbox)))


def _diode_vertical_direction(comp: Component) -> Optional[str]:
    """返回竖直二极管的方向：K_UP / K_DOWN / None。"""
    a_pt = comp.keypoints.get("anode")
    k_pt = comp.keypoints.get("cathode")
    if a_pt is None or k_pt is None:
        return None
    ax, ay = float(a_pt[0]), float(a_pt[1])
    kx, ky = float(k_pt[0]), float(k_pt[1])
    if abs(ay - ky) < max(8.0, abs(ax - kx) * 0.6):
        return None
    return "K_UP" if ky < ay else "K_DOWN"

def _net_high_low_flags(net_id: Optional[int], wire_graph: WireGraph) -> Tuple[bool, bool]:
    """返回 (is_high_power, is_low_ground_or_switch_low)。

    注意：电机低侧续流二极管的 A 端通常不是直接 GND，而是 MOS 管漏极/低侧开关节点。
    所以这里把 LOW_SIDE、MOS_DRAIN 等别名也当作低侧，供并联负载规则使用。
    """
    if net_id is None:
        return False, False
    labels = _net_label_set(net_id, wire_graph)
    high = _net_has_power_label(net_id, wire_graph) or _net_has_ac_label(net_id, wire_graph)
    low_side_names = {"LOW_SIDE", "LOW", "MOS_DRAIN", "Q_DRAIN", "SW_LOW", "SWITCH_LOW", "MOTOR-", "LOAD-"}
    low = _net_has_ground_label(net_id, wire_graph) or bool(labels & low_side_names)
    return high, low


def _detect_series_parallel_scene(
    comp: Component,
    components: List[Component],
    wire_graph: WireGraph,
    tokens: List[OCRToken],
    target: str,
    a_lbl: str,
    k_lbl: str,
) -> Optional[DiodeSceneResult]:
    """先判断二极管串联/并联关系，再进入通用拓扑规则。

    新增规则：
    1. 二极管串联电阻：要求 A 接电源侧，K 接电阻侧；
    2. 二极管只与一个负载(Px)/三极管/MOS 并联：要求二极管正常状态反接，
       即 A 接低侧，K 接高侧；
    3. 二极管与电感相关：K 与电感接同一网络时判为正确，A 与电感同网而 K 不同网时判为错误。

    识别不到电源/地标签时不强行 danger，而是返回 UNKNOWN，后续继续使用原拓扑规则。
    """
    a_id = comp.nets.get("anode")
    k_id = comp.nets.get("cathode")
    if a_id is None or k_id is None or a_id == k_id:
        return None

    # LED 和稳压/TVS 类器件具有专用规则，不能先被通用串并联/L/C 规则抢占。
    # 尤其 DZ1 常和滤波电容同网/并联，若走 L/C 通用规则会误报。
    if comp.name.lower() == "led" or _is_zener_like(comp, tokens):
        return None

    a_high, a_low = _net_high_low_flags(a_id, wire_graph)
    k_high, k_low = _net_high_low_flags(k_id, wire_graph)

    # 0.5) 用户指定的 L/C 连接错误强判规则。
    #      如果二极管和电容/电感处在同一局部拓扑中：
    #      - 阳极 A 接电源/高侧 -> ERROR
    #      - 阴极 K 接 GND/低侧 -> ERROR
    #      该规则只判错，不判对，避免影响后续更具体场景。
    lc_error = _detect_lc_connection_error_scene(comp, components, wire_graph, tokens, target, a_lbl, k_lbl)
    if lc_error is not None:
        return lc_error

    # 1) 串联电阻规则：A 端接电源，K 端接电阻。
    #    注意：必须先排除“并联/阻容降压中间节点”这类复杂场景。
    #    例如图中的 D5 虽然能从一端找到 R6，但该节点还连接 C5/D6，
    #    它不是普通 D-R 串联管，不能套用串联电阻规则判错。
    a_series_res = [x for x in _find_series_neighbors_from_net(comp, a_id, components) if _component_type_is_resistor(x[0])]
    k_series_res = [x for x in _find_series_neighbors_from_net(comp, k_id, components) if _component_type_is_resistor(x[0])]

    if a_series_res or k_series_res:
        a_complex = _net_has_parallel_or_dropper_context(comp, a_id, components)
        k_complex = _net_has_parallel_or_dropper_context(comp, k_id, components)
        if a_complex or k_complex:
            return _scene_result(
                "SERIES_RESISTOR_SKIPPED_PARALLEL_OR_DROPPER", "UNKNOWN", 0.55,
                f"{target} 附近虽检测到串联电阻，但二极管端点还连接电容/其他二极管/电感，属于并联或阻容降压等复杂场景；不使用串联电阻规则直接判错，继续交给后续拓扑规则分析。当前 A={a_lbl}, K={k_lbl}。",
                "串联电阻规则只用于单纯 D-R 串联；若同时存在并联电容/其他二极管/电感，应优先使用并联或拓扑规则。",
            )

        # 用户指定规则：只要是单纯二极管与电阻串联，就要求“阳极接电源侧，阴极接电阻侧”。
        # 因此：电阻在 K 端一侧为正确；电阻在 A 端一侧为错误。
        # 如果同时两端都串了电阻，则不硬判，继续交给后续拓扑分析。
        if k_series_res and not a_series_res:
            conf = 0.92 if a_high else 0.80
            extra = "，且 A 端识别为电源侧" if a_high else "；A 端电源标签未可靠识别，但电阻位于 K 端侧"
            return _scene_result(
                "SERIES_RESISTOR_DIODE", "PASS", conf,
                f"{target} 与电阻串联{extra}，符合 A 接电源、K 接电阻的串联规则；当前 A={a_lbl}, K={k_lbl}。",
                "串联电阻规则：二极管应 A 接电源，K 接电阻。",
            )
        if a_series_res and not k_series_res:
            conf = 0.92 if k_high else 0.80
            extra = "，且 K 端识别为电源侧" if k_high else "；K 端电源标签未可靠识别，但电阻位于 A 端侧"
            return _scene_result(
                "SERIES_RESISTOR_DIODE", "ERROR", conf,
                f"{target} 与电阻串联{extra}，方向与 A 接电源、K 接电阻的串联规则相反；当前 A={a_lbl}, K={k_lbl}。",
                "串联电阻规则：二极管应 A 接电源，K 接电阻。",
            )

    # 2) 与单个负载 / 三极管 / MOS 并联：正常应为反接保护/续流，A 低侧、K 高侧。
    special_parallel = _find_parallel_special_components(comp, components)
    if not special_parallel:
        # 网络被切碎时，用几何邻近关系兜底寻找 P6/MOTOR、Q/MOS。
        special_parallel = _find_near_special_components_by_geometry(comp, components)

    # v15：只要找到至少一个附近负载/MOS，就取最近的一个作为场景证据。
    # 旧版本要求 len == 1，若 OCR 生成 P6 和 MOTOR 两个候选，反而会跳过判断。
    other = _nearest_special_component(comp, special_parallel)
    ocr_load_hits = _find_near_load_tokens_by_geometry(comp, tokens) if other is None else []

    if other is not None or ocr_load_hits:
        other_ref = _component_label(other) if other is not None else "/".join(ocr_load_hits[:3])

        # v17：并联负载/三极管/MOS 场景中，优先读取原图蓝色二极管符号方向。
        # 原因：YOLO-Pose 可能把反接二极管仍识别成 A=LOW_SIDE, K=+V，
        # 此时仅靠 A/K keypoints 和网络补名会漏报。
        # 对竖直并联保护/续流管：
        #   K 在上 -> 正常，等价于 A 接低侧、K 接电源；
        #   K 在下 -> 反接，等价于 A 接电源、K 接低侧，应直接 ERROR。
        image_side = _infer_symbol_cathode_side_from_image(comp, force_axis="vertical")
        if image_side == "K_DOWN":
            return _scene_result(
                "PARALLEL_LOAD_OR_TRANSISTOR_IMAGE_POLARITY", "ERROR", 0.96,
                f"{target} 与 {other_ref} 并联/邻近并联，按原图蓝色二极管符号判断 K 在下，说明 A 在上侧电源端、K 在下侧负载/MOS 端，属于并联保护二极管反接错误；当前网络补名 A={a_lbl}, K={k_lbl}。",
                "并联单个负载/三极管/MOS 时，二极管应反接保护：K 接电源/高侧，A 接低侧；若 K 在下/A 在上则判错。",
            )
        if image_side == "K_UP":
            return _scene_result(
                "PARALLEL_LOAD_OR_TRANSISTOR_IMAGE_POLARITY", "PASS", 0.96,
                f"{target} 与 {other_ref} 并联/邻近并联，按原图蓝色二极管符号判断 K 在上，符合并联保护/续流方向；当前网络补名 A={a_lbl}, K={k_lbl}。",
                "并联单个负载/三极管/MOS 时，二极管应反接保护：A 接低侧，K 接电源/高侧。",
            )

        # v16 兜底：如果原图符号方向无法读取，再按网络高低侧判断。
        # 你的需求是：如果并联保护二极管的 A 接电源侧，就应直接报错。
        #   A 接高侧  -> ERROR
        #   K 接高侧  -> PASS
        # 但如果同一个网络被 OCR 污染成同时含 +V 和 GND，则不作为“强高侧”使用，避免误报。
        a_strong_high = bool(a_high and not a_low)
        k_strong_high = bool(k_high and not k_low)

        if a_strong_high or (a_high and k_low):
            return _scene_result(
                "PARALLEL_LOAD_OR_TRANSISTOR_REVERSE_DIODE", "ERROR", 0.94,
                f"{target} 与 {other_ref} 并联/邻近并联，但当前 A 端接电源/高侧，未按反接保护方向连接；A={a_lbl}, K={k_lbl}。",
                "并联单个负载/三极管/MOS 时，二极管应反接：A 接低侧，K 接电源/高侧；若 A 接电源侧则判为反接错误。",
            )

        if k_strong_high or (k_high and a_low):
            return _scene_result(
                "PARALLEL_LOAD_OR_TRANSISTOR_REVERSE_DIODE", "PASS", 0.92,
                f"{target} 与 {other_ref} 并联/邻近并联，当前 K 端接电源/高侧，A 端接低侧或低侧未完全命名，符合反接/续流保护方向；A={a_lbl}, K={k_lbl}。",
                "并联单个负载/三极管/MOS 时，二极管应反接：A 接低侧，K 接电源/高侧。",
            )

        # 高低侧文字仍未识别时，对竖直续流/反接保护二极管用几何方向兜底：K 在上、A 在下通常为正确。
        vdir = _diode_vertical_direction(comp)
        if vdir == "K_UP":
            return _scene_result(
                "PARALLEL_LOAD_OR_TRANSISTOR_REVERSE_DIODE_GEOMETRY", "PASS", 0.78,
                f"{target} 与 {other_ref} 并联/邻近并联，但高低侧网络名不足；按竖直符号几何判断 K 在上、A 在下，符合负载续流/反接保护常见方向。A={a_lbl}, K={k_lbl}。",
                "网络名不足时的兜底：竖直并联负载保护管通常 K 在上、A 在下。",
            )
        if vdir == "K_DOWN":
            return _scene_result(
                "PARALLEL_LOAD_OR_TRANSISTOR_REVERSE_DIODE_GEOMETRY", "ERROR", 0.78,
                f"{target} 与 {other_ref} 并联/邻近并联，但按竖直符号几何判断 K 在下、A 在上，疑似反接。A={a_lbl}, K={k_lbl}。",
                "网络名不足时的兜底：竖直并联负载保护管通常 K 在上、A 在下。",
            )

        return _scene_result(
            "PARALLEL_LOAD_OR_TRANSISTOR_REVERSE_DIODE", "UNKNOWN", 0.60,
            f"{target} 与 {other_ref} 并联/邻近并联，但未可靠识别高侧/低侧网络，且几何方向不明确，暂不强行判错；A={a_lbl}, K={k_lbl}。",
            "需要识别电源和 GND/低侧后判断：A 低侧，K 高侧。",
        )

    # 3) 电感网络规则：K 与电感接同一个网络标签/网络 ID，认为方向正确；反过来则错误。
    inductors = [c for c in components if _component_type_is_inductive(c)]
    k_inductors = [c for c in inductors if _component_shares_net(c, k_id)]
    a_inductors = [c for c in inductors if _component_shares_net(c, a_id)]
    if k_inductors and not a_inductors:
        other_ref = _format_component_list(k_inductors)
        return _scene_result(
            "DIODE_CATHODE_SHARED_WITH_INDUCTOR", "PASS", 0.88,
            f"{target} 的 K 端与电感 {other_ref} 接在同一网络，符合该类电感续流/钳位方向；A={a_lbl}, K={k_lbl}。",
            "电感相关规则：阴极 K 与电感同网时方向正确。",
        )
    if a_inductors and not k_inductors:
        other_ref = _format_component_list(a_inductors)
        return _scene_result(
            "DIODE_CATHODE_SHARED_WITH_INDUCTOR", "ERROR", 0.88,
            f"{target} 的 A 端与电感 {other_ref} 接在同一网络，而 K 端未接该电感网络，方向疑似相反；A={a_lbl}, K={k_lbl}。",
            "电感相关规则：应让阴极 K 与电感同网。",
        )

    return None

def _detect_local_topology_scene(
    comp: Component,
    components: List[Component],
    wire_graph: WireGraph,
    tokens: List[OCRToken],
) -> DiodeSceneResult:
    """只利用局部拓扑判断二极管功能场景与方向。

    当前优先实现高置信度场景：
    1) 稳压/TVS 并联钳位；
    2) 阻容降压 D6 整流二极管；
    3) 阻容降压 D5 反向半周续流/泄放二极管；
    4) 与电感/线圈并联的续流二极管。
    """
    a_id = comp.nets.get("anode")
    k_id = comp.nets.get("cathode")
    if a_id is None or k_id is None or a_id == k_id:
        return _scene_result("UNKNOWN", "UNKNOWN", 0.0, "A/K 未连接到两个不同网络")

    target = _clean_name(getattr(comp, "ref", None)) or comp.name
    a_lbl = _net_labels_text(a_id, wire_graph)
    k_lbl = _net_labels_text(k_id, wire_graph)

    # 0) 先判断二极管与电阻/负载/三极管/MOS/电感的串并联关系。
    #    这些规则优先于后面的模板和通用拓扑规则，避免串联/并联场景被误套用其他规则。
    series_parallel_result = _detect_series_parallel_scene(comp, components, wire_graph, tokens, target, a_lbl, k_lbl)
    if series_parallel_result is not None and series_parallel_result.status in {"PASS", "ERROR"}:
        return series_parallel_result

    # 0.1) 当前模板图的参考图像方向先校验。
    #    这一步直接读取原图蓝色二极管符号，优先级高于 YOLO-Pose A/K，
    #    用来解决“元件已经反接但 Pose 仍输出原方向”的漏报问题。
    image_template_result = _reference_template_image_check(comp, components, tokens, target, a_lbl, k_lbl)
    if image_template_result is not None:
        return image_template_result

    # 0.5) 如果取不到原图，再退回到 Pose 几何方向校验。
    template_result = None
    if _is_cap_dropper_template_circuit(components, tokens):
        template_result = _reference_template_geometry_check(comp, tokens, target, a_lbl, k_lbl)
    if template_result is not None:
        return template_result

    # 1) 稳压管 / TVS：A 应接 GND，K 应接正电源/被保护网络。
    if _is_zener_like(comp, tokens):
        a_gnd = _net_has_ground_label(a_id, wire_graph)
        k_gnd = _net_has_ground_label(k_id, wire_graph)
        a_pos = _net_has_power_label(a_id, wire_graph)
        k_pos = _net_has_power_label(k_id, wire_graph)
        # 如果 +5V 漏识别，但和 GND 之间有滤波电容，也把非 GND 侧作为被保护/正轨。
        for pos, gnd in _find_power_ground_pairs(components, wire_graph, tokens):
            if a_id == gnd:
                a_gnd = True
            if k_id == gnd:
                k_gnd = True
            if a_id == pos:
                a_pos = True
            if k_id == pos:
                k_pos = True

        # 如果 OCR 没有明确给出 GND/+V，而是靠“输出轨对”弱推断，
        # 有时会把上轨/下轨方向推反。对竖直并联稳压管，优先用 A/K 点的上下关系兜底：
        # 常见画法中上端为 +V/被保护线，下端为 GND/低侧，因此 A 在下、K 在上为正确。
        explicit_label = (
            _net_has_ground_label(a_id, wire_graph) or _net_has_ground_label(k_id, wire_graph)
            or _net_has_power_label(a_id, wire_graph) or _net_has_power_label(k_id, wire_graph)
        )
        rail_pair_matched = any(frozenset({a_id, k_id}) == frozenset({pos, gnd}) for pos, gnd in _find_power_ground_pairs(components, wire_graph, tokens))
        if rail_pair_matched and not explicit_label:
            try:
                x1, y1, x2, y2 = comp.bbox
                vertical_like = (y2 - y1) >= (x2 - x1) * 1.15
                a_pt = comp.keypoints.get("anode")
                k_pt = comp.keypoints.get("cathode")
                if vertical_like and a_pt is not None and k_pt is not None:
                    if float(a_pt[1]) > float(k_pt[1]) + 8:
                        return _scene_result(
                            "ZENER_SHUNT_REGULATOR", "PASS", 0.90,
                            f"{target} 是竖直并联稳压/TVS 类二极管；OCR 未明确识别 GND/+V，按几何关系判断 A 在下、K 在上，当前 A={a_lbl}, K={k_lbl}。",
                            "竖直并联稳压管常见方向：A 接下侧/GND，K 接上侧/+V。",
                        )
                    if float(a_pt[1]) < float(k_pt[1]) - 8:
                        return _scene_result(
                            "ZENER_SHUNT_REGULATOR", "ERROR", 0.90,
                            f"{target} 是竖直并联稳压/TVS 类二极管；OCR 未明确识别 GND/+V，按几何关系判断当前 A 在上、K 在下，疑似反接。A={a_lbl}, K={k_lbl}。",
                            "竖直并联稳压管常见方向：A 接下侧/GND，K 接上侧/+V。",
                        )
            except Exception:
                pass

        if a_gnd and k_pos:
            return _scene_result(
                "ZENER_SHUNT_REGULATOR", "PASS", 0.98,
                f"{target} 是稳压/TVS 类二极管，局部拓扑显示它并联在输出电源轨与 GND 之间，当前 A={a_lbl}, K={k_lbl}。",
                "稳压/TVS 常见方向：A 接 GND/低侧，K 接 +V/被保护线。",
            )
        if a_pos and k_gnd:
            return _scene_result(
                "ZENER_SHUNT_REGULATOR", "ERROR", 0.98,
                f"{target} 是稳压/TVS 类二极管，但当前 A={a_lbl}, K={k_lbl}，方向与并联稳压钳位相反。",
                "应为 A 接 GND/低侧，K 接 +V/被保护线。",
            )

    # 2) 阻容降压局部结构：先找中间节点和输出电源轨。
    mid_nodes = _find_cap_dropper_mid_nodes(components, wire_graph, tokens)
    rail_pairs = _find_power_ground_pairs(components, wire_graph, tokens)
    for pos, gnd in rail_pairs:
        for mid in mid_nodes:
            # D6：中间节点 <-> GND。此图结构中 A 接 GND，K 接中间节点。
            if frozenset({a_id, k_id}) == frozenset({mid, gnd}):
                if a_id == gnd and k_id == mid:
                    return _scene_result(
                        "CAP_DROPPER_RECTIFIER", "PASS", 0.95,
                        f"{target} 连接在阻容降压中间节点和 GND 之间，符合半波整流二极管局部拓扑；当前 A={a_lbl}, K={k_lbl}。",
                        "此结构中整流管应 A 接 GND，K 接阻容降压中间节点。",
                    )
                if a_id == mid and k_id == gnd:
                    return _scene_result(
                        "CAP_DROPPER_RECTIFIER", "ERROR", 0.95,
                        f"{target} 连接在阻容降压中间节点和 GND 之间，但 A/K 与整流规则相反；当前 A={a_lbl}, K={k_lbl}。",
                        "此结构中整流管应 A 接 GND，K 接阻容降压中间节点。",
                    )

            # D5：中间节点 <-> 输出正轨。此图结构中 A 接中间节点，K 接输出正轨/ACL/+5V。
            if frozenset({a_id, k_id}) == frozenset({mid, pos}):
                if a_id == mid and k_id == pos:
                    return _scene_result(
                        "CAP_DROPPER_RETURN_DIODE", "PASS", 0.95,
                        f"{target} 连接在阻容降压中间节点和输出正轨之间，符合反向半周续流/泄放二极管局部拓扑；当前 A={a_lbl}, K={k_lbl}。",
                        "此结构中续流/泄放管应 A 接阻容降压中间节点，K 接输出正轨/ACL/+V。",
                    )
                if a_id == pos and k_id == mid:
                    return _scene_result(
                        "CAP_DROPPER_RETURN_DIODE", "ERROR", 0.95,
                        f"{target} 连接在阻容降压中间节点和输出正轨之间，但 A/K 与续流/泄放规则相反；当前 A={a_lbl}, K={k_lbl}。",
                        "此结构中续流/泄放管应 A 接阻容降压中间节点，K 接输出正轨/ACL/+V。",
                    )

    # 3) 与电感/继电器/电机等并联的续流二极管。
    d_pair = _component_net_pair(comp)
    if d_pair is not None:
        for other in components:
            if other is comp:
                continue
            if not _component_type_is_inductive(other):
                continue
            if _component_net_pair(other) != d_pair:
                continue
            a_low = _net_has_ground_label(a_id, wire_graph)
            k_low = _net_has_ground_label(k_id, wire_graph)
            a_high = _net_has_power_label(a_id, wire_graph)
            k_high = _net_has_power_label(k_id, wire_graph)
            other_ref = _component_label(other)
            if a_low and k_high:
                return _scene_result(
                    "FLYBACK_DIODE", "PASS", 0.93,
                    f"{target} 与感性元件 {other_ref} 并联，当前 A={a_lbl}, K={k_lbl}，符合续流方向。",
                    "常见低侧开关续流管：A 接低侧/开关管侧，K 接电源正侧。",
                )
            if a_high and k_low:
                return _scene_result(
                    "FLYBACK_DIODE", "ERROR", 0.93,
                    f"{target} 与感性元件 {other_ref} 并联，但当前 A={a_lbl}, K={k_lbl} 与常见续流方向相反。",
                    "常见低侧开关续流管：A 接低侧/开关管侧，K 接电源正侧。",
                )
            return _scene_result(
                "FLYBACK_DIODE", "UNKNOWN", 0.70,
                f"{target} 与感性元件 {other_ref} 并联，但两端高低电位未可靠识别，暂不判定方向。",
                "需要识别电源正侧和低侧/开关管侧。",
            )

    # 4) 阻容降压图的几何兜底：当导线被切碎导致 D6/D5 未精确连到 mid/gnd/pos 时，
    # 只要整图已识别出阻容降压局部结构，就用 D5/D6 的 A/K 几何方向给出较低置信度判断。
    fallback = _geometry_cap_dropper_fallback(comp, components, wire_graph, tokens, target, a_lbl, k_lbl)
    if fallback is not None:
        return fallback

    return _scene_result("UNKNOWN", "UNKNOWN", 0.0, "未匹配到高置信度局部拓扑场景")


def _make_diode_report(
    comp: Component,
    target: str,
    a_name: str,
    k_name: str,
    scene: DiodeSceneResult,
    final_status: Optional[str] = None,
    final_message: Optional[str] = None,
) -> Dict[str, object]:
    return {
        "target": target,
        "ref": _clean_name(getattr(comp, "ref", None)),
        "class": comp.name,
        "anode_net": a_name,
        "cathode_net": k_name,
        "scene": scene.scene,
        "scene_status": scene.status,
        "final_status": final_status or scene.status,
        "confidence": scene.confidence,
        "reason": scene.reason,
        "expected": scene.expected,
        "message": final_message or scene.reason,
        "bbox": comp.bbox,
    }

def _find_series_neighbors_from_net(
    diode: Component,
    endpoint_net: Optional[int],
    components: List[Component],
) -> List[Tuple[Component, int]]:
    """查找从二极管某一端出发、经过一个两端元件连接到另一个网络的关系。

    例如：D1.K -> NET_19 -> R9 -> GND，
    则在 K 端返回 (R9, GND_NET_ID)。
    """
    if endpoint_net is None:
        return []

    out: List[Tuple[Component, int]] = []
    for other in components:
        if other is diode:
            continue
        if other.name.lower() not in TWO_PIN_COMPONENT_CLASSES:
            continue
        other_net = _other_net_of_component(other, endpoint_net)
        if other_net is not None:
            out.append((other, other_net))
    return out


def _format_series_neighbors(
    neighbors: Iterable[Tuple[Component, int]],
    net_names: Dict[int, str],
) -> str:
    parts = []
    for comp, other_net in neighbors:
        parts.append(f"{_component_label(comp)}->{net_name(other_net, net_names)}")
    return ",".join(parts)


class RuleChecker:
    """基于网络拓扑、OCR 文本与 A/K 关键点的二极管极性规则判断。

    改进点：
    1. 先判断二极管串联/并联关系：串联电阻、并联负载/三极管/MOS、电感同网规则。
    2. 再判断大致使用场景：LED 指示、稳压/钳位、Boost、Buck、续流/钳位、普通串联。
    3. 支持“中间隔着电阻/电感/电容等元器件”的弱推断：
       例如 LED 的 K 端不是直接 GND，而是 K -> R -> GND，也可判断为低侧路径。
    4. 网络名不足时仍保留 warning，不轻易 danger，避免误报。
    """

    def __init__(self, only_danger: bool = False):
        self.only_danger = only_danger
        # 保存每个二极管的场景判断结果，pipeline/app.py 可显示 PASS/ERROR/UNKNOWN。
        self.last_diode_reports: List[Dict[str, object]] = []

    def check(self, components: List[Component], wire_graph: WireGraph, tokens: List[OCRToken]) -> List[Fault]:
        faults: List[Fault] = []
        self.last_diode_reports = []

        for comp in components:
            cname = comp.name.lower()
            if cname not in DIODE_CLASSES:
                continue

            if _looks_like_non_diode_false_positive(comp, tokens):
                continue
            # 过滤 zener 假阳性，避免把 DZ1 局部框、D5/导线误判成一个新二极管。
            if _is_unreliable_zener_candidate(comp, tokens):
                continue
            if _is_duplicate_unlabeled_zener_candidate(comp, components, tokens):
                continue
            if _is_duplicate_inside_stronger_zener(comp, components, tokens):
                continue

            ref = _clean_name(getattr(comp, "ref", None)) or _infer_ref_from_ocr(comp, tokens)
            target = ref or _class_cn(comp.name)

            if not comp.has_polarity():
                scene = _scene_result("UNKNOWN", "UNKNOWN", 0.0, "未识别到完整 A/K 关键点")
                self.last_diode_reports.append(_make_diode_report(
                    comp, target, "未连接", "未连接", scene,
                    final_status="WARNING",
                    final_message=f"{target} 未识别到完整 A/K 关键点，无法判断极性。",
                ))
                faults.append(Fault(
                    level="warning",
                    fault_type="missing_keypoints",
                    target=target,
                    message=f"{target} 未识别到完整 A/K 关键点，无法判断极性。",
                    bbox=comp.bbox,
                    evidence={"class": comp.name, "ref": ref or ""},
                ))
                continue

            a_id = comp.nets.get("anode")
            k_id = comp.nets.get("cathode")
            a_name = net_name(a_id, wire_graph.net_names)
            k_name = net_name(k_id, wire_graph.net_names)

            a_pt = comp.keypoints.get("anode")
            k_pt = comp.keypoints.get("cathode")

            a_ctx = self._endpoint_context(comp, "A", a_id, a_name, a_pt, components, tokens, wire_graph.net_names)
            k_ctx = self._endpoint_context(comp, "K", k_id, k_name, k_pt, components, tokens, wire_graph.net_names)

            parallel_components = _find_parallel_components(comp, components)
            a_series_neighbors = _find_series_neighbors_from_net(comp, a_id, components)
            k_series_neighbors = _find_series_neighbors_from_net(comp, k_id, components)

            evidence = {
                "anode_net": a_name,
                "cathode_net": k_name,
                "class": comp.name,
                "ref": ref or "",
                "near_texts": ",".join(_near_texts(comp, tokens, pad=150)),
                "a_context": ",".join(sorted(a_ctx)),
                "k_context": ",".join(sorted(k_ctx)),
                "parallel_components": _format_component_list(parallel_components),
                "a_series_neighbors": _format_series_neighbors(a_series_neighbors, wire_graph.net_names),
                "k_series_neighbors": _format_series_neighbors(k_series_neighbors, wire_graph.net_names),
            }

            if a_id is None or k_id is None:
                scene = _scene_result("UNKNOWN", "UNKNOWN", 0.0, "A/K 至少有一个引脚没有匹配到导线网络")
                self.last_diode_reports.append(_make_diode_report(
                    comp, target, a_name, k_name, scene,
                    final_status="WARNING",
                    final_message=f"{target} 的 A/K 至少有一个引脚没有匹配到导线网络：A={a_name}, K={k_name}。",
                ))
                faults.append(Fault(
                    level="warning",
                    fault_type="pin_not_connected",
                    target=target,
                    message=f"{target} 的 A/K 至少有一个引脚没有匹配到导线网络：A={a_name}, K={k_name}。",
                    bbox=comp.bbox,
                    evidence=evidence,
                ))
                continue

            # LED 专用规则优先级最高。
            # 旧逻辑会先进入局部拓扑/串并联规则，可能被“与电阻/电感同网”等规则提前 PASS，
            # 导致 K=+3.3V、A=PWR_LED_K 这类 LED 反接没有报错。
            if cname == "led":
                scene = _scene_result(
                    "LED_INDICATOR", "UNKNOWN", 0.90,
                    "LED 专用规则优先判断：先检查 K 是否接电源、A 是否接 GND，以及 A/K 与限流电阻的真实拓扑关系。",
                    "LED 指示灯常见方向：A 接 +V/VCC/输出正侧，K 接限流电阻侧或低侧。",
                )
                f = self._check_led(comp, target, a_name, k_name, a_ctx, k_ctx, evidence)
                final_status = "PASS" if f is None else ("ERROR" if f.level == "danger" else "WARNING")
                final_message = "LED 专用规则未发现明确极性错误。" if f is None else f.message
                self.last_diode_reports.append(_make_diode_report(
                    comp, target, a_name, k_name, scene,
                    final_status=final_status,
                    final_message=final_message,
                ))
                if f is not None:
                    faults.append(f)
                continue

            # 先做“局部拓扑场景”判断。它比普通电源/地启发式更可靠，
            # 可避免把阻容降压中的 D5/D6 误判成普通二极管反接。
            scene = _detect_local_topology_scene(comp, components, wire_graph, tokens)
            evidence["local_scene"] = scene.scene
            evidence["local_scene_status"] = scene.status
            evidence["local_scene_confidence"] = f"{scene.confidence:.2f}"
            evidence["local_scene_reason"] = scene.reason
            evidence["local_scene_expected"] = scene.expected

            if scene.status == "PASS":
                self.last_diode_reports.append(_make_diode_report(comp, target, a_name, k_name, scene, final_status="PASS"))
                continue

            if scene.status == "ERROR":
                self.last_diode_reports.append(_make_diode_report(comp, target, a_name, k_name, scene, final_status="ERROR"))
                faults.append(Fault(
                    level="danger",
                    fault_type="diode_scene_polarity_error",
                    target=target,
                    message=f"{target} 局部拓扑判断为 {scene.scene}，极性错误。{scene.reason} {scene.expected}",
                    bbox=_expand_bbox(comp.bbox),
                    evidence=evidence,
                ))
                continue

            if cname == "led":
                f = self._check_led(comp, target, a_name, k_name, a_ctx, k_ctx, evidence)
            elif cname == "zener_diode" or _is_zener_like(comp, tokens):
                f = self._check_zener(comp, target, a_name, k_name, a_ctx, k_ctx, evidence)
            else:
                f = self._check_general_or_schottky(comp, target, a_name, k_name, a_ctx, k_ctx, evidence, tokens)

            final_status = "PASS" if f is None else ("ERROR" if f.level == "danger" else "WARNING")
            final_message = "未发现明确极性错误。" if f is None else f.message
            self.last_diode_reports.append(_make_diode_report(
                comp, target, a_name, k_name, scene,
                final_status=final_status,
                final_message=final_message,
            ))

            if f is not None:
                faults.append(f)

        if self.only_danger:
            faults = [f for f in faults if f.level == "danger"]

        return faults

    # --------------------------- 上下文推断 ---------------------------

    def _endpoint_context(
        self,
        comp: Component,
        pin_label: str,
        net_id: Optional[int],
        net: str,
        point: Optional[Point],
        components: List[Component],
        tokens: List[OCRToken],
        net_names: Dict[int, str],
    ) -> Set[str]:
        """推断一个端点的电气上下文。返回标签集合：power/ground/output/switch/series_to_ground 等。"""
        ctx: Set[str] = set()
        n = _clean_name(net)

        if _is_input_power_net(n):
            ctx.add("power")
        if _is_output_net(n):
            ctx.add("output")
            ctx.add("power_like")
        if _is_boost_switch_net(n):
            ctx.add("switch")
        if _is_low_or_ground_net(n):
            ctx.add("ground")
            ctx.add("low")

        # OCR 文本近邻辅助判断，半径不宜过大，否则 A/K 两端会同时扫到同一文字
        near_texts = _texts_near_point(point, tokens, radius=170)
        for text in near_texts:
            if _is_input_power_net(text):
                ctx.add("near_power")
                ctx.add("power")
            if _is_output_net(text):
                ctx.add("near_output")
                ctx.add("output")
                ctx.add("power_like")
            if _is_boost_switch_net(text):
                ctx.add("near_switch")
                ctx.add("switch")
            if _is_low_or_ground_net(text):
                ctx.add("near_ground")
                ctx.add("ground")
                ctx.add("low")

        # 如果端点附近有电阻/电感/电容，再看这些元件附近是否存在 GND 或电源文字。
        # 这是对 “D -> R -> GND” / “D -> R -> +V” 这类间接连接的弱推断。
        for other in components:
            if other is comp:
                continue
            if point is None:
                continue

            oname = other.name.lower()
            if oname not in {"resistor", "capacitor", "electrolytic_capacitor", "inductor", "power", "gnd"}:
                continue

            oc = _safe_center(other)
            if _dist(point, oc) > 260:
                continue

            if oname == "gnd":
                ctx.add("ground")
                ctx.add("low")
                continue
            if oname == "power":
                ctx.add("power")
                continue

            if oname == "resistor":
                ctx.add("near_resistor")
            elif oname in {"capacitor", "electrolytic_capacitor"}:
                ctx.add("near_capacitor")
            elif oname == "inductor":
                ctx.add("near_inductor")

            # 看该元件周围 OCR 是否有 GND / +V / OUT / SW 等信息
            around = _texts_near_point(oc, tokens, radius=360)
            if any(_is_low_or_ground_net(t) for t in around):
                ctx.add("series_to_ground")
                ctx.add("low")
            if any(_is_input_power_net(t) for t in around):
                ctx.add("series_to_power")
                ctx.add("power")
            if any(_is_output_net(t) for t in around):
                ctx.add("series_to_output")
                ctx.add("power_like")
            if any(_is_boost_switch_net(t) for t in around):
                ctx.add("series_to_switch")
                ctx.add("switch")

        # 基于真实网络拓扑的一跳串联推断：
        # D 端点所在网络 -> R/C/L 等两端元件 -> 另一个网络。
        # 这比单纯靠 OCR 距离更可靠，可用于判断 LED 串联电阻到 GND、
        # 稳压管并联电容到电源轨、保护二极管隔着负载到电源/地等情况。
        for other, other_net_id in _find_series_neighbors_from_net(comp, net_id, components):
            oname = other.name.lower()
            other_name = net_name(other_net_id, net_names)

            if oname == "resistor":
                ctx.add("via_resistor")
            elif oname in {"capacitor", "electrolytic_capacitor"}:
                ctx.add("via_capacitor")
            elif oname == "inductor":
                ctx.add("via_inductor")

            if _is_low_or_ground_net(other_name):
                ctx.add("series_to_ground")
                ctx.add("low")
            if _is_input_power_net(other_name):
                ctx.add("series_to_power")
                ctx.add("power")
            if _is_output_net(other_name):
                ctx.add("series_to_output")
                ctx.add("power_like")
            if _is_boost_switch_net(other_name):
                ctx.add("series_to_switch")
                ctx.add("switch")

        # 端点自身附近文本没有扫到时，比较两端到 GND/Power 文字的距离，给更近一侧打弱标签
        if point is not None:
            ground_d = _nearest_text_distance(point, tokens, _is_low_or_ground_net, max_radius=520)
            power_d = _nearest_text_distance(point, tokens, lambda t: _is_input_power_net(t) or _is_output_net(t), max_radius=520)
            switch_d = _nearest_text_distance(point, tokens, _is_boost_switch_net, max_radius=420)

            if ground_d < 260:
                ctx.add("near_ground")
            if power_d < 260:
                ctx.add("near_power")
            if switch_d < 220:
                ctx.add("near_switch")

        return ctx

    @staticmethod
    def _ctx_is_power(ctx: Set[str]) -> bool:
        return bool({"power", "power_like", "output", "near_power", "series_to_power", "series_to_output"} & ctx)

    @staticmethod
    def _ctx_is_ground(ctx: Set[str]) -> bool:
        return bool({"ground", "low", "near_ground", "series_to_ground"} & ctx)

    @staticmethod
    def _ctx_is_switch(ctx: Set[str]) -> bool:
        return bool({"switch", "near_switch", "series_to_switch"} & ctx)

    @staticmethod
    def _ctx_is_output(ctx: Set[str]) -> bool:
        return bool({"output", "near_output", "series_to_output"} & ctx)

    # --------------------------- 各类二极管规则 ---------------------------

    def _check_led(
        self,
        comp: Component,
        target: str,
        a_name: str,
        k_name: str,
        a_ctx: Set[str],
        k_ctx: Set[str],
        evidence: Dict[str, str],
    ) -> Optional[Fault]:
        """LED 指示灯极性判断。

        重要修正：
        旧版本把端点附近 170~520 像素内扫到的 +3.3V / OUT 文字也当成
        “该端直接接电源”，导致本图这种 LED：

            +3.3V -> LED(A) -> LED(K) -> R -> GND

        被误判成 “K 接电源”。

        现在把电源/地分成两级：
        - strong：网络本身命名为 +V/GND，或通过串联关系明确到 +V/GND；
        - weak：仅仅是 OCR 文字距离较近。

        只有 strong K-power 才会判 LED 阴极接电源错误。
        如果 K 端连接电阻，且 A 端疑似电源/输出侧，则认为是常见指示灯串联电阻接法。
        """
        evidence["scenario"] = "led_indicator"

        def _strong_power(ctx: Set[str], name: str) -> bool:
            # 直接网络名或拓扑一跳串联到电源，才算强电源。
            if _is_input_power_net(name) or _is_output_net(name):
                return True
            if "series_to_power" in ctx or "series_to_output" in ctx:
                return True
            # endpoint_context 中：直接网络名会加 power/output，但不会加 near_power/near_output；
            # OCR 近邻会同时加 near_power/near_output，所以这里排除 near_*。
            if "power" in ctx and "near_power" not in ctx:
                return True
            if "output" in ctx and "near_output" not in ctx:
                return True
            return False

        def _weak_power(ctx: Set[str], name: str) -> bool:
            return _strong_power(ctx, name) or "near_power" in ctx or "near_output" in ctx or "power_like" in ctx

        def _strong_ground(ctx: Set[str], name: str) -> bool:
            if _is_low_or_ground_net(name):
                return True
            if "series_to_ground" in ctx:
                return True
            if ("ground" in ctx or "low" in ctx) and "near_ground" not in ctx:
                return True
            return False

        def _weak_ground(ctx: Set[str], name: str) -> bool:
            return _strong_ground(ctx, name) or "near_ground" in ctx

        a_power_strong = _strong_power(a_ctx, a_name)
        k_power_strong = _strong_power(k_ctx, k_name)
        a_power_weak = _weak_power(a_ctx, a_name)
        k_power_weak = _weak_power(k_ctx, k_name)
        a_ground_strong = _strong_ground(a_ctx, a_name)
        k_ground_strong = _strong_ground(k_ctx, k_name)
        a_ground_weak = _weak_ground(a_ctx, a_name)
        k_ground_weak = _weak_ground(k_ctx, k_name)

        # LED 判断中只把真实拓扑一跳经过电阻 via_resistor 当成“限流电阻侧”。
        # near_resistor 只是图像距离近，不能用来抵消 K 接电源的强错误。
        a_has_resistor_path = "via_resistor" in a_ctx
        k_has_resistor_path = "via_resistor" in k_ctx
        a_near_resistor = "near_resistor" in a_ctx
        k_near_resistor = "near_resistor" in k_ctx

        evidence["led_a_power_strong"] = str(a_power_strong)
        evidence["led_k_power_strong"] = str(k_power_strong)
        evidence["led_a_power_weak"] = str(a_power_weak)
        evidence["led_k_power_weak"] = str(k_power_weak)
        evidence["led_a_ground_strong"] = str(a_ground_strong)
        evidence["led_k_ground_strong"] = str(k_ground_strong)
        evidence["led_a_ground_weak"] = str(a_ground_weak)
        evidence["led_k_ground_weak"] = str(k_ground_weak)
        evidence["led_a_has_resistor_path"] = str(a_has_resistor_path)
        evidence["led_k_has_resistor_path"] = str(k_has_resistor_path)
        evidence["led_a_near_resistor"] = str(a_near_resistor)
        evidence["led_k_near_resistor"] = str(k_near_resistor)

        # 强错误 1：A 端明确接 GND/低侧，且不是被误扫到的弱标签。
        if a_ground_strong and not a_power_strong:
            return Fault(
                "danger",
                "led_anode_to_ground",
                target,
                f"{target} LED 极性错误：LED 阳极 A 明确接到 GND/低侧；当前 A={a_name}, K={k_name}。",
                _expand_bbox(comp.bbox),
                evidence,
            )

        # 强错误 2：K 端明确接 +V/OUT 正侧，且 K 端不是经电阻去低侧的普通指示灯支路。
        # 注意：仅 near_power 不再触发这里，避免把靠近 +3.3V 文本的 K 端误判为接电源。
        if k_power_strong and not k_ground_strong and not k_has_resistor_path:
            return Fault(
                "danger",
                "led_cathode_to_power",
                target,
                f"{target} LED 极性错误：LED 阴极 K 明确接电源/输出正侧；当前 A={a_name}, K={k_name}。",
                _expand_bbox(comp.bbox),
                evidence,
            )

        # v20：LED 串联限流电阻的专用规则。
        # 你的指示灯模板按 +V/OUT -> LED(A) -> LED(K) -> R -> GND 判断。
        # 如果 A 端在电阻侧，而 K 端疑似在电源/输出侧，就是 LED 反接。
        # 这条规则专门解决 NET_75/NET_87 未完全命名时，K 接 +3.3V 但没有报错的问题。
        if a_has_resistor_path and not k_has_resistor_path and (k_power_strong or k_power_weak):
            return Fault(
                "danger",
                "led_anode_on_resistor_cathode_on_power",
                target,
                f"{target} LED 极性错误：检测到 A 端在限流电阻侧，而 K 端在电源/输出侧；当前 A={a_name}, K={k_name}。正确应为 A 接电源/输出侧，K 接限流电阻侧。",
                _expand_bbox(comp.bbox),
                evidence,
            )

        # 如果 topology.py 已经把网络补成 LED_RESISTOR_SIDE，即使 power_weak 没有触发，也按串联电阻侧规则判断。
        if _clean_name(a_name) == "LED_RESISTOR_SIDE" and _clean_name(k_name) != "LED_RESISTOR_SIDE":
            return Fault(
                "danger",
                "led_anode_on_resistor_side",
                target,
                f"{target} LED 极性错误：A 端被识别为限流电阻侧，方向与 A 接电源、K 接电阻的指示灯规则相反；当前 A={a_name}, K={k_name}。",
                _expand_bbox(comp.bbox),
                evidence,
            )

        # 常见正确 1：A 接电源/输出侧，K 直接或经电阻到低侧/GND。
        if a_power_weak and (k_ground_weak or k_has_resistor_path):
            return None

        # 常见正确 2：A 经电阻接电源，K 接 GND。
        # 例如 +V -> R -> LED(A) -> LED(K) -> GND。
        if (a_power_weak or a_has_resistor_path) and k_ground_weak and not k_power_strong:
            return None

        # 只有弱电源提示时不直接判反接，降低为复核。
        if k_power_weak and not k_power_strong:
            return Fault(
                "warning",
                "led_k_near_power_need_review",
                target,
                f"{target} 的 K 端附近有电源/输出文字，但未确认 K 端真正接到电源网络；当前 A={a_name}, K={k_name}。建议检查网络合并。",
                comp.bbox,
                evidence,
            )

        if a_power_weak and not (k_ground_weak or k_has_resistor_path):
            return Fault(
                "warning",
                "led_low_side_unknown",
                target,
                f"{target} 的 A 端疑似接电源/输出侧，但 K 端没有追踪到 GND 或串联电阻低侧路径；当前 A={a_name}, K={k_name}。",
                comp.bbox,
                evidence,
            )

        if (k_ground_weak or k_has_resistor_path) and not a_power_weak:
            return Fault(
                "warning",
                "led_high_side_unknown",
                target,
                f"{target} 的 K 端疑似为低侧/串联电阻侧，但 A 端没有追踪到电源/输出侧；当前 A={a_name}, K={k_name}。",
                comp.bbox,
                evidence,
            )

        return Fault(
            "warning",
            "led_unknown",
            target,
            f"{target} 已识别 A={a_name}, K={k_name}，但 LED 两端电源/地或串联电阻关系不足，暂不能可靠判断。",
            comp.bbox,
            evidence,
        )

    def _check_zener(
        self,
        comp: Component,
        target: str,
        a_name: str,
        k_name: str,
        a_ctx: Set[str],
        k_ctx: Set[str],
        evidence: Dict[str, str],
    ) -> Optional[Fault]:
        """稳压/齐纳二极管：常见钳位场景 A 接 GND/低侧，K 接信号/电源/被保护线。"""
        evidence["scenario"] = "zener_clamp"

        a_low = self._ctx_is_ground(a_ctx) or _is_low_or_ground_net(a_name)
        k_low = self._ctx_is_ground(k_ctx) or _is_low_or_ground_net(k_name)
        a_protected = self._ctx_is_power(a_ctx) or self._ctx_is_output(a_ctx) or _is_signal_or_protected_net(a_name)
        k_protected = self._ctx_is_power(k_ctx) or self._ctx_is_output(k_ctx) or _is_signal_or_protected_net(k_name)

        if a_low and k_protected:
            return None

        # 只有当 K 端确实是明确低侧/GND 时才报 danger。
        # 仅靠“附近有 GND 文字”的弱上下文容易把正常 DZ1 误报成反接。
        explicit_k_low = _is_low_or_ground_net(k_name)
        explicit_a_protected = _is_signal_or_protected_net(a_name)
        if a_protected and k_low and (explicit_k_low or explicit_a_protected):
            return Fault(
                "danger",
                "zener_reverse",
                target,
                f"{target} 疑似稳压/钳位方向接反：常见钳位连接应 K 接信号/电源侧、A 接 GND/低侧；当前 A={a_name}, K={k_name}。",
                _expand_bbox(comp.bbox),
                evidence,
            )

        return Fault(
            "warning",
            "zener_unknown",
            target,
            f"{target} 已识别 A={a_name}, K={k_name}，但未可靠追踪到 GND/电源/信号网络，无法直接判定稳压方向。",
            comp.bbox,
            evidence,
        )

    def _check_general_or_schottky(
        self,
        comp: Component,
        target: str,
        a_name: str,
        k_name: str,
        a_ctx: Set[str],
        k_ctx: Set[str],
        evidence: Dict[str, str],
        tokens: List[OCRToken],
    ) -> Optional[Fault]:
        a = _clean_name(a_name)
        k = _clean_name(k_name)

        # 并联元件信息只作为场景辅助，不单独作为反接依据。
        # 例如 zener 与电容并联，通常提示电源钳位/稳压；
        # diode 与电感并联，通常提示续流/钳位场景。
        parallel_text = evidence.get("parallel_components", "")
        if parallel_text:
            evidence["has_parallel_component"] = "True"
        else:
            evidence["has_parallel_component"] = "False"

        # 1. Boost 输出二极管场景
        boost_fault = self._check_boost_output_diode(comp, target, a, k, a_ctx, k_ctx, evidence)
        if boost_fault is not None:
            return boost_fault

        # 2. Buck 低侧续流二极管场景
        buck_fault = self._check_buck_freewheel_diode(comp, target, a, k, a_ctx, k_ctx, evidence)
        if buck_fault is not None:
            return buck_fault

        # 3. 电感/线圈/电机/继电器续流/钳位场景
        flyback_like = self._is_flyback_like(comp, tokens, a_name, k_name, a_ctx, k_ctx)
        if "inductor" in parallel_text:
            flyback_like = True
        evidence["flyback_like"] = str(flyback_like)

        if flyback_like:
            evidence["scenario"] = "flyback_or_clamp"
            k_high = self._ctx_is_power(k_ctx) or self._ctx_is_switch(k_ctx) or _is_input_power_net(k) or _is_boost_switch_net(k)
            a_low = self._ctx_is_ground(a_ctx) or self._ctx_is_switch(a_ctx) or _is_low_or_ground_net(a)

            a_high = self._ctx_is_power(a_ctx) or self._ctx_is_switch(a_ctx) or _is_input_power_net(a) or _is_boost_switch_net(a)
            k_low = self._ctx_is_ground(k_ctx) or _is_low_or_ground_net(k)

            if k_high and a_low:
                return None

            if a_high and k_low:
                return Fault(
                    "danger",
                    "flyback_diode_reverse",
                    target,
                    f"{target} 疑似续流/钳位二极管接反：此类二极管通常 K 接电源/高电位/开关节点侧，A 接低侧；当前 A={a_name}, K={k_name}。",
                    _expand_bbox(comp.bbox),
                    evidence,
                )

        # 4. 普通串联工作二极管场景
        series_fault = self._check_series_working_diode(comp, target, a, k, a_ctx, k_ctx, evidence)
        if series_fault is not None:
            return series_fault

        # 5. 直接或间接电源/地关系
        a_power = self._ctx_is_power(a_ctx) or _is_input_power_net(a) or _is_output_net(a)
        k_power = self._ctx_is_power(k_ctx) or _is_input_power_net(k) or _is_output_net(k)
        a_ground = self._ctx_is_ground(a_ctx) or _is_low_or_ground_net(a)
        k_ground = self._ctx_is_ground(k_ctx) or _is_low_or_ground_net(k)

        if a_power and k_ground:
            # 普通二极管从电源侧到低侧通常是合理的，但具体是否该导通仍需场景。给 warning 不直接判错。
            return Fault(
                "warning",
                "diode_power_to_ground_context_needed",
                target,
                f"{target} A 侧疑似电源/输出，K 侧疑似 GND/低侧。若这是普通串联导通二极管可能正常；若是钳位/续流用途则需要结合场景复核。当前 A={a_name}, K={k_name}。",
                comp.bbox,
                evidence,
            )

        if a_ground and k_power:
            return Fault(
                "danger",
                "diode_reverse",
                target,
                f"{target} 普通/肖特基二极管方向疑似错误：A 端疑似接 GND/低侧，K 端疑似接电源/输出侧；当前 A={a_name}, K={k_name}。若不是专门钳位/续流场景，通常为反接。",
                _expand_bbox(comp.bbox),
                evidence,
            )

        # 6. 网络名不足时不强行判错
        if _is_unknown_net(a_name) or _is_unknown_net(k_name):
            return Fault(
                "warning",
                "diode_unknown",
                target,
                f"{target} 已识别 A={a_name}, K={k_name}，但网络名或间接连接关系不足，暂不能可靠判断是否反接。建议完善 OCR 网络命名和导线拓扑。",
                comp.bbox,
                evidence,
            )

        return Fault(
            "warning",
            "diode_context_needed",
            target,
            f"{target} 已识别 A={a_name}, K={k_name}，但未匹配到 Boost、Buck、LED、稳压、续流或普通串联规则，建议人工复核。",
            comp.bbox,
            evidence,
        )

    def _check_boost_output_diode(
        self,
        comp: Component,
        target: str,
        a: str,
        k: str,
        a_ctx: Set[str],
        k_ctx: Set[str],
        evidence: Dict[str, str],
    ) -> Optional[Fault]:
        """Boost 输出二极管：正常 A->SOUT/SW/LX，K->OUT/VOUT。"""
        a_switch = _is_boost_switch_net(a) or self._ctx_is_switch(a_ctx)
        k_switch = _is_boost_switch_net(k) or self._ctx_is_switch(k_ctx)
        a_out = _is_output_net(a) or self._ctx_is_output(a_ctx)
        k_out = _is_output_net(k) or self._ctx_is_output(k_ctx)

        if a_switch and k_out:
            evidence["scenario"] = "boost_output_diode"
            return None

        if a_out and k_switch:
            evidence["scenario"] = "boost_output_diode"
            evidence["boost_rule"] = "A should connect to SOUT/SW/LX, K should connect to OUT/VOUT"
            return Fault(
                "danger",
                "boost_diode_reverse",
                target,
                f"{target} 疑似 Boost 输出二极管接反：Boost 中二极管应 A 接 SOUT/SW/LX 开关节点，K 接 OUT/VOUT 输出端；当前 A={a}, K={k}。",
                _expand_bbox(comp.bbox),
                evidence,
            )

        return None

    def _check_buck_freewheel_diode(
        self,
        comp: Component,
        target: str,
        a: str,
        k: str,
        a_ctx: Set[str],
        k_ctx: Set[str],
        evidence: Dict[str, str],
    ) -> Optional[Fault]:
        """Buck 低侧续流二极管：正常 A->GND/低侧，K->SW/LX。"""
        a_low = _is_low_or_ground_net(a) or self._ctx_is_ground(a_ctx)
        k_low = _is_low_or_ground_net(k) or self._ctx_is_ground(k_ctx)
        a_switch = _is_boost_switch_net(a) or self._ctx_is_switch(a_ctx)
        k_switch = _is_boost_switch_net(k) or self._ctx_is_switch(k_ctx)

        if a_low and k_switch:
            evidence["scenario"] = "buck_freewheel_diode"
            return None

        if a_switch and k_low:
            evidence["scenario"] = "buck_freewheel_diode"
            evidence["buck_rule"] = "freewheel diode should connect A to GND/low side and K to SW/LX"
            return Fault(
                "danger",
                "buck_freewheel_diode_reverse",
                target,
                f"{target} 疑似 Buck 低侧续流二极管接反：续流二极管通常 A 接 GND/低侧，K 接 SW/LX 开关节点；当前 A={a}, K={k}。",
                _expand_bbox(comp.bbox),
                evidence,
            )

        return None

    def _check_series_working_diode(
        self,
        comp: Component,
        target: str,
        a: str,
        k: str,
        a_ctx: Set[str],
        k_ctx: Set[str],
        evidence: Dict[str, str],
    ) -> Optional[Fault]:
        """普通串联工作二极管：常见正常方向 VIN/VCC -> A，K -> OUT/VOUT/负载。"""
        a_input = _is_input_power_net(a) or "power" in a_ctx or "series_to_power" in a_ctx
        k_input = _is_input_power_net(k) or "power" in k_ctx or "series_to_power" in k_ctx
        a_out = _is_output_net(a) or self._ctx_is_output(a_ctx)
        k_out = _is_output_net(k) or self._ctx_is_output(k_ctx)

        if a_input and k_out:
            evidence["scenario"] = "series_working_diode"
            return None

        if a_out and k_input:
            evidence["scenario"] = "series_working_diode"
            evidence["series_rule"] = "normal series diode usually A to input power, K to output/load"
            return Fault(
                "danger",
                "series_diode_reverse",
                target,
                f"{target} 疑似普通串联工作二极管接反：常见方向应 A 接 VIN/VCC 输入侧，K 接 OUT/VOUT 负载侧；当前 A={a}, K={k}。",
                _expand_bbox(comp.bbox),
                evidence,
            )

        return None

    def _is_flyback_like(
        self,
        comp: Component,
        tokens: List[OCRToken],
        a_name: str,
        k_name: str,
        a_ctx: Set[str],
        k_ctx: Set[str],
    ) -> bool:
        """判断是否像续流/钳位二极管环境。"""
        cx, cy = comp.center

        for t in tokens:
            tx, ty = t.center
            txt = _token_text(t)

            is_inductive_text = (
                INDUCTOR_TEXT_PAT.fullmatch(txt) is not None
                or txt in {"L", "COIL", "MOTOR", "RELAY", "K1", "K2", "INDUCTOR"}
            )

            if is_inductive_text and (tx - cx) ** 2 + (ty - cy) ** 2 < 320 ** 2:
                return True

        a = _clean_name(a_name)
        k = _clean_name(k_name)

        if (_is_low_or_ground_net(a) or self._ctx_is_ground(a_ctx)) and (
            _is_input_power_net(k) or _is_boost_switch_net(k) or self._ctx_is_power(k_ctx) or self._ctx_is_switch(k_ctx)
        ):
            return True

        if (_is_low_or_ground_net(k) or self._ctx_is_ground(k_ctx)) and (
            _is_input_power_net(a) or _is_boost_switch_net(a) or self._ctx_is_power(a_ctx) or self._ctx_is_switch(a_ctx)
        ):
            return True

        return False
