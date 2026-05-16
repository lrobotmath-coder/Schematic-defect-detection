from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple, Set

import numpy as np

from .ocr_reader import extract_net_keywords, normalize_text, get_last_image_path
from .types import Component, OCRToken, WireGraph
from .wire_extractor import WireExtractor


# -----------------------------------------------------------------------------
# 文本规范化与网络名判断
# -----------------------------------------------------------------------------

def _norm_text(text: str | None) -> str:
    """统一 OCR 文本，便于网络命名和规则判断。

    例：
    - "vbus (+5v)" -> "VBUS"
    - "＋3.3V" -> "+3.3V"
    - " gnd " -> "GND"
    """
    if text is None:
        return ""

    s = str(text).strip().upper()
    s = s.replace(" ", "")
    s = s.replace("＋", "+")
    s = s.replace("（", "(").replace("）", ")")
    s = s.replace("，", ",").replace("：", ":")

    # OCR 常见误识别：0V 被识别成 OV
    s = s.replace("OV", "0V")

    # VBUS(+5V)、VBUS (+5V) 等统一成 VBUS，避免正则匹配失败
    if "VBUS" in s:
        return "VBUS"

    return s


POWER_PAT = re.compile(
    r"^("
    r"VCC|VDD|VIN|VOUT|VBUS|VBAT|BAT|PWR|PWR_FLAG|"
    r"\+?\d+(?:\.\d+)?V|\+?3V3|"
    r"3V3"
    r")$",
    re.I,
)

GND_PAT = re.compile(r"^(GND|0V|AGND|DGND|PGND|VSS)$", re.I)

SIGNAL_NET_PAT = re.compile(
    r"^(OUT|SOUT|SW|LX|PHASE|DRAIN|IN|EN|LOAD|OUTPUT|ACL|ACLIN|ACL_IN|ACN|ACN_IN|LOW_SIDE|MOS_DRAIN|Q_DRAIN|SW_LOW|SWITCH_LOW|MOTOR\-|LOAD\-)$",
    re.I,
)

# 元件编号，不应该被当成网络名。例如 D1、R7、C18、Q1、U2、P1、J1、V1。
REF_PAT = re.compile(
    r"^(D\d+|DZ\d*\??|ZD\d*\??|TVS\d*|LED\d*|R\d+|C\d+|L\d+|U\d+|Q\d+|P\d+|J\d+|V\d+)$",
    re.I,
)

# 元件参数/型号，不能作为网络名。
# 例如 1N4007、470UF/25、10R/2W、222、1K、4.7UF 等。
VALUE_PAT = re.compile(
    r"^("
    r"\d+(\.\d+)?R(/\d+(\.\d+)?W)?|"
    r"\d+(\.\d+)?K|"
    r"\d+(\.\d+)?M|"
    r"\d+(\.\d+)?UF(/\d+(\.\d+)?V)?|"
    r"\d+(\.\d+)?NF|"
    r"\d+(\.\d+)?PF|"
    r"\d+(\.\d+)?V|"
    r"\d+N\d+|"
    r"1N\d+|"
    r"\d+"
    r")$",
    re.I,
)

# 明显像器件型号/参数的文本，不作为网络名。
MODEL_OR_VALUE_HINTS = {
    "1N", "UF", "NF", "PF", "OHM", "KOHM", "MOHM", "CBB", "X2", "OXAL", "SSY"
}

def _is_power_text(text: str | None) -> bool:
    return POWER_PAT.match(_norm_text(text)) is not None


def _is_ground_text(text: str | None) -> bool:
    return GND_PAT.match(_norm_text(text)) is not None


def _is_signal_text(text: str | None) -> bool:
    return SIGNAL_NET_PAT.match(_norm_text(text)) is not None


def _is_generic_net_label(text: str | None) -> bool:
    """允许用户自定义网络名，例如 ACL、ACN、W+、CE、I/RB。

    但排除 D5、R6、C5、1N4007、470UF/25、10R/2W 这类
    元件编号、器件型号或参数，避免污染 net_names。
    """
    s = _norm_text(text)

    if not s:
        return False

    if REF_PAT.match(s):
        return False

    if VALUE_PAT.match(s):
        return False

    # 斜杠文本很容易是 OCR 把多段文字粘在一起，例如 ACL/C/T、470UF/25V、10R/2W。
    # 为避免污染网络名，这里先全部不作为网络名；真实网络名建议用 ACL_IN、ACN_IN 这类形式。
    if "/" in s:
        return False

    # 以 + 开头但不是合法电源名的文本通常是 OCR 粘连，例如 +C1，不能作为网络名。
    if s.startswith("+"):
        return False

    # 太长的 IC 型号、参数字符串不要作为网络名。
    if len(s) > 12:
        return False

    # 至少包含一个字母，避免 222、5、3 这类数字进入网络名。
    if re.search(r"[A-Z]", s) is None:
        return False

    # 单个字母很容易是 OCR 噪声，例如 C/T/X/E/A，不作为网络名。
    if len(s) <= 1:
        return False

    # 排除明显型号/参数提示。
    if any(hint in s for hint in MODEL_OR_VALUE_HINTS):
        return False

    # 当前项目里不要让任意英文串污染网络名。
    # 自定义网络名只允许：带下划线、常见接口/信号名，或明显的网络前缀。
    allowed_custom = {
        "CE", "CS", "RST", "RESET", "KEY", "SDA", "SCL", "TX", "RX",
        "MOSI", "MISO", "SCK", "CLK", "FB", "EN", "IO", "GPIO",
        "W+", "W-", "L", "N", "LIVE", "NEUTRAL",
    }
    if s not in allowed_custom and "_" not in s and not re.match(r"^(IO|GPIO|NET)[A-Z0-9_+\-]*$", s):
        return False

    # 允许常见网络名字符：字母、数字、+、-、_。
    if re.fullmatch(r"[A-Z0-9_+\-]+", s) is None:
        return False

    return True


def _is_useful_net_text(text: str | None) -> bool:
    """允许电源/地/常见信号名，也允许 ACL/ACN/W+/CE 等自定义网络名。"""
    s = _norm_text(text)
    if not s:
        return False
    if REF_PAT.match(s):
        return False
    return (
        _is_power_text(s)
        or _is_ground_text(s)
        or _is_signal_text(s)
        or _is_generic_net_label(s)
    )


def _name_priority(name: str | None) -> int:
    """网络名优先级：GND > 电源 > 明确信号 > 自定义网络名 > 空。"""
    s = _norm_text(name)
    if _is_ground_text(s):
        return 4
    if _is_power_text(s):
        return 3
    if _is_signal_text(s):
        return 2
    if _is_generic_net_label(s):
        return 1
    return 0


def _choose_better_name(old_name: str | None, new_name: str) -> bool:
    old_p = _name_priority(old_name)
    new_p = _name_priority(new_name)
    if new_p > old_p:
        return True
    if new_p < old_p:
        return False

    # 同等级时：保留已有名字，避免 OUT/SOUT 被附近其他文本反复覆盖。
    return not _norm_text(old_name)


# -----------------------------------------------------------------------------
# 元件编号匹配辅助：避免 DZ1 被附近的 D5 文本错误命名
# -----------------------------------------------------------------------------

def _ref_prefix(text: str) -> str:
    s = _norm_text(text).replace("?", "")
    m = re.match(r"^([A-Z]+)", s)
    return m.group(1) if m else ""


def _ref_distance_limit(comp: Component) -> float:
    x1, y1, x2, y2 = comp.bbox
    return max(80.0, min(190.0, max(x2 - x1, y2 - y1) * 2.2))


def _ref_compatible_with_component(ref: str, comp: Component) -> bool:
    s = _norm_text(ref).replace("?", "")
    cname = comp.name.lower()
    if cname == "zener_diode":
        return s.startswith(("DZ", "ZD", "TVS"))
    if cname in {"diode", "schottky_diode", "led"}:
        return s.startswith("LED") or (s.startswith("D") and not s.startswith(("DZ", "ZD")))
    if cname in {"capacitor", "electrolytic_capacitor"}:
        return s.startswith("C")
    if cname == "resistor":
        return s.startswith("R")
    if cname == "inductor":
        return s.startswith("L")
    if cname in {"nmos", "pmos", "npn_transistor", "pnp_transistor"}:
        return s.startswith("Q")
    return True


# -----------------------------------------------------------------------------
# 稳压管 A/K 修正
# -----------------------------------------------------------------------------

def _is_zener_component(comp: Component) -> bool:
    """判断一个检测框是否是稳压/TVS 类二极管。"""
    cname = comp.name.lower()
    ref = _norm_text(getattr(comp, "ref", ""))
    return cname in {"zener_diode", "tvs"} or ref.startswith(("DZ", "ZD", "TVS"))


def correct_zener_keypoints(components: List[Component]) -> None:
    """修正 zener_diode 关键点顺序。

    你的数据中 DZ1 经常出现“图纸正确但一直报错”的现象，通常是
    YOLO-Pose 对稳压管符号的 A/K 标注顺序与普通二极管规则相反。
    这里对 zener_diode / DZ / ZD / TVS 统一交换 anode 与 cathode。

    该修正是幂等的：同一个 Component 只交换一次，避免 assign_nets 被多次调用时来回翻转。
    如果你的新模型已经把稳压管 A/K 标注完全修正，可把下面函数调用注释掉。
    """
    for comp in components:
        if not _is_zener_component(comp):
            continue
        if getattr(comp, "_zener_keypoints_corrected", False):
            continue
        if comp.keypoints.get("anode") is None or comp.keypoints.get("cathode") is None:
            continue
        comp.keypoints["anode"], comp.keypoints["cathode"] = comp.keypoints["cathode"], comp.keypoints["anode"]
        # 如果之前已经分配过网络，也一并交换，保证二次调用时仍一致。
        if "anode" in comp.nets or "cathode" in comp.nets:
            comp.nets["anode"], comp.nets["cathode"] = comp.nets.get("cathode"), comp.nets.get("anode")
        setattr(comp, "_zener_keypoints_corrected", True)


# -----------------------------------------------------------------------------
# OCR bbox 辅助点：不要只用 OCR 框中心点，因为文字通常在导线旁边。
# -----------------------------------------------------------------------------

def _token_points(token: OCRToken) -> List[Tuple[float, float]]:
    """返回 OCR 文本框的多个候选点，用于匹配最近导线网络。

    用中心点 + 四边中点 + 四角点，比只用 center 更容易把 +3.3V/GND/OUT
    绑定到旁边的导线网络。
    """
    pts: List[Tuple[float, float]] = []

    try:
        cx, cy = token.center
        pts.append((float(cx), float(cy)))
    except Exception:
        pass

    box = getattr(token, "bbox", None)
    if not box:
        return pts

    try:
        xs = [float(p[0]) for p in box]
        ys = [float(p[1]) for p in box]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

        pts.extend(
            [
                (cx, cy),
                (cx, y1),
                (cx, y2),
                (x1, cy),
                (x2, cy),
                (x1, y1),
                (x2, y1),
                (x1, y2),
                (x2, y2),
            ]
        )
    except Exception:
        pass

    # 去重，避免重复调用 nearest_net_id
    out: List[Tuple[float, float]] = []
    seen = set()
    for x, y in pts:
        key = (round(x, 2), round(y, 2))
        if key not in seen:
            seen.add(key)
            out.append((x, y))
    return out


def _nearest_net_for_token(labels: np.ndarray, token: OCRToken, radius: int) -> Optional[int]:
    """从 OCR 文本框多个点中找最近网络。"""
    best_nid: Optional[int] = None

    # 先用原半径找，找不到再用 1.35 倍半径补救。
    radii = [radius, int(radius * 1.35)]
    for r in radii:
        for pt in _token_points(token):
            nid = WireExtractor.nearest_net_id(labels, pt, radius=r)
            if nid is not None:
                return nid

    return best_nid



# -----------------------------------------------------------------------------
# OCR 兜底生成电感元件
# -----------------------------------------------------------------------------

INDUCTOR_REF_PAT = re.compile(r"^L\d+$", re.I)
INDUCTOR_VALUE_PAT = re.compile(r"(\d+(?:\.\d+)?(?:UH|ΜH|µH|MH|H)|INDUCTOR|COIL|CHOKE)", re.I)


def _token_bbox_rect(token: OCRToken) -> BBox:
    try:
        xs = [float(p[0]) for p in token.bbox]
        ys = [float(p[1]) for p in token.bbox]
        return (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
    except Exception:
        cx, cy = token.center
        return (int(cx - 10), int(cy - 8), int(cx + 10), int(cy + 8))


def _merge_bbox(a: BBox, b: BBox) -> BBox:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def _expand_rect(b: BBox, xpad: int, ypad: int) -> BBox:
    return (max(0, int(b[0] - xpad)), max(0, int(b[1] - ypad)), int(b[2] + xpad), int(b[3] + ypad))


def _bbox_center_xy(b: BBox) -> Tuple[float, float]:
    return ((float(b[0]) + float(b[2])) / 2.0, (float(b[1]) + float(b[3])) / 2.0)


def _bbox_distance(a: BBox, b: BBox) -> float:
    ax, ay = _bbox_center_xy(a)
    bx, by = _bbox_center_xy(b)
    return float(((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5)


def _has_near_component(components: List[Component], bbox: BBox, names: Set[str], max_dist: float = 130.0) -> bool:
    for c in components:
        if c.name.lower() not in names:
            continue
        if _bbox_distance(c.bbox, bbox) <= max_dist:
            return True
    return False


def synthesize_missing_inductors_from_ocr(components: List[Component], tokens: List[OCRToken]) -> None:
    """当 YOLO 没有识别到电感时，用 OCR 的 L2 / 15uH 等文字兜底生成 inductor。

    这不能替代重新训练 YOLO，但能让规则层先拿到 L2 元件，避免“与电感同网”规则完全失效。
    """
    if not tokens:
        return

    ref_tokens: List[OCRToken] = []
    value_tokens: List[OCRToken] = []
    for t in tokens:
        txt = _norm_text(getattr(t, "text", ""))
        if INDUCTOR_REF_PAT.fullmatch(txt):
            ref_tokens.append(t)
        elif INDUCTOR_VALUE_PAT.search(txt):
            value_tokens.append(t)

    # 已经有电感检测框时，不重复生成。
    for rt in ref_tokens:
        ref = _norm_text(rt.text)
        base = _token_bbox_rect(rt)
        if any(c.name.lower() == "inductor" and (_component_ref_name(c) == ref or _bbox_distance(c.bbox, base) < 160) for c in components):
            continue

        # L2 文字通常在电感符号上方/旁边，找到最近的 15uH/xxuH 参数文字一起扩框。
        box = base
        cx, cy = rt.center
        best_val = None
        best_d = 1e18
        for vt in value_tokens:
            vx, vy = vt.center
            d = ((vx - cx) ** 2 + (vy - cy) ** 2) ** 0.5
            if d < best_d and d <= 190:
                best_d = d
                best_val = vt
        if best_val is not None:
            box = _merge_bbox(box, _token_bbox_rect(best_val))

        # 横向电感居多，左右扩得比上下更大，便于 pin1/pin2 接到两侧导线。
        box = _expand_rect(box, xpad=90, ypad=45)
        if _has_near_component(components, box, {"inductor"}, max_dist=180):
            continue
        components.append(Component(cls_id=-1, name="inductor", bbox=box, conf=0.55, keypoints={}, nets={}, ref=ref))

    # 如果没有 L2 文字，但识别到了 15uH/10uH 等参数，也生成一个低置信度电感。
    if not ref_tokens:
        made = 0
        for vt in value_tokens:
            box = _expand_rect(_token_bbox_rect(vt), xpad=95, ypad=45)
            if _has_near_component(components, box, {"inductor"}, max_dist=180):
                continue
            components.append(Component(cls_id=-1, name="inductor", bbox=box, conf=0.42, keypoints={}, nets={}, ref="L_AUTO"))
            made += 1
            if made >= 3:
                break



# -----------------------------------------------------------------------------
# OCR 兜底生成负载/接插件元件，例如 P6 MOTOR
# -----------------------------------------------------------------------------

LOAD_REF_PAT = re.compile(r"^P\d+$", re.I)
LOAD_TEXT_PAT = re.compile(r"^(MOTOR|MOTO|LOAD|FAN|PUMP|RELAY|COIL|BUZZER|VALVE)$", re.I)


def synthesize_missing_loads_from_ocr(components: List[Component], tokens: List[OCRToken]) -> None:
    """当 YOLO 没有识别到 P6/负载时，用 OCR 的 P6 + MOTOR/MOTO 兜底生成 load。

    这样二极管与电机/负载并联时，rules.py 才能先进入“并联负载续流/反接保护”规则，
    不会因为缺少 P6 而一直输出 UNKNOWN。
    """
    if not tokens:
        return

    ref_tokens: List[OCRToken] = []
    value_tokens: List[OCRToken] = []
    for t in tokens:
        txt = _norm_text(getattr(t, "text", ""))
        if LOAD_REF_PAT.fullmatch(txt):
            ref_tokens.append(t)
        elif LOAD_TEXT_PAT.fullmatch(txt):
            value_tokens.append(t)

    made_refs: Set[str] = set()
    for rt in ref_tokens:
        ref = _norm_text(rt.text)
        if ref in made_refs:
            continue
        base = _token_bbox_rect(rt)
        if any(c.name.lower() in {"load", "connector", "terminal", "motor"} and (_component_ref_name(c) == ref or _bbox_distance(c.bbox, base) < 160) for c in components):
            continue

        box = base
        cx, cy = rt.center
        # P6 文字旁通常有 MOTOR/LOAD；把两者合并后扩框，覆盖接插件两端引脚。
        best_vals = []
        for vt in value_tokens:
            vx, vy = vt.center
            d = ((vx - cx) ** 2 + (vy - cy) ** 2) ** 0.5
            if d <= 260:
                best_vals.append(vt)
        for vt in best_vals:
            box = _merge_bbox(box, _token_bbox_rect(vt))

        # 接插件/电机符号通常竖向两端，横向和纵向都适度扩张。
        box = _expand_rect(box, xpad=95, ypad=95)
        components.append(Component(cls_id=-1, name="load", bbox=box, conf=0.46, keypoints={}, nets={}, ref=ref))
        made_refs.add(ref)

    # 如果没有识别到 P6，但识别到 MOTOR/MOTO，也生成一个低置信度负载。
    if not ref_tokens:
        made = 0
        for vt in value_tokens:
            box = _expand_rect(_token_bbox_rect(vt), xpad=100, ypad=90)
            if _has_near_component(components, box, {"load", "connector", "terminal", "motor"}, max_dist=190):
                continue
            components.append(Component(cls_id=-1, name="load", bbox=box, conf=0.38, keypoints={}, nets={}, ref="P_AUTO"))
            made += 1
            if made >= 2:
                break



# -----------------------------------------------------------------------------
# LED 编号兜底修正：YOLO 已检测到 LED，但 OCR 编号 D2 位于符号右侧时，
# 普通最近编号匹配可能会把它错命名为 D1。这里专门为 LED 再做一次更宽松的
# OCR 编号匹配，优先使用 D/LED 编号，而不是把 D2 当成普通文字丢掉。
# -----------------------------------------------------------------------------

LED_REF_PAT = re.compile(r"^(D\d+|LED\d*)$", re.I)
LED_HINT_PAT = re.compile(r"^(RED|GREEN|BLUE|YELLOW|WHITE|AMBER|POWERED[_\-]?ON|PWR[_\-]?LED|LED)$", re.I)


def _point_to_bbox_distance(pt: Tuple[float, float], box: BBox) -> float:
    """点到 bbox 的最短距离；点在框内时为 0。"""
    x, y = float(pt[0]), float(pt[1])
    x1, y1, x2, y2 = [float(v) for v in box]
    dx = max(x1 - x, 0.0, x - x2)
    dy = max(y1 - y, 0.0, y - y2)
    return float((dx * dx + dy * dy) ** 0.5)


def _fix_led_refs_from_ocr(components: List[Component], tokens: List[OCRToken]) -> None:
    """修正 LED 的 ref，例如把右侧指示灯从 D1 改为 D2。

    你的图里 D2 文字在 LED 符号右侧，而 YOLO 只给了 led 类别；普通 ref 匹配
    可能会选到别处的 D1。这里对 led 元件使用 bbox 距离 + 方向偏置：
    - D/LED 编号在 LED 右侧、上方或紧邻符号时优先；
    - 太远的 D1/D5/D6 不会被抢过来；
    - 如果已经有其他二极管很近地使用了同一个 ref，则不强行抢占。
    """
    if not tokens:
        return

    ref_tokens: List[Tuple[str, OCRToken]] = []
    hint_tokens: List[OCRToken] = []
    for t in tokens:
        txt = _norm_text(getattr(t, "text", "")).replace("?", "")
        if LED_REF_PAT.fullmatch(txt):
            ref_tokens.append((txt, t))
        elif LED_HINT_PAT.fullmatch(txt):
            hint_tokens.append(t)

    if not ref_tokens:
        return

    for comp in components:
        if comp.name.lower() != "led":
            continue

        x1, y1, x2, y2 = [float(v) for v in comp.bbox]
        cx, cy = comp.center
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)

        # 有颜色/Powered_On 等文字时，说明这个 LED 的编号常在文字旁边，适当放宽范围。
        has_hint_near = False
        for ht in hint_tokens:
            if _point_to_bbox_distance(ht.center, comp.bbox) <= max(180.0, 2.2 * max(w, h)):
                has_hint_near = True
                break

        max_dist = max(210.0, 2.8 * max(w, h)) if has_hint_near else max(170.0, 2.2 * max(w, h))

        best_ref = None
        best_score = 1e18
        for ref, tok in ref_tokens:
            tx, ty = tok.center
            d_box = _point_to_bbox_distance((tx, ty), comp.bbox)
            d_center = ((float(tx) - float(cx)) ** 2 + (float(ty) - float(cy)) ** 2) ** 0.5
            if d_box > max_dist and d_center > max_dist:
                continue

            score = min(d_box, d_center * 0.72)

            # LED 编号经常位于符号右侧/右上侧。你的 D2 就是这种情况。
            if tx >= x1 - 20 and tx <= x2 + 230 and y1 - 110 <= ty <= y2 + 140:
                score -= 45.0
            if tx >= x2 - 5 and abs(ty - cy) <= max(90.0, h * 2.0):
                score -= 35.0
            # LED1 这类显式编号优先于普通 D1/D2。
            if ref.startswith("LED"):
                score -= 30.0

            if score < best_score:
                best_score = score
                best_ref = ref

        if best_ref:
            comp.ref = best_ref


def synthesize_missing_leds_from_ocr(components: List[Component], tokens: List[OCRToken]) -> None:
    """当 YOLO 漏检 LED 时，用 OCR 的 D2/LED + RED/Powered_On 生成低置信度 LED。

    注意：没有 A/K 关键点时，该兜底 LED 不能做严格极性判断；它主要用于
    元件列表显示和后续人工复核提示。若 YOLO 已经检测到 led，则只修正 ref。
    """
    if not tokens:
        return

    ref_tokens: List[Tuple[str, OCRToken]] = []
    hint_tokens: List[OCRToken] = []
    for t in tokens:
        txt = _norm_text(getattr(t, "text", "")).replace("?", "")
        if LED_REF_PAT.fullmatch(txt):
            ref_tokens.append((txt, t))
        elif LED_HINT_PAT.fullmatch(txt):
            hint_tokens.append(t)

    if not ref_tokens or not hint_tokens:
        return

    for ref, rt in ref_tokens:
        base = _token_bbox_rect(rt)
        if any(c.name.lower() in {"led", "diode", "schottky_diode", "zener_diode"} and (_bbox_distance(c.bbox, base) < 180 or _component_ref_name(c) == ref) for c in components):
            continue

        box = base
        rx, ry = rt.center
        matched_hint = False
        for ht in hint_tokens:
            hx, hy = ht.center
            d = ((hx - rx) ** 2 + (hy - ry) ** 2) ** 0.5
            if d <= 230:
                box = _merge_bbox(box, _token_bbox_rect(ht))
                matched_hint = True

        if not matched_hint:
            continue

        box = _expand_rect(box, xpad=95, ypad=60)
        if _has_near_component(components, box, {"led", "diode", "schottky_diode"}, max_dist=160):
            continue
        components.append(Component(cls_id=-1, name="led", bbox=box, conf=0.30, keypoints={}, nets={}, ref=ref))


# -----------------------------------------------------------------------------
# 组件去重与绿色导线兜底提取
# -----------------------------------------------------------------------------

def _bbox_area(b: BBox) -> float:
    x1, y1, x2, y2 = b
    return float(max(1, x2 - x1) * max(1, y2 - y1))


def _bbox_intersection(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    return float(max(0, ix2 - ix1) * max(0, iy2 - iy1))


def _bbox_iou(a: BBox, b: BBox) -> float:
    inter = _bbox_intersection(a, b)
    if inter <= 0:
        return 0.0
    return inter / max(1.0, _bbox_area(a) + _bbox_area(b) - inter)


def _bbox_min_overlap(a: BBox, b: BBox) -> float:
    inter = _bbox_intersection(a, b)
    if inter <= 0:
        return 0.0
    return inter / max(1.0, min(_bbox_area(a), _bbox_area(b)))


def _component_family_for_dedup(comp: Component) -> str:
    name = comp.name.lower()
    if name in {"capacitor", "electrolytic_capacitor"}:
        return "capacitor"
    if name in {"diode", "schottky_diode", "zener_diode", "led"}:
        return "diode"
    if name in {"resistor"}:
        return "resistor"
    if name in {"inductor"}:
        return "inductor"
    if name in {"load", "connector", "terminal", "motor"}:
        return "load"
    return name


def _dedup_score(comp: Component) -> float:
    score = float(getattr(comp, "conf", 0.0) or 0.0)
    ref = _component_ref_name(comp)
    if ref:
        score += 0.08
    # 电解电容比普通 capacitor 更有信息，重叠时略优先保留。
    if comp.name.lower() == "electrolytic_capacitor":
        score += 0.04
    return score


def _same_component_candidate(a: Component, b: Component) -> bool:
    fam_a = _component_family_for_dedup(a)
    fam_b = _component_family_for_dedup(b)
    if fam_a != fam_b:
        return False

    ref_a = _component_ref_name(a)
    ref_b = _component_ref_name(b)
    min_ov = _bbox_min_overlap(a.bbox, b.bbox)
    iou = _bbox_iou(a.bbox, b.bbox)

    if ref_a and ref_b and ref_a == ref_b:
        if min_ov >= 0.18 or _bbox_distance(a.bbox, b.bbox) <= 80:
            return True

    # 同类强重叠认为是同一个元件的重复检测。
    if min_ov >= 0.62 or iou >= 0.42:
        return True

    return False


def deduplicate_components_inplace(components: List[Component]) -> None:
    """删除 YOLO/OCR 兜底造成的重复元件框。

    例如 L1 被检测成 4 个重叠电感框，D1 被检测成两个 Schottky 框，
    只保留置信度/编号更可靠的一个。这样界面蓝框会少很多，规则也不会重复判断。
    """
    if not components:
        return

    used = [False] * len(components)
    kept: List[Component] = []

    for i, comp in enumerate(components):
        if used[i]:
            continue
        group = [i]
        used[i] = True
        changed = True
        while changed:
            changed = False
            for j, other in enumerate(components):
                if used[j]:
                    continue
                if any(_same_component_candidate(components[k], other) for k in group):
                    group.append(j)
                    used[j] = True
                    changed = True

        best_idx = max(group, key=lambda idx: _dedup_score(components[idx]))
        kept.append(components[best_idx])

    # 保持大致从左到右、从上到下的显示顺序。
    kept.sort(key=lambda c: (c.bbox[1], c.bbox[0]))
    components[:] = kept


def _wire_graph_has_nets(wire_graph: WireGraph) -> bool:
    try:
        if int(getattr(wire_graph, "num_nets", 0) or 0) > 0:
            return True
        labels = getattr(wire_graph, "label_image", None)
        if labels is not None and hasattr(labels, "max") and int(labels.max()) > 0:
            return True
    except Exception:
        pass
    return False


def _ensure_wire_graph_from_image_if_empty(wire_graph: WireGraph) -> None:
    """当 wire_extractor 没有提取到任何网络时，用原图颜色做兜底导线提取。

    你当前这张电源图使用绿色导线，而旧的 wire_extractor 往往只适配蓝色导线，
    结果网络数量为 0。这里只在 num_nets==0 时启用，不影响原来能正常提取蓝色导线的图。
    """
    if _wire_graph_has_nets(wire_graph):
        return

    img_path = get_last_image_path()
    if not img_path:
        return

    try:
        import cv2  # type: ignore
        img = cv2.imread(str(img_path))
        if img is None:
            return
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # 绿色导线：KiCad/示例图常见颜色。过滤掉浅灰网格和红/黄器件边框。
        green = ((h >= 35) & (h <= 95) & (s >= 45) & (v >= 45)).astype("uint8") * 255

        # 形态学连接断点。不要膨胀过大，否则不同网络容易粘连。
        k1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(green, cv2.MORPH_CLOSE, k1, iterations=1)
        mask = cv2.dilate(mask, k1, iterations=1)

        n, labels, stats, _cent = cv2.connectedComponentsWithStats(mask, connectivity=8)
        new_labels = np.zeros(labels.shape, dtype=np.int32)
        new_id = 0
        for lab in range(1, n):
            x, y, w, hh, area = stats[lab]
            # 排除小噪声和短文字笔画，只保留线状/较长连接区域。
            if area < 18:
                continue
            if max(w, hh) < 18:
                continue
            # 过小且不像线的区域，多半是文字碎片。
            if area < 50 and min(w, hh) > 10:
                continue
            new_id += 1
            new_labels[labels == lab] = new_id

        if new_id > 0:
            wire_graph.label_image = new_labels
            wire_graph.num_nets = int(new_id)
            if getattr(wire_graph, "net_names", None) is None:
                wire_graph.net_names = {}
            if getattr(wire_graph, "net_aliases", None) is None:
                wire_graph.net_aliases = {}
    except Exception:
        # 兜底失败时保留原结果，不让主流程崩溃。
        return

# -----------------------------------------------------------------------------
# 元件编号与网络分配
# -----------------------------------------------------------------------------

def assign_component_refs(components: List[Component], tokens: List[OCRToken]) -> None:
    """把 D1/R1/C2/Q1 等 OCR 文字匹配给最近元件。

    这里不要简单地把最近的任意 ref 绑定给元件。否则在你的图里，
    DZ1 的 OCR 可能识别成 DZ?，而附近的 D5 更容易被绑定到 DZ1 上。
    修改后会按元件类别筛选 ref：zener_diode 优先 DZ/ZD/TVS，
    普通 diode 优先 D/LED，电阻只匹配 R，电容只匹配 C。
    """
    # 先用 OCR 兜底补出 YOLO 漏检的电感/负载/LED，例如 L2 / 15uH、P6 / MOTOR、D2 / RED。
    synthesize_missing_inductors_from_ocr(components, tokens)
    synthesize_missing_loads_from_ocr(components, tokens)
    synthesize_missing_leds_from_ocr(components, tokens)

    ref_tokens = []
    for t in tokens:
        txt = _norm_text(getattr(t, "text", ""))
        if REF_PAT.match(txt):
            ref_tokens.append(t)

    for comp in components:
        cx, cy = comp.center
        best = None
        best_d2 = 1e18
        limit = _ref_distance_limit(comp)

        # 第一轮：只使用与元件类别兼容的编号。
        for t in ref_tokens:
            txt = _norm_text(t.text)
            if not _ref_compatible_with_component(txt, comp):
                continue
            tx, ty = t.center
            d2 = (tx - cx) ** 2 + (ty - cy) ** 2
            if d2 < best_d2 and d2 <= limit ** 2:
                best_d2 = d2
                best = txt.replace("?", "")

        # 不再做“任意最近 ref”兜底。
        # 原先这里会把 DZ1 误绑定给 C1/C3，把 D5 误绑定给 zener 假框，
        # 进而让规则模块把假检测框当成真实 D5/DZ1 去判断。
        # 现在所有元件编号都必须与元件类别兼容：
        #   zener 只接收 DZ/ZD/TVS；diode 只接收 D/LED；
        #   capacitor 只接收 C；resistor 只接收 R；inductor 只接收 L。
        if best is not None:
            comp.ref = best

    # LED 的编号经常在符号右侧，普通最近匹配会偶发绑定到错误 D 编号；
    # 最后再单独修正一次，解决右侧 D2 LED 被显示成 D1 的问题。
    _fix_led_refs_from_ocr(components, tokens)

    # 去除同一个元件的多重重叠检测框，例如 L1 出现 4 个框、D1 出现 2 个框。
    deduplicate_components_inplace(components)


TWO_PIN_COMPONENT_CLASSES = {
    "resistor",
    "capacitor",
    "electrolytic_capacitor",
    "inductor",
    # OCR 兜底生成的两端负载/接插件，例如 P6 MOTOR。
    # 这些不是极性元件，只是为了让 rules.py 能判断二极管与负载是否并联。
    "load",
    "connector",
    "terminal",
    "motor",
}


def _clip_point_to_image(point: Tuple[float, float], shape: Tuple[int, int]) -> Tuple[float, float]:
    """把候选端点限制到 label_image 范围内。"""
    h, w = shape[:2]
    x, y = point
    x = min(max(float(x), 0.0), float(w - 1))
    y = min(max(float(y), 0.0), float(h - 1))
    return x, y


def _nearest_net_from_candidates(
    labels: np.ndarray,
    points: List[Tuple[float, float]],
    radius: int,
) -> Optional[int]:
    """从多个候选端点中选择最可靠的导线网络。

    旧逻辑遇到第一个可命中的 NET 就返回，容易把二极管/电容端点接到
    元件框擦除后残留的小碎片网络，导致 D6 这类下方二极管无法接到真实 GND 轨。
    现在会统计所有候选点在多个半径下命中的网络，优先选择：
    1) 被多个候选点重复命中的网络；
    2) label_image 中面积更大的网络；
    3) 较小搜索半径即可命中的网络。
    """
    if labels is None or not points:
        return None

    search_radii = [radius, int(radius * 1.35), int(radius * 1.75), int(radius * 2.25)]
    shape = labels.shape[:2]
    scores: Dict[int, float] = {}

    for ri, r in enumerate(search_radii):
        radius_bonus = max(0.0, 3.0 - ri * 0.65)
        for pt in points:
            pt = _clip_point_to_image(pt, shape)
            nid = WireExtractor.nearest_net_id(labels, pt, radius=r)
            if nid is None:
                continue
            nid = int(nid)
            area_score = min(_net_area(labels, nid), 6000) / 1000.0
            scores[nid] = scores.get(nid, 0.0) + 10.0 + radius_bonus + area_score

    if not scores:
        return None

    return max(scores.items(), key=lambda kv: kv[1])[0]


def _nearest_net_id_excluding(
    labels: np.ndarray,
    point: Tuple[float, float],
    radius: int,
    exclude_nets: Set[int],
) -> Optional[int]:
    """在 point 周围找最近网络，但排除 exclude_nets。

    用途：二极管 A/K 被同一个大网络吸附时，例如 D5 显示 A_net=SW、K_net=SW，
    需要从端点外侧继续寻找另一个真实网络，比如 GND。
    """
    if labels is None:
        return None

    h, w = labels.shape[:2]
    x, y = _clip_point_to_image(point, labels.shape[:2])
    xi, yi = int(round(x)), int(round(y))
    r = int(max(1, radius))

    x1, x2 = max(0, xi - r), min(w - 1, xi + r)
    y1, y2 = max(0, yi - r), min(h - 1, yi + r)
    if x2 < x1 or y2 < y1:
        return None

    win = labels[y1:y2 + 1, x1:x2 + 1]
    mask = win > 0
    for ex in exclude_nets or set():
        mask &= win != int(ex)

    if not np.any(mask):
        return None

    ys, xs = np.where(mask)
    xs = xs + x1
    ys = ys + y1
    d2 = (xs.astype(float) - x) ** 2 + (ys.astype(float) - y) ** 2
    best_i = int(np.argmin(d2))
    return int(labels[int(ys[best_i]), int(xs[best_i])])


def _nearest_net_from_candidates_excluding(
    labels: np.ndarray,
    points: List[Tuple[float, float]],
    radius: int,
    exclude_nets: Set[int],
) -> Optional[int]:
    """从候选点中寻找一个不属于 exclude_nets 的网络。

    候选点本身已经按“从元件外侧较远处到较近处”排序，
    所以这里优先相信远离元件本体的点，避免再次吸到二极管符号内部残留网络。
    """
    if labels is None or not points:
        return None

    shape = labels.shape[:2]
    search_radii = [radius, int(radius * 1.35), int(radius * 1.75)]
    scores: Dict[int, float] = {}

    for pi, pt in enumerate(points):
        pt = _clip_point_to_image(pt, shape)
        # 越靠前的候选点越远离元件本体，权重略高。
        order_bonus = max(0.0, 4.0 - pi * 0.12)
        for ri, r in enumerate(search_radii):
            nid = _nearest_net_id_excluding(labels, pt, r, exclude_nets)
            if nid is None:
                continue
            area_score = min(_net_area(labels, nid), 5000) / 1000.0
            radius_bonus = max(0.0, 2.0 - ri * 0.55)
            scores[nid] = scores.get(nid, 0.0) + 10.0 + order_bonus + radius_bonus + area_score

    if not scores:
        return None
    return max(scores.items(), key=lambda kv: kv[1])[0]


def _repair_same_net_polar_component(labels: np.ndarray, comp: Component, radius: int = 70) -> None:
    """修复有极性元件 A/K 被吸附到同一网络的问题。

    常见现象：开关电源续流二极管 D5 一端接 SW，另一端接 GND，
    但由于二极管符号本身是蓝色线条，导线提取/最近网络吸附会把 A/K 都接到 SW。
    该函数在 A_net == K_net 时，沿 A/K 端点外侧继续搜索“非当前网络”的候选网络。
    """
    if labels is None:
        return

    if comp.name.lower() not in {"diode", "zener_diode", "schottky_diode", "led"}:
        return

    a_net = comp.nets.get("anode")
    k_net = comp.nets.get("cathode")
    if a_net is None or k_net is None or int(a_net) != int(k_net):
        return

    a_pt = comp.keypoints.get("anode")
    k_pt = comp.keypoints.get("cathode")
    if a_pt is None or k_pt is None:
        return

    same = int(a_net)
    a_alt = _nearest_net_from_candidates_excluding(
        labels, _infer_polar_pin_candidate_points(comp, "anode"), radius=radius, exclude_nets={same}
    )
    k_alt = _nearest_net_from_candidates_excluding(
        labels, _infer_polar_pin_candidate_points(comp, "cathode"), radius=radius, exclude_nets={same}
    )

    if a_alt is None and k_alt is None:
        return

    ax, ay = float(a_pt[0]), float(a_pt[1])
    kx, ky = float(k_pt[0]), float(k_pt[1])

    # 竖直二极管最常见：上端接 SW/+V，下端接 GND/低侧。
    # 如果 A/K 都吸到同一网络，优先把“下方端点”改成搜索到的非同网。
    if abs(ay - ky) >= abs(ax - kx):
        if ay >= ky:
            if a_alt is not None:
                comp.nets["anode"] = int(a_alt)
            elif k_alt is not None:
                comp.nets["cathode"] = int(k_alt)
        else:
            if k_alt is not None:
                comp.nets["cathode"] = int(k_alt)
            elif a_alt is not None:
                comp.nets["anode"] = int(a_alt)
        return

    # 横向二极管没有固定上下关系，优先修正有替代网络的一侧。
    if a_alt is not None and k_alt is None:
        comp.nets["anode"] = int(a_alt)
    elif k_alt is not None and a_alt is None:
        comp.nets["cathode"] = int(k_alt)
    elif a_alt is not None and k_alt is not None:
        # 两侧都有候选时，按端点方向选更外侧的一侧；默认先修 anode。
        comp.nets["anode"] = int(a_alt)


def _infer_two_pin_candidate_points(comp: Component) -> Dict[str, List[Tuple[float, float]]]:
    """根据 bbox 为两端元件估计 pin1/pin2 候选端点。

    关键修改：候选点优先放在 bbox 外侧较远的位置。因为 wire_extractor 会把
    元件框内部擦掉，如果仍然只在框边缘找，容易匹配到被擦剩的小碎片 NET，
    造成 C5/R6/D6 等元件端点不是同一个真实网络。
    """
    x1, y1, x2, y2 = [float(v) for v in comp.bbox]
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    cx, cy = comp.center
    cx, cy = float(cx), float(cy)

    # 从远到近搜索，优先命中元件外侧真实导线，而不是 bbox 内残留碎线。
    offs = [36.0, 26.0, 16.0, 8.0, 0.0]

    if w >= h:
        left = []
        right = []
        for off in offs:
            left.extend([
                (x1 - off, cy),
                (x1 - off, y1 + h * 0.35),
                (x1 - off, y2 - h * 0.35),
            ])
            right.extend([
                (x2 + off, cy),
                (x2 + off, y1 + h * 0.35),
                (x2 + off, y2 - h * 0.35),
            ])
        return {"pin1": left, "pin2": right}

    top = []
    bottom = []
    for off in offs:
        top.extend([
            (cx, y1 - off),
            (x1 + w * 0.35, y1 - off),
            (x2 - w * 0.35, y1 - off),
        ])
        bottom.extend([
            (cx, y2 + off),
            (x1 + w * 0.35, y2 + off),
            (x2 - w * 0.35, y2 + off),
        ])
    return {"pin1": top, "pin2": bottom}


def _infer_polar_pin_candidate_points(comp: Component, pin_name: str) -> List[Tuple[float, float]]:
    """为二极管 A/K 端点生成候选连接点。

    YOLO pose 的 A/K 点常落在符号内部或被元件框擦除区域内。
    所以这里根据“元件中心 -> 关键点”的方向，把候选点向 bbox 外侧延伸，
    优先匹配外侧真实导线。
    """
    pt = comp.keypoints.get(pin_name)
    if pt is None:
        return []

    x1, y1, x2, y2 = [float(v) for v in comp.bbox]
    cx, cy = comp.center
    px, py = float(pt[0]), float(pt[1])
    vx, vy = px - float(cx), py - float(cy)

    # 如果关键点几乎在中心，退化为原点附近搜索。
    if abs(vx) < 1e-3 and abs(vy) < 1e-3:
        return [(px, py)]

    # 判断主要方向，生成 bbox 外侧点。
    pts: List[Tuple[float, float]] = []
    offs = [44.0, 32.0, 22.0, 12.0, 0.0]
    if abs(vx) >= abs(vy):
        # 横向二极管：左端或右端。
        if vx >= 0:
            for off in offs:
                pts.extend([(x2 + off, py), (x2 + off, cy), (x2 + off, y1 + (y2 - y1) * 0.5)])
        else:
            for off in offs:
                pts.extend([(x1 - off, py), (x1 - off, cy), (x1 - off, y1 + (y2 - y1) * 0.5)])
    else:
        # 纵向二极管：上端或下端。
        if vy >= 0:
            for off in offs:
                pts.extend([(px, y2 + off), (cx, y2 + off), (x1 + (x2 - x1) * 0.5, y2 + off)])
        else:
            for off in offs:
                pts.extend([(px, y1 - off), (cx, y1 - off), (x1 + (x2 - x1) * 0.5, y1 - off)])

    pts.append((px, py))
    return pts



def _infer_two_pin_candidate_groups(comp: Component) -> List[Dict[str, List[Tuple[float, float]]]]:
    """同时生成横向和纵向两种两端候选。

    有些电容检测框接近正方形，单靠 bbox 宽高会把 C5 误判成竖向元件；
    这里同时评估横向/纵向，用命中的网络面积和两端是否不同来选更合理的一组。
    """
    x1, y1, x2, y2 = [float(v) for v in comp.bbox]
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    cx, cy = comp.center
    cx, cy = float(cx), float(cy)
    offs = [44.0, 34.0, 24.0, 14.0, 6.0, 0.0]

    left, right, top, bottom = [], [], [], []
    for off in offs:
        left.extend([(x1 - off, cy), (x1 - off, y1 + h * 0.35), (x1 - off, y2 - h * 0.35)])
        right.extend([(x2 + off, cy), (x2 + off, y1 + h * 0.35), (x2 + off, y2 - h * 0.35)])
        top.extend([(cx, y1 - off), (x1 + w * 0.35, y1 - off), (x2 - w * 0.35, y1 - off)])
        bottom.extend([(cx, y2 + off), (x1 + w * 0.35, y2 + off), (x2 - w * 0.35, y2 + off)])

    return [
        {"pin1": left, "pin2": right, "orientation": [(-1.0, 0.0), (1.0, 0.0)]},
        {"pin1": top, "pin2": bottom, "orientation": [(0.0, -1.0), (0.0, 1.0)]},
    ]


def _net_area(labels: np.ndarray, nid: Optional[int]) -> int:
    if nid is None:
        return 0
    try:
        return int(np.count_nonzero(labels == int(nid)))
    except Exception:
        return 0


def _choose_two_pin_nets_by_groups(
    labels: np.ndarray,
    groups: List[Dict[str, List[Tuple[float, float]]]],
    radius: int,
) -> Tuple[Optional[int], Optional[int]]:
    best = (None, None)
    best_score = -1.0
    for group in groups:
        n1 = _nearest_net_from_candidates(labels, group["pin1"], radius=radius)
        n2 = _nearest_net_from_candidates(labels, group["pin2"], radius=radius)
        score = 0.0
        if n1 is not None:
            score += 1.0
        if n2 is not None:
            score += 1.0
        if n1 is not None and n2 is not None and n1 != n2:
            score += 8.0
        # 真实导线网络通常面积明显大于 bbox 内残留的小碎片。
        score += min(_net_area(labels, n1), 3000) / 1000.0
        score += min(_net_area(labels, n2), 3000) / 1000.0
        if score > best_score:
            best_score = score
            best = (n1, n2)
    return best



def _component_is_load_like_for_topology(comp: Component) -> bool:
    name = comp.name.lower()
    ref = _component_ref_name(comp)
    return name in {"load", "connector", "terminal", "motor"} or bool(re.fullmatch(r"P\d+", ref))


def _repair_load_nets_from_nearby_diodes(components: List[Component]) -> None:
    """用附近续流二极管修复 OCR 兜底负载的第二端网络。

    P6/MOTOR 这类负载有时是 OCR 生成的大框，pin1/pin2 可能只接到 +12V，
    另一端显示“未连接”。这会让“二极管与负载并联”只能依赖几何兜底。
    这里在负载附近寻找竖直二极管：K 在上作为高侧，A 在下作为低侧，
    然后把负载缺失的一端补成对应的二极管另一端网络。
    这不会改变二极管 A/K，只是让 P6 在日志中更像真实两端负载。
    """
    diodes = [c for c in components if c.name.lower() in {"diode", "schottky_diode", "zener_diode", "led"}]
    loads = [c for c in components if _component_is_load_like_for_topology(c)]
    if not diodes or not loads:
        return

    for load in loads:
        n1 = load.nets.get("pin1")
        n2 = load.nets.get("pin2")
        if n1 is not None and n2 is not None and n1 != n2:
            continue

        lcx, lcy = load.center
        best = None
        best_d = 1e18
        for d in diodes:
            a = d.nets.get("anode")
            k = d.nets.get("cathode")
            a_pt = d.keypoints.get("anode")
            k_pt = d.keypoints.get("cathode")
            if a is None or k is None or a == k or a_pt is None or k_pt is None:
                continue
            # 负载续流/反接保护管通常竖直放在负载旁边。
            if abs(float(a_pt[1]) - float(k_pt[1])) < max(10.0, abs(float(a_pt[0]) - float(k_pt[0])) * 0.7):
                continue
            dcx, dcy = d.center
            dist = ((float(lcx) - float(dcx)) ** 2 + (float(lcy) - float(dcy)) ** 2) ** 0.5
            if dist < best_d and dist <= 700:
                best_d = dist
                best = d

        if best is None:
            continue

        a = best.nets.get("anode")
        k = best.nets.get("cathode")
        if a is None or k is None or a == k:
            continue

        # 缺哪一端就补哪一端；如果两端都缺，则按负载跨接高侧/低侧补齐。
        if n1 is None and n2 is None:
            load.nets["pin1"] = k
            load.nets["pin2"] = a
        elif n1 is None:
            load.nets["pin1"] = a if n2 == k else k
        elif n2 is None:
            load.nets["pin2"] = a if n1 == k else k

        load.nets["anode"] = load.nets.get("pin1")
        load.nets["cathode"] = load.nets.get("pin2")

def assign_nets(
    components: List[Component],
    wire_graph: WireGraph,
    pin_search_radius: int = 28,
) -> None:
    # 如果原导线提取器没有识别到绿色导线，这里用原图颜色做一次兜底提取。
    _ensure_wire_graph_from_image_if_empty(wire_graph)
    labels = wire_graph.label_image

    # 先修正稳压管 A/K，再根据关键点去接网。
    correct_zener_keypoints(components)

    for comp in components:
        cname = comp.name.lower()

        # 1) 有 A/K 关键点的元件，优先按关键点方向向 bbox 外侧追踪真实导线。
        #    只用关键点本身会经常命中“被擦除后剩下的小碎片 NET”，导致局部拓扑断裂。
        for pin_name, pt in comp.keypoints.items():
            if pt is None:
                comp.nets[pin_name] = None
            else:
                candidates = _infer_polar_pin_candidate_points(comp, pin_name)
                comp.nets[pin_name] = _nearest_net_from_candidates(
                    labels,
                    candidates,
                    radius=max(pin_search_radius, 42),
                )

        # 1.5) 修复 A/K 同时吸到同一个网络的情况。
        #      例如续流二极管 D5 应为 A=GND、K=SW，但可能被导线提取成 A=SW、K=SW。
        _repair_same_net_polar_component(labels, comp, radius=max(pin_search_radius, 78))

        # 2) 电阻、电容、电解电容、电感没有 A/K 点，按 bbox 两端估计 pin1/pin2。
        #    这一步是后续 rules.py 判断“二极管并联元件”和“一跳串联路径”的基础。
        if cname in TWO_PIN_COMPONENT_CLASSES:
            groups = _infer_two_pin_candidate_groups(comp)
            n1, n2 = _choose_two_pin_nets_by_groups(
                labels,
                groups,
                radius=max(pin_search_radius, 42),
            )
            comp.nets["pin1"] = n1
            comp.nets["pin2"] = n2
            # GUI 当前统一用 A_net/K_net 列显示。对电阻/普通电容/电感，
            # 这里把 pin1/pin2 同步到 anode/cathode 只是为了显示和调试，
            # 不代表这些元件真的有阳极/阴极。Component.has_polarity() 仍由 keypoints 决定。
            # 注意：不能用 setdefault，因为前面 keypoints 循环可能已经写入 None，
            # setdefault 不会覆盖 None，导致 R/C/L 仍显示“未连接”。
            comp.nets["anode"] = n1
            comp.nets["cathode"] = n2

        # 3) 对没有关键点、也不是两端元件的 power/gnd 符号，使用 bbox 中心关联网络。
        if not comp.nets:
            comp.nets["center"] = WireExtractor.nearest_net_id(labels, comp.center, radius=pin_search_radius)

    # v16：P6/MOTOR 这类 OCR 兜底负载可能只接到一端网络，
    # 这里利用附近续流二极管把另一端补齐，便于 rules.py 做并联判断。
    _repair_load_nets_from_nearby_diodes(components)

    # 如果调用顺序是 build_net_names -> assign_nets，则第一次命名时还没有 A/K 网络。
    # 这里在接网完成后再补一次 GND，解决 D5.A=NET_x、D5.K=SW 时 GND 不显示的问题。
    if getattr(wire_graph, "net_names", None) is not None and getattr(wire_graph, "net_aliases", None) is not None:
        _infer_general_power_ground_aliases(wire_graph.net_names, wire_graph.net_aliases, components)


# -----------------------------------------------------------------------------
# 当前阻容降压模板的电源/GND 网络补全
# -----------------------------------------------------------------------------

def _component_ref_name(comp: Component) -> str:
    return _norm_text(getattr(comp, "ref", "") or "").replace("?", "")


def _add_net_alias(
    nid: Optional[int],
    alias: str,
    net_names: Dict[int, str],
    net_aliases: Dict[int, List[str]],
) -> None:
    if nid is None:
        return
    nid = int(nid)
    alias = _norm_text(alias)
    if not alias:
        return

    # v16：防止同一个网络同时被污染成 +12V/GND。
    # 这种污染会让 rules.py 同时认为该网是高侧和低侧，导致并联负载规则不稳定。
    # 如果一个网络已经有明确电源标签，就不要再给它加 GND；反之亦然。
    existing = {_norm_text(net_names.get(nid, ""))}
    existing.update(_norm_text(x) for x in net_aliases.get(nid, []))
    existing = {x for x in existing if x}
    has_power = any(_is_power_text(x) for x in existing)
    has_gnd = any(_is_ground_text(x) for x in existing)
    if _is_ground_text(alias) and has_power and not has_gnd:
        return
    if _is_power_text(alias) and has_gnd and not has_power:
        return

    net_aliases.setdefault(nid, [])
    if alias not in net_aliases[nid]:
        net_aliases[nid].append(alias)
    old = net_names.get(nid)
    if _choose_better_name(old, alias):
        net_names[nid] = alias


def _alias_set(nid: Optional[int], net_names: Dict[int, str], net_aliases: Dict[int, List[str]]) -> Set[str]:
    if nid is None:
        return set()
    nid = int(nid)
    out = {_norm_text(net_names.get(nid, ""))}
    out.update(_norm_text(x) for x in net_aliases.get(nid, []))
    return {x for x in out if x}


def _component_refs(components: List[Component]) -> Set[str]:
    refs: Set[str] = set()
    for c in components:
        r = _component_ref_name(c)
        if r:
            refs.add(r)
    return refs


def _looks_like_cap_dropper_template(components: List[Component]) -> bool:
    """只在阻容降压模板图里启用 +V/GND 补别名。

    之前版本会把任何含 D5/D6/DZ1 的电路都硬补成 +5V/GND，
    导致普通 Buck 电路的 +6.3V 被显示成 +5V，甚至 D5 两端都变成 +5V。
    这里要求同时检测到 C5、R6、R7、D6 这类阻容降压模板特征，
    普通开关电源图不再套用该模板。
    """
    refs = _component_refs(components)
    has_dropper_parts = {"C5", "R6", "R7", "D6"}.issubset(refs)
    has_d5 = "D5" in refs
    has_zener = any(r.startswith(("DZ", "ZD", "TVS")) for r in refs)
    return bool(has_dropper_parts and (has_d5 or has_zener))


def _detected_power_alias(net_names: Dict[int, str], net_aliases: Dict[int, List[str]]) -> Optional[str]:
    """返回 OCR 已识别到的电源电压名，例如 +6.3V、+12V、VCC。"""
    best = None
    best_score = -1
    for nid, aliases in net_aliases.items():
        vals = list(aliases) + [net_names.get(nid, "")]
        for v in vals:
            t = _norm_text(v)
            if not t or not _is_power_text(t):
                continue
            # 优先保留带 + 和具体电压的名字，如 +6.3V；其次 VCC/VDD。
            score = 0
            if t.startswith("+"):
                score += 3
            if re.fullmatch(r"\+?\d+(?:\.\d+)?V", t):
                score += 2
            if score > best_score:
                best_score = score
                best = t
    return best


def _infer_template_power_ground_aliases(
    net_names: Dict[int, str],
    net_aliases: Dict[int, List[str]],
    components: List[Component],
) -> None:
    """仅对阻容降压模板补全输出正轨和 GND。

    重要：不要把所有电路都硬补成 +5V。
    如果 OCR 已识别到 +6.3V、+12V 等，就使用真实电压名；
    只有当前图确认为阻容降压模板且没有识别到具体电压时，才默认 +5V。
    """
    if not _looks_like_cap_dropper_template(components):
        return

    pos_alias = _detected_power_alias(net_names, net_aliases) or "+5V"

    pos_candidates: List[int] = []
    gnd_candidates: List[int] = []

    # 1) 已经有 ACL/ACLIN 标签的长上轨，通常就是本模板中的输出正轨。
    for nid, aliases in list(net_aliases.items()):
        labels = {_norm_text(x) for x in aliases}
        labels.add(_norm_text(net_names.get(nid, "")))
        if labels & {"ACL", "ACLIN", "ACL_IN", "+5V", "5V", "VCC", "VDD"}:
            pos_candidates.append(int(nid))

    # 2) 从已识别的 D5/DZ/D6 端点补充候选。
    for comp in components:
        cname = comp.name.lower()
        ref = _component_ref_name(comp)
        a = comp.nets.get("anode")
        k = comp.nets.get("cathode")

        if ref == "D5":
            # 本模板 D5 上端为正轨，通常由 K 端接到顶部 ACL/+5V。
            if k is not None:
                pos_candidates.append(int(k))
        if ref.startswith(("DZ", "ZD", "TVS")) or cname in {"zener_diode", "tvs"}:
            # 并联稳压管的 K 端在上方正轨，A 端在下方 GND。
            if k is not None:
                pos_candidates.append(int(k))
            if a is not None:
                gnd_candidates.append(int(a))
        if ref == "D6":
            # 本模板 D6 的 A 端在右侧低侧输出轨，即 GND。
            if a is not None:
                gnd_candidates.append(int(a))

    def _most_common(items: List[int], forbidden: Optional[int] = None) -> Optional[int]:
        counts: Dict[int, int] = {}
        for x in items:
            if x is None or x == forbidden:
                continue
            counts[int(x)] = counts.get(int(x), 0) + 1
        if not counts:
            return None
        return max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]

    pos = _most_common(pos_candidates)
    gnd = _most_common(gnd_candidates, forbidden=pos)

    if pos is not None:
        _add_net_alias(pos, pos_alias, net_names, net_aliases)
        # ACL 是交流输入/上轨文字，只作为别名保留，不覆盖真实电压名。
        _add_net_alias(pos, "ACL", net_names, net_aliases)
    if gnd is not None and gnd != pos:
        _add_net_alias(gnd, "GND", net_names, net_aliases)




def _infer_general_power_ground_aliases(
    net_names: Dict[int, str],
    net_aliases: Dict[int, List[str]],
    components: List[Component],
) -> None:
    """给普通 Buck/Boost 等电源图补充 GND 与真实输出电压别名。

    关键修正：这个函数可能在 build_net_names 阶段被调用，也可能在 assign_nets
    之后被再次调用。后者很重要，因为只有 assign_nets 之后，D5.A/D5.K 才知道
    自己接到了哪个网络。否则像 Buck 续流二极管会长期显示 A=NET_x, K=SW，
    无法把 A 端低侧网络命名成 GND。
    """
    sw_labels = {"SW", "LX", "PHASE", "DRAIN", "SOUT", "SWITCH"}

    # 1) Buck 续流二极管：K 接 SW/LX，A 在下侧，A 端网络补 GND。
    for comp in components:
        cname = comp.name.lower()
        if cname not in {"diode", "schottky_diode", "zener_diode", "led"}:
            continue

        a = comp.nets.get("anode")
        k = comp.nets.get("cathode")
        if a is None or k is None:
            continue

        k_labels = _alias_set(k, net_names, net_aliases)
        a_labels = _alias_set(a, net_names, net_aliases)
        a_pt = comp.keypoints.get("anode")
        k_pt = comp.keypoints.get("cathode")

        k_is_sw = bool(k_labels & sw_labels)
        a_is_sw = bool(a_labels & sw_labels)

        # 常见正确连接：K=SW、A=GND。
        if k_is_sw and not a_is_sw and a != k:
            if a_pt is None or k_pt is None or float(a_pt[1]) >= float(k_pt[1]) - 5:
                _add_net_alias(a, "GND", net_names, net_aliases)

        # 如果 A/K 被同一个 SW 网络吸住，不要把 SW 本身命名成 GND。
        # 这种情况交给 _repair_same_net_polar_component 先把 A 端修成另一个 NET。

    # 2) 如果某个网络已经有明确 GND 别名，不要让后续 NET_x 覆盖它。
    for nid, vals in list(net_aliases.items()):
        labels = {_norm_text(x) for x in vals}
        labels.add(_norm_text(net_names.get(nid, "")))
        if any(_is_ground_text(x) for x in labels):
            net_names[int(nid)] = "GND"





def _infer_ground_aliases_from_diode_and_tokens(
    net_names: Dict[int, str],
    net_aliases: Dict[int, List[str]],
    components: List[Component],
    tokens: List[OCRToken],
) -> None:
    """用 GND 文本 + 二极管端点几何关系补 GND。

    有些图中 GND 文字/符号离导线较远，普通 OCR 最近网络绑定不到。
    但二极管下端点通常就在 GND 符号上方，例如 Buck 续流管 D5。
    如果 A/K 中较低的端点附近存在 GND 文本，就把该端点网络补名为 GND。
    """
    gnd_tokens = [t for t in tokens if _is_ground_text(getattr(t, "text", ""))]
    if not gnd_tokens:
        return

    def near_gnd(pt: Optional[Tuple[float, float]]) -> bool:
        if pt is None:
            return False
        px, py = float(pt[0]), float(pt[1])
        for t in gnd_tokens:
            tx, ty = t.center
            # GND 文字通常在端点正下方或斜下方，放宽 y 方向距离。
            if abs(float(tx) - px) <= 120 and -45 <= float(ty) - py <= 210:
                return True
            if ((float(tx) - px) ** 2 + (float(ty) - py) ** 2) ** 0.5 <= 135:
                return True
        return False

    for comp in components:
        if comp.name.lower() not in {"diode", "schottky_diode", "zener_diode", "led"}:
            continue
        a = comp.nets.get("anode")
        k = comp.nets.get("cathode")
        a_pt = comp.keypoints.get("anode")
        k_pt = comp.keypoints.get("cathode")
        if a is not None and near_gnd(a_pt):
            _add_net_alias(a, "GND", net_names, net_aliases)
        if k is not None and near_gnd(k_pt):
            _add_net_alias(k, "GND", net_names, net_aliases)



def _infer_load_flyback_aliases_from_ocr(
    net_names: Dict[int, str],
    net_aliases: Dict[int, List[str]],
    components: List[Component],
    tokens: List[OCRToken],
) -> None:
    """根据 D1 与 P6/MOTOR 的邻近关系补充高侧/低侧别名。

    用途：电机/继电器续流二极管经常一端接 +12V，一端接 MOS 漏极低侧开关节点。
    OCR/导线拓扑可能无法把二极管 K 端追到 +12V 标签，也不能把 A 端直接命名为 GND。
    这里不把 A 端硬改成 GND，而是标成 LOW_SIDE；K 端尽量使用附近真实电源文字，如 +12V。
    """
    if not tokens:
        return

    load_tokens = []
    power_tokens = []
    for t in tokens:
        txt = _norm_text(getattr(t, "text", ""))
        if re.fullmatch(r"P\d+", txt) or txt in {"MOTOR", "MOTO", "LOAD", "FAN", "PUMP", "RELAY", "COIL", "BUZZER", "VALVE"}:
            load_tokens.append(t)
        if _is_power_text(txt):
            power_tokens.append(t)

    if not load_tokens:
        return

    def near_load(comp: Component) -> bool:
        cx, cy = comp.center
        for t in load_tokens:
            tx, ty = t.center
            if ((float(tx) - cx) ** 2 + (float(ty) - cy) ** 2) ** 0.5 <= 560:
                return True
        return False

    def nearest_power_name(comp: Component) -> Optional[str]:
        # 优先找二极管附近或负载附近的真实电源文字，如 +12V、+6.3V。
        cx, cy = comp.center
        best = None
        best_d = 1e18
        for t in power_tokens:
            tx, ty = t.center
            d = ((float(tx) - cx) ** 2 + (float(ty) - cy) ** 2) ** 0.5
            if d < best_d and d <= 760:
                best_d = d
                best = _norm_text(t.text)
        return best

    for comp in components:
        if comp.name.lower() not in {"diode", "schottky_diode", "zener_diode", "led"}:
            continue
        if not near_load(comp):
            continue
        a = comp.nets.get("anode")
        k = comp.nets.get("cathode")
        if a is None or k is None or a == k:
            continue
        a_pt = comp.keypoints.get("anode")
        k_pt = comp.keypoints.get("cathode")
        if a_pt is None or k_pt is None:
            continue
        # 竖直/近竖直续流二极管：上端通常为高侧，低端通常为 MOS/低侧开关节点。
        if abs(float(a_pt[1]) - float(k_pt[1])) < 8:
            continue
        power_alias = nearest_power_name(comp) or "LOAD+"
        if float(k_pt[1]) < float(a_pt[1]):
            _add_net_alias(k, power_alias, net_names, net_aliases)
            _add_net_alias(a, "LOW_SIDE", net_names, net_aliases)
        else:
            _add_net_alias(a, power_alias, net_names, net_aliases)
            _add_net_alias(k, "LOW_SIDE", net_names, net_aliases)


def _infer_led_indicator_aliases(
    net_names: Dict[int, str],
    net_aliases: Dict[int, List[str]],
    components: List[Component],
    tokens: List[OCRToken],
) -> None:
    """根据 LED 与限流电阻的串联关系补充电源侧/电阻侧网络别名。

    典型指示灯结构为：
        +V/OUT -> LED(A) -> LED(K) -> R -> GND

    YOLO-Pose 有时能检测到 LED A/K，但导线网络没有把 LED 左端合并到 +3.3V，
    于是 rules.py 只能看到 NET_75 / NET_87，无法判断 K 是否接了电源。

    这里不修改 A/K 点，只补充网络语义：
    - 只与电阻相连的一侧标记为 LED_RESISTOR_SIDE；
    - 另一侧标记为附近真实电源名，例如 +3.3V / +5V / VBUS / OUT。
    这样如果模型识别出 K 在电源侧，LED 规则就能明确报错。
    """
    if not components:
        return

    resistors = [c for c in components if c.name.lower() == "resistor"]
    leds = [c for c in components if c.name.lower() == "led"]
    if not leds or not resistors:
        return

    power_tokens = []
    for t in tokens or []:
        txt = _norm_text(getattr(t, "text", ""))
        if _is_power_text(txt) or txt in {"OUT", "VOUT", "VO", "VCC", "VDD", "VBUS"}:
            power_tokens.append(t)

    def has_resistor_on_net(nid: Optional[int]) -> bool:
        if nid is None:
            return False
        for r in resistors:
            vals = [v for v in (r.nets or {}).values() if v is not None]
            if int(nid) in {int(v) for v in vals}:
                return True
        return False

    def nearest_power_alias(comp: Component) -> str:
        cx, cy = comp.center
        best_txt = "LED_HIGH"
        best_d = 1e18
        for t in power_tokens:
            tx, ty = t.center
            d = ((float(tx) - float(cx)) ** 2 + (float(ty) - float(cy)) ** 2) ** 0.5
            if d < best_d and d <= 620:
                best_d = d
                best_txt = _norm_text(getattr(t, "text", ""))
        # VBUS(+5V) 经过 _norm_text 会保留 VBUS 字样；作为电源语义足够。
        return best_txt or "LED_HIGH"

    for led in leds:
        a = led.nets.get("anode")
        k = led.nets.get("cathode")
        if a is None or k is None or a == k:
            continue
        a_res = has_resistor_on_net(a)
        k_res = has_resistor_on_net(k)
        if a_res == k_res:
            continue
        pwr = nearest_power_alias(led)
        if a_res and not k_res:
            _add_net_alias(a, "LED_RESISTOR_SIDE", net_names, net_aliases)
            _add_net_alias(k, pwr, net_names, net_aliases)
        elif k_res and not a_res:
            _add_net_alias(k, "LED_RESISTOR_SIDE", net_names, net_aliases)
            _add_net_alias(a, pwr, net_names, net_aliases)


# -----------------------------------------------------------------------------
# 网络命名：核心修改部分
# -----------------------------------------------------------------------------

def build_net_names(
    wire_graph: WireGraph,
    tokens: List[OCRToken],
    components: List[Component],
    ocr_search_radius: int = 55,
) -> Dict[int, str]:
    _ensure_wire_graph_from_image_if_empty(wire_graph)
    labels = wire_graph.label_image
    net_names: Dict[int, str] = {}
    net_aliases: Dict[int, List[str]] = {}

    # 1) OCR 文本命名网络。
    #    不只用 extract_net_keywords 的结果，也兜底遍历 tokens，避免 +3.3V / VBUS(+5V) 被过滤漏掉。
    candidate_tokens: List[OCRToken] = []
    seen_ids = set()

    for token in list(extract_net_keywords(tokens)) + list(tokens):
        tid = id(token)
        if tid in seen_ids:
            continue
        seen_ids.add(tid)
        text = _norm_text(getattr(token, "text", ""))
        if _is_useful_net_text(text):
            candidate_tokens.append(token)

    for token in candidate_tokens:
        text = _norm_text(token.text)
        # 电源/GND 标签通常是红色并且离导线/电源符号稍远，
        # 用更大的搜索半径；普通信号仍用较小半径，避免误绑定。
        bind_radius = ocr_search_radius
        if _is_power_text(text) or _is_ground_text(text):
            bind_radius = max(ocr_search_radius, 115)
        nid = _nearest_net_for_token(labels, token, radius=bind_radius)

        if nid is None:
            continue

        net_aliases.setdefault(nid, [])
        if text not in net_aliases[nid]:
            net_aliases[nid].append(text)

        old = net_names.get(nid)
        if _choose_better_name(old, text):
            net_names[nid] = text

    # 2) YOLO 检测到 power/gnd 符号时也作为网络命名依据。
    #    这里的优先级低于 OCR 中明确的 +3.3V/VBUS/GND，但可以补漏。
    for comp in components:
        cname = comp.name.lower()
        if cname not in {"gnd", "power"}:
            continue

        nid = WireExtractor.nearest_net_id(labels, comp.center, radius=ocr_search_radius)
        if nid is None:
            continue

        new_name = "GND" if cname == "gnd" else "VCC"
        net_aliases.setdefault(nid, [])
        if new_name not in net_aliases[nid]:
            net_aliases[nid].append(new_name)

        old = net_names.get(nid)
        if _choose_better_name(old, new_name):
            net_names[nid] = new_name

    # 2.5) 当前阻容降压模板中，OCR 经常漏识别红色 +5V/GND。
    #      这里按 D5/DZ/D6 的局部拓扑给输出正轨和低侧轨补别名，
    #      让日志从 ACL/NET_x 变成 +5V/GND，同时让规则能使用电源/地语义。
    _infer_template_power_ground_aliases(net_names, net_aliases, components)

    # 2.6) 普通 Buck/Boost 等电源图的安全补全。
    #      不会默认写 +5V，只根据已识别的 SW/LX 和二极管低侧关系补 GND。
    _infer_general_power_ground_aliases(net_names, net_aliases, components)

    # 2.7) GND 文本离导线较远时，结合二极管端点几何关系补 GND。
    _infer_ground_aliases_from_diode_and_tokens(net_names, net_aliases, components, tokens)

    # 2.8) 电机/负载续流二极管：根据 P6/MOTOR 和附近 +12V 等文字补别名。
    _infer_load_flyback_aliases_from_ocr(net_names, net_aliases, components, tokens)

    # 2.9) LED 指示灯：根据 LED 与限流电阻关系补充 +V/LED_RESISTOR_SIDE，避免 K 接电源漏报。
    _infer_led_indicator_aliases(net_names, net_aliases, components, tokens)

    # 3) 未命名网络自动命名。
    for nid in range(1, wire_graph.num_nets + 1):
        net_names.setdefault(nid, f"NET_{nid}")
        net_aliases.setdefault(nid, [])
        # 主名称也作为别名，便于 rules.py 统一判断。
        if net_names[nid] not in net_aliases[nid]:
            net_aliases[nid].append(net_names[nid])

    # 再补一次：如果当前流程是 assign_nets -> build_net_names，
    # 这里能直接根据 D5.K=SW 推断 D5.A 所在网络为 GND。
    _infer_general_power_ground_aliases(net_names, net_aliases, components)
    _infer_ground_aliases_from_diode_and_tokens(net_names, net_aliases, components, tokens)
    _infer_load_flyback_aliases_from_ocr(net_names, net_aliases, components, tokens)
    _infer_led_indicator_aliases(net_names, net_aliases, components, tokens)

    wire_graph.net_names = net_names
    wire_graph.net_aliases = net_aliases
    return net_names


# -----------------------------------------------------------------------------
# 给 rules.py 使用的网络名接口
# -----------------------------------------------------------------------------

def net_name(net_id: Optional[int], names: Dict[int, str]) -> str:
    if net_id is None:
        return "未连接"

    if names is None:
        return f"NET_{net_id}"

    value = names.get(net_id)
    if value is None:
        value = names.get(str(net_id))  # 兼容字符串 key

    if value:
        return _norm_text(value)

    return f"NET_{net_id}"


def net_aliases(net_id: Optional[int], wire_graph: WireGraph) -> List[str]:
    """返回一个网络的全部文字别名。

    例如同一根导线既被 OCR 到 ACL，又被 OCR 到 +5V：
    net_names 可能只显示 +5V，但 net_aliases 会保留 [ACL, +5V]。
    """
    if net_id is None:
        return []

    aliases = getattr(wire_graph, "net_aliases", {}) or {}
    values = aliases.get(net_id)
    if values is None:
        values = aliases.get(str(net_id), [])

    out: List[str] = []
    for v in values or []:
        nv = _norm_text(v)
        if nv and nv not in out:
            out.append(nv)

    primary = net_name(net_id, wire_graph.net_names)
    if primary and primary not in out:
        out.append(primary)

    return out


def is_power_name(name: str) -> bool:
    return _is_power_text(name)


def is_ground_name(name: str) -> bool:
    return _is_ground_text(name)
