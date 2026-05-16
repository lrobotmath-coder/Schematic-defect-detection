from __future__ import annotations

import re
from typing import Iterable, List

from .types import OCRToken

# 最近一次 OCR 读取的原图路径。rules.py 会用它做二极管符号方向的图像兜底判断。
LAST_IMAGE_PATH: str | None = None


def get_last_image_path() -> str | None:
    return LAST_IMAGE_PATH


# 只用于“网络标签”的正则。不要把 D5/R6/C1/1N4007/470uF 等当成网络名。
NET_KEYWORD_RE = re.compile(
    r"^("
    r"VCC|VDD|VIN|VOUT|VBUS|VPP|VBAT|BAT|PWR|PWR_FLAG|"
    r"OUT|SOUT|SW|LX|PHASE|DRAIN|LOAD|OUTPUT|"
    r"GND|PGND|AGND|DGND|SGND|VSS|0V|"
    r"ACL|ACN|ACL_IN|ACN_IN|ACLIN|ACNIN|ACIN|L|N|LIVE|NEUTRAL|"
    r"\+?\d+(?:\.\d+)?V|\+?3V3|"
    r"3V3"
    r")$",
    flags=re.IGNORECASE,
)

REFDES_RE = re.compile(r"^(D\d+|DZ\d+|ZD\d+|TVS\d*|LED\d*|R\d+|C\d+|L\d+|U\d+|Q\d+|P\d+|J\d+|V\d+)$", re.I)
VALUE_RE = re.compile(
    r"^("
    r"1N\d+|"
    r"\d+(?:\.\d+)?(?:R|K|M)(?:/\d+(?:\.\d+)?W)?|"
    r"\d+(?:\.\d+)?(?:UF|NF|PF|U|N|P)(?:/\d+(?:\.\d+)?V)?|"
    r"\d+(?:\.\d+)?V|"
    r"\d+(?:\.\d+)?W|"
    r"\d+"
    r")$",
    re.I,
)

MODEL_OR_VALUE_HINTS = ("1N", "UF", "NF", "PF", "OHM", "KOHM", "MOHM", "CBB", "X2")


def normalize_text(text: str) -> str:
    """统一 OCR 文本格式。"""
    text = str(text).strip().replace(" ", "")
    text = text.replace("＋", "+").replace("（", "(").replace("）", ")")
    text = text.replace("：", ":").replace("，", ",")
    text = text.upper()
    # 只在整个字符串等于 OV 时改成 0V，避免误伤 VOUT 等文本。
    if text == "OV":
        text = "0V"
    return text


def is_valid_net_label(text: str) -> bool:
    """判断 OCR 文本是否可作为网络标签。"""
    t = normalize_text(text)
    if not t:
        return False

    if NET_KEYWORD_RE.fullmatch(t):
        return True

    # 元件编号和参数型号绝不能作为网络名。
    if REFDES_RE.fullmatch(t) or VALUE_RE.fullmatch(t):
        return False

    # 含数字的斜杠文本通常是参数，如 470UF/25V、10R/2W。
    # 不含数字的斜杠文本容易来自 OCR 把多段文字粘在一起，也先不作为网络名。
    if "/" in t:
        return False

    if len(t) > 12:
        return False
    if not re.search(r"[A-Z]", t):
        return False
    if any(h in t for h in MODEL_OR_VALUE_HINTS):
        return False
    if not re.fullmatch(r"[A-Z0-9_+\-]+", t):
        return False

    # 允许用户自定义信号网名，例如 CE、RST、KEY、SDA、ACL_TEST。
    return True


def classify_ocr_text(text: str) -> str:
    """粗分类 OCR 文本，供拓扑模块决定是否用于网络命名。"""
    t = normalize_text(text)
    if not t:
        return "noise"
    if is_valid_net_label(t):
        return "net_label"
    if REFDES_RE.fullmatch(t):
        return "refdes"
    if VALUE_RE.fullmatch(t) or any(h in t for h in MODEL_OR_VALUE_HINTS):
        return "value"
    if len(t) <= 24:
        return "text"
    return "noise"


def _should_keep_ocr_token(text: str, kind: str) -> bool:
    """过滤会造成界面大量粉色框的 OCR 噪声。

    保留网络名、元件编号、有单位的参数、以及负载/LED 等有规则意义的文本；
    丢弃纯数字、单个字母、孤立符号和明显无意义短片段。
    """
    t = normalize_text(text)
    if not t:
        return False

    if kind in {"net_label", "refdes"}:
        return True

    # 纯数字通常只是 IC 引脚号，例如 1/2/3/5/12，不能帮助网络命名，界面会很乱。
    if re.fullmatch(r"\d+(?:\.\d+)?", t):
        return False

    # 单个字母或单个符号大多是 OCR 噪声。
    if len(t) <= 1:
        return False

    # 需要保留的规则提示文本。
    important_words = {
        "MOTOR", "MOTO", "LOAD", "POWERED_ON", "POWER", "SUPPLY",
        "RED", "GREEN", "BLUE", "SCHOTTKY", "MOS", "MOSFET",
        "NMOS", "PMOS", "OUT", "IN", "SW", "FB", "EN", "VCC", "VDD",
    }
    if t in important_words:
        return True

    # 有单位的参数保留，例如 4.7UF、100NF/50V、1000UH、22UF10V。
    if re.search(r"(UF|NF|PF|UH|MH|H|OHM|Ω|R|K|M|V|W)", t):
        return True

    # 普通较长英文说明可以保留，太短的随机噪声丢弃。
    if kind == "text" and len(t) >= 3 and re.search(r"[A-Z]", t):
        # 排除常见 OCR 噪声片段。
        if re.fullmatch(r"[A-Z]{1,2}", t) and t not in important_words:
            return False
        return True

    return kind == "value" and len(t) >= 3


class OCRReader:
    """OCR 文字识别封装。

    默认 backend='none'，这样系统可以先完成 YOLO+导线+规则主流程。
    若安装了 PaddleOCR，可把 config.yaml 中 ocr.backend 改成 paddle。
    """

    def __init__(self, backend: str = "none"):
        self.backend = backend.lower().strip()
        self.warning = ""
        self.reader = None
        if self.backend == "paddle":
            self._load_paddle()

    def _load_paddle(self) -> None:
        try:
            from paddleocr import PaddleOCR
            self.reader = PaddleOCR(use_angle_cls=True, lang="en")
        except Exception as exc:  # pragma: no cover
            self.warning = f"PaddleOCR 加载失败：{exc}。已跳过 OCR。"
            self.reader = None

    def read(self, image_path: str) -> List[OCRToken]:
        global LAST_IMAGE_PATH
        LAST_IMAGE_PATH = image_path
        if self.backend == "none":
            return []
        if self.backend == "paddle" and self.reader is not None:
            return self._read_paddle(image_path)
        return []

    @staticmethod
    def _make_token(text: str, box, conf: float) -> OCRToken | None:
        text = normalize_text(text)
        kind = classify_ocr_text(text)
        if not text or kind == "noise" or not _should_keep_ocr_token(text, kind):
            return None
        try:
            return OCRToken(text=text, bbox=box, conf=float(conf), kind=kind)
        except TypeError:
            # 兼容旧版 OCRToken 没有 kind 字段的情况。
            return OCRToken(text=text, bbox=box, conf=float(conf))

    def _read_paddle(self, image_path: str) -> List[OCRToken]:
        tokens: List[OCRToken] = []

        try:
            if hasattr(self.reader, "predict"):
                result = self.reader.predict(input=image_path)
            else:
                result = self.reader.ocr(image_path)
        except Exception as exc:
            self.warning = f"OCR 识别失败：{exc}"
            return tokens

        # PaddleOCR 3.x 返回结果解析
        if isinstance(result, list):
            for item in result:
                data = None

                if isinstance(item, dict):
                    data = item.get("res", item)
                elif hasattr(item, "json"):
                    try:
                        j = item.json
                        data = j() if callable(j) else j
                        if isinstance(data, dict) and "res" in data:
                            data = data["res"]
                    except Exception:
                        data = None

                if isinstance(data, dict):
                    texts = data.get("rec_texts", [])
                    scores = data.get("rec_scores", [])
                    boxes = data.get("rec_boxes", None)
                    polys = data.get("rec_polys", None) or data.get("dt_polys", None)

                    for i, raw_text in enumerate(texts):
                        conf = 1.0
                        try:
                            conf = float(scores[i])
                        except Exception:
                            pass

                        box = None
                        try:
                            if boxes is not None and i < len(boxes):
                                x1, y1, x2, y2 = boxes[i]
                                box = [
                                    (float(x1), float(y1)),
                                    (float(x2), float(y1)),
                                    (float(x2), float(y2)),
                                    (float(x1), float(y2)),
                                ]
                        except Exception:
                            box = None

                        if box is None:
                            try:
                                if polys is not None and i < len(polys):
                                    box = [(float(x), float(y)) for x, y in polys[i]]
                            except Exception:
                                box = None

                        if box is None:
                            continue

                        token = self._make_token(str(raw_text), box, conf)
                        if token is not None:
                            tokens.append(token)
                    continue

                # PaddleOCR 2.x 兼容：item 可能本身就是若干 line。
                lines = []
                if isinstance(item, list):
                    if item and isinstance(item[0], list) and len(item[0]) == 4:
                        lines.append(item)
                    else:
                        lines.extend(item)

                for line in lines:
                    try:
                        box = [(float(x), float(y)) for x, y in line[0]]
                        text, conf = line[1][0], float(line[1][1])
                    except Exception:
                        continue
                    token = self._make_token(str(text), box, conf)
                    if token is not None:
                        tokens.append(token)

        return tokens


def extract_net_keywords(tokens: Iterable[OCRToken]) -> List[OCRToken]:
    """只提取可作为网络名的 OCR 文本。

    注意：D5、R6、C1、1N4007、470uF/25V 等会被过滤，避免污染 net_names。
    """
    out: List[OCRToken] = []
    for token in tokens:
        txt = normalize_text(getattr(token, "text", ""))
        if is_valid_net_label(txt):
            try:
                out.append(OCRToken(text=txt, bbox=token.bbox, conf=token.conf, kind="net_label"))
            except TypeError:
                out.append(OCRToken(text=txt, bbox=token.bbox, conf=token.conf))
    return out
