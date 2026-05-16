from __future__ import annotations

import os

# 必须放在 PaddleOCR / PaddlePaddle 导入之前，用于规避部分 Windows + PaddlePaddle 3.x 的 PIR/oneDNN 兼容问题。
os.environ.setdefault("FLAGS_enable_pir_api", "0")
os.environ.setdefault("FLAGS_use_onednn", "0")
os.environ.setdefault("FLAGS_use_mkldnn", "0")

import threading
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.yaml"
RESULT_DIR = ROOT / "results" / "gui"


class ImagePanel(ttk.Frame):
    """带标题栏的图片显示面板。"""

    def __init__(self, master: tk.Misc, title: str, subtitle: str) -> None:
        super().__init__(master, style="Card.TFrame")
        self.title = title
        self.subtitle = subtitle
        self.photo: ImageTk.PhotoImage | None = None

        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        header = ttk.Frame(self, style="Card.TFrame")
        header.grid(row=0, column=0, sticky="ew", padx=16, pady=(14, 8))
        header.columnconfigure(0, weight=1)

        ttk.Label(header, text=title, style="PanelTitle.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(header, text=subtitle, style="PanelSub.TLabel").grid(row=1, column=0, sticky="w", pady=(2, 0))

        self.canvas = tk.Canvas(
            self,
            height=430,
            highlightthickness=0,
            bg="#F8FAFC",
            bd=0,
        )
        self.canvas.grid(row=1, column=0, sticky="nsew", padx=16, pady=(0, 16))
        self.canvas.bind("<Configure>", lambda _e: self._redraw_placeholder())
        self._placeholder_text = "等待上传图片" if "原图" in title else "等待检测结果"
        self._redraw_placeholder()

    def _redraw_placeholder(self) -> None:
        if self.photo is not None:
            return
        self.canvas.delete("all")
        w = max(1, self.canvas.winfo_width())
        h = max(1, self.canvas.winfo_height())
        self.canvas.create_rectangle(12, 12, w - 12, h - 12, outline="#CBD5E1", dash=(4, 4), width=2)
        self.canvas.create_text(w // 2, h // 2 - 12, text=self._placeholder_text, fill="#64748B", font=("Microsoft YaHei UI", 14, "bold"))
        self.canvas.create_text(w // 2, h // 2 + 18, text="支持 PNG / JPG / JPEG / BMP", fill="#94A3B8", font=("Microsoft YaHei UI", 10))

    def clear(self, placeholder: str | None = None) -> None:
        self.photo = None
        if placeholder:
            self._placeholder_text = placeholder
        self._redraw_placeholder()

    def show_image(self, image_path: str | Path) -> None:
        path = Path(image_path)
        img = Image.open(path).convert("RGB")
        canvas_w = max(320, self.canvas.winfo_width() - 32)
        canvas_h = max(240, self.canvas.winfo_height() - 32)

        try:
            resample = Image.Resampling.LANCZOS
        except AttributeError:  # Pillow 老版本兼容
            resample = Image.LANCZOS

        img.thumbnail((canvas_w, canvas_h), resample)
        self.photo = ImageTk.PhotoImage(img)

        self.canvas.delete("all")
        self.canvas.create_rectangle(10, 10, self.canvas.winfo_width() - 10, self.canvas.winfo_height() - 10, outline="#E2E8F0", width=1)
        self.canvas.create_image(self.canvas.winfo_width() // 2, self.canvas.winfo_height() // 2, image=self.photo, anchor="center")


class CircuitPolarityApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("电路原理图极性错误检测系统")
        self.geometry("1280x820")
        self.minsize(1120, 720)
        self.configure(bg="#EEF2F7")

        self.image_path: str | None = None
        self.pipeline = None
        self.is_running = False

        self._setup_style()
        self._build_ui()

    def _setup_style(self) -> None:
        self.style = ttk.Style(self)
        try:
            self.style.theme_use("clam")
        except tk.TclError:
            pass

        self.colors = {
            "bg": "#EEF2F7",
            "card": "#FFFFFF",
            "primary": "#2563EB",
            "primary_dark": "#1D4ED8",
            "success": "#16A34A",
            "warning": "#D97706",
            "danger": "#DC2626",
            "text": "#0F172A",
            "muted": "#64748B",
            "border": "#E2E8F0",
        }

        font_base = ("Microsoft YaHei UI", 10)
        font_title = ("Microsoft YaHei UI", 18, "bold")
        font_panel = ("Microsoft YaHei UI", 12, "bold")
        font_mono = ("Consolas", 10)

        self.style.configure("Root.TFrame", background=self.colors["bg"])
        self.style.configure("Header.TFrame", background="#0F172A")
        self.style.configure("Card.TFrame", background=self.colors["card"], relief="flat")
        self.style.configure("Title.TLabel", background="#0F172A", foreground="#FFFFFF", font=font_title)
        self.style.configure("HeaderSub.TLabel", background="#0F172A", foreground="#CBD5E1", font=font_base)
        self.style.configure("PanelTitle.TLabel", background=self.colors["card"], foreground=self.colors["text"], font=font_panel)
        self.style.configure("PanelSub.TLabel", background=self.colors["card"], foreground=self.colors["muted"], font=font_base)
        self.style.configure("Status.TLabel", background=self.colors["card"], foreground=self.colors["muted"], font=font_base)
        self.style.configure("MetricName.TLabel", background=self.colors["card"], foreground=self.colors["muted"], font=("Microsoft YaHei UI", 9))
        self.style.configure("MetricValue.TLabel", background=self.colors["card"], foreground=self.colors["text"], font=("Microsoft YaHei UI", 16, "bold"))

        self.style.configure("Primary.TButton", font=("Microsoft YaHei UI", 10, "bold"), foreground="#FFFFFF", background=self.colors["primary"], borderwidth=0, padding=(16, 10))
        self.style.map("Primary.TButton", background=[("active", self.colors["primary_dark"]), ("disabled", "#93C5FD")])
        self.style.configure("Ghost.TButton", font=("Microsoft YaHei UI", 10), foreground=self.colors["text"], background="#F8FAFC", borderwidth=1, padding=(14, 9))
        self.style.map("Ghost.TButton", background=[("active", "#E2E8F0")])
        self.style.configure("Danger.TButton", font=("Microsoft YaHei UI", 10), foreground="#FFFFFF", background=self.colors["danger"], borderwidth=0, padding=(14, 9))
        self.style.map("Danger.TButton", background=[("active", "#B91C1C")])
        self.option_add("*Font", font_base)
        self._font_mono = font_mono

    def _build_ui(self) -> None:
        root = ttk.Frame(self, style="Root.TFrame")
        root.pack(fill=tk.BOTH, expand=True)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)

        self._build_header(root)
        self._build_main(root)
        self._build_bottom(root)

    def _build_header(self, parent: ttk.Frame) -> None:
        header = ttk.Frame(parent, style="Header.TFrame")
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)

        title_box = ttk.Frame(header, style="Header.TFrame")
        title_box.grid(row=0, column=0, sticky="w", padx=24, pady=18)
        ttk.Label(title_box, text="电路原理图极性错误检测系统", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            title_box,
            text="YOLOv8-Pose 元件与 A/K 极性识别  ·  OCR 网络文本识别  ·  导线拓扑与规则判断",
            style="HeaderSub.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))

        btn_box = ttk.Frame(header, style="Header.TFrame")
        btn_box.grid(row=0, column=1, sticky="e", padx=24, pady=18)
        self.upload_btn = ttk.Button(btn_box, text="上传图片", style="Primary.TButton", command=self.choose_image)
        self.upload_btn.grid(row=0, column=0, padx=(0, 10))
        self.detect_btn = ttk.Button(btn_box, text="开始检测", style="Primary.TButton", command=self.start_detect)
        self.detect_btn.grid(row=0, column=1, padx=(0, 10))
        ttk.Button(btn_box, text="关闭", style="Danger.TButton", command=self.destroy).grid(row=0, column=2)

    def _build_main(self, parent: ttk.Frame) -> None:
        main = ttk.Frame(parent, style="Root.TFrame")
        main.grid(row=1, column=0, sticky="nsew", padx=18, pady=16)
        main.columnconfigure(0, weight=1)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        self.original_panel = ImagePanel(main, "原图", "待检测的电路原理图")
        self.original_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 9))
        self.result_panel = ImagePanel(main, "检测结果", "故障框、极性点与规则结论")
        self.result_panel.grid(row=0, column=1, sticky="nsew", padx=(9, 0))

    def _build_bottom(self, parent: ttk.Frame) -> None:
        bottom = ttk.Frame(parent, style="Root.TFrame")
        bottom.grid(row=2, column=0, sticky="ew", padx=18, pady=(0, 16))
        bottom.columnconfigure(0, weight=1)

        info = ttk.Frame(bottom, style="Card.TFrame")
        info.grid(row=0, column=0, sticky="ew")
        info.columnconfigure(0, weight=1)

        status_line = ttk.Frame(info, style="Card.TFrame")
        status_line.grid(row=0, column=0, sticky="ew", padx=16, pady=(14, 8))
        status_line.columnconfigure(1, weight=1)

        ttk.Label(status_line, text="状态", style="PanelTitle.TLabel").grid(row=0, column=0, sticky="w")
        self.status_var = tk.StringVar(value="请先上传原理图图片。")
        ttk.Label(status_line, textvariable=self.status_var, style="Status.TLabel").grid(row=0, column=1, sticky="w", padx=(14, 0))
        self.progress = ttk.Progressbar(status_line, mode="indeterminate", length=170)
        self.progress.grid(row=0, column=2, sticky="e")

        metrics = ttk.Frame(info, style="Card.TFrame")
        metrics.grid(row=1, column=0, sticky="ew", padx=16, pady=(0, 10))
        for i in range(4):
            metrics.columnconfigure(i, weight=1)

        self.metric_components = self._metric_card(metrics, 0, "检测元件", "0")
        self.metric_ocr = self._metric_card(metrics, 1, "OCR文字", "0")
        self.metric_nets = self._metric_card(metrics, 2, "网络数量", "0")
        self.metric_faults = self._metric_card(metrics, 3, "故障 / 复核", "0 / 0")

        text_frame = ttk.Frame(info, style="Card.TFrame")
        text_frame.grid(row=2, column=0, sticky="ew", padx=16, pady=(0, 16))
        text_frame.columnconfigure(0, weight=1)

        self.text = tk.Text(
            text_frame,
            height=9,
            font=self._font_mono,
            bg="#0B1120",
            fg="#E5E7EB",
            insertbackground="#FFFFFF",
            relief="flat",
            padx=12,
            pady=10,
            wrap="word",
        )
        self.text.grid(row=0, column=0, sticky="ew")
        scroll = ttk.Scrollbar(text_frame, orient="vertical", command=self.text.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.text.configure(yscrollcommand=scroll.set)
        self.text.tag_configure("danger", foreground="#FCA5A5")
        self.text.tag_configure("warning", foreground="#FCD34D")
        self.text.tag_configure("ok", foreground="#86EFAC")
        self.text.tag_configure("info", foreground="#BFDBFE")
        self._write_log("请上传图片后点击“开始检测”。", "info")

    def _metric_card(self, parent: ttk.Frame, column: int, name: str, value: str) -> tk.StringVar:
        box = ttk.Frame(parent, style="Card.TFrame")
        box.grid(row=0, column=column, sticky="ew", padx=(0 if column == 0 else 8, 0))
        var = tk.StringVar(value=value)
        ttk.Label(box, text=name, style="MetricName.TLabel").pack(anchor="w")
        ttk.Label(box, textvariable=var, style="MetricValue.TLabel").pack(anchor="w", pady=(2, 0))
        return var

    def choose_image(self) -> None:
        path = filedialog.askopenfilename(
            title="选择原理图图片",
            filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp"), ("All files", "*.*")],
        )
        if not path:
            return

        self.image_path = path
        self.original_panel.show_image(path)
        self.result_panel.clear("等待检测结果")
        self.status_var.set(f"已选择：{Path(path).name}")
        self._reset_metrics()
        self._clear_log()
        self._write_log(f"已选择图片：{path}", "info")

    def start_detect(self) -> None:
        if not self.image_path:
            messagebox.showwarning("提示", "请先上传图片。")
            return
        if self.is_running:
            return

        self.is_running = True
        self.detect_btn.state(["disabled"])
        self.upload_btn.state(["disabled"])
        self.status_var.set("正在检测，请稍等……")
        self.progress.start(12)
        self._clear_log()
        self._write_log("检测任务已启动……", "info")

        t = threading.Thread(target=self._detect_worker, daemon=True)
        t.start()

    def _detect_worker(self) -> None:
        try:
            if self.pipeline is None:
                # 延迟导入，避免 GUI 启动阶段因为模型/OCR 依赖问题直接崩溃。
                from src.pipeline import CircuitPolarityPipeline

                self.pipeline = CircuitPolarityPipeline(str(CONFIG_PATH))

            result = self.pipeline.run(self.image_path, out_dir=str(RESULT_DIR))
            self.after(0, lambda r=result: self._show_result(r))
        except Exception:
            err_msg = traceback.format_exc()
            self.after(0, lambda msg=err_msg: self._show_error(msg))

    def _show_result(self, result: dict) -> None:
        self._finish_running()
        result_image = result.get("result_image")
        if result_image:
            self.result_panel.show_image(result_image)

        components = result.get("components", [])
        ocr_tokens = result.get("ocr_tokens", [])
        net_names = result.get("net_names", {})
        net_aliases = result.get("net_aliases", {})
        diode_reports = result.get("diode_reports", [])
        faults = result.get("faults", [])
        danger = [f for f in faults if f.get("level") == "danger"]
        warning = [f for f in faults if f.get("level") != "danger"]

        self.metric_components.set(str(len(components)))
        self.metric_ocr.set(str(len(ocr_tokens)))
        self.metric_nets.set(str(len(net_names)))
        self.metric_faults.set(f"{len(danger)} / {len(warning)}")

        self._clear_log()
        for w in result.get("warnings", []):
            self._write_log("[系统提示] " + str(w), "warning")

        self._write_log(f"检测元件数：{len(components)}", "info")
        self._write_log(f"OCR文字数：{len(ocr_tokens)}", "info")
        self._write_log(f"网络数量：{len(net_names)}", "info")
        self._write_log(f"明确故障数：{len(danger)}，需复核项：{len(warning)}", "info")
        self._write_log("-" * 90, "info")

        self._write_component_details(components, net_names)
        self._write_ocr_summary(ocr_tokens)
        self._write_net_summary(net_names, net_aliases)
        self._write_diode_scene_reports(diode_reports)
        self._write_log("-" * 90, "info")

        if faults:
            for f in faults:
                level = str(f.get("level", "warning"))
                tag = "danger" if level == "danger" else "warning"
                self._write_log(
                    f"[{level}] {f.get('target')} | {f.get('fault_type')} | {f.get('message')}",
                    tag,
                )
        else:
            self._write_log("未发现明确极性错误。", "ok")

        if result_image:
            self._write_log("", "info")
            self._write_log("结果图：" + str(result_image), "info")

        self.status_var.set("检测完成。")


    def _net_display(self, net_id, net_names: dict) -> str:
        if net_id is None:
            return "未连接"

        key = str(net_id)
        value = net_names.get(key)

        if value is None:
            value = net_names.get(net_id)

        if value:
            return str(value)

        return f"NET_{net_id}"

    def _fmt_point(self, point) -> str:
        if point is None:
            return "None"

        try:
            x, y = point
            return f"({float(x):.1f}, {float(y):.1f})"
        except Exception:
            return str(point)

    def _write_component_details(self, components: list, net_names: dict) -> None:
        if not components:
            self._write_log("元件详情：未检测到元件。", "warning")
            return

        self._write_log("检测到的元件详情：", "info")

        for idx, comp in enumerate(components, start=1):
            name = str(comp.get("name", "-"))
            ref = comp.get("ref") or "-"
            conf = comp.get("conf", 0.0)
            bbox = comp.get("bbox")
            nets = comp.get("nets", {}) or {}
            keypoints = comp.get("keypoints", {}) or {}

            a_net = self._net_display(nets.get("anode"), net_names)
            k_net = self._net_display(nets.get("cathode"), net_names)
            a_pt = self._fmt_point(keypoints.get("anode"))
            k_pt = self._fmt_point(keypoints.get("cathode"))

            try:
                conf_text = f"{float(conf):.3f}"
            except Exception:
                conf_text = str(conf)

            tag = "info"
            if name.lower() in {"diode", "schottky_diode", "zener_diode", "led"}:
                tag = "warning" if ref == "-" or a_net.startswith("NET_") or k_net.startswith("NET_") else "ok"

            self._write_log(
                f"[{idx}] {ref} | {name} | conf={conf_text} | "
                f"A_net={a_net} | K_net={k_net} | "
                f"A={a_pt} | K={k_pt} | bbox={bbox}",
                tag,
            )

    def _write_ocr_summary(self, ocr_tokens: list) -> None:
        if not ocr_tokens:
            self._write_log("OCR关键词：无。", "warning")
            return

        texts = []
        for token in ocr_tokens:
            txt = str(token.get("text", "")).strip()
            if txt and txt not in texts:
                texts.append(txt)

        if texts:
            shown = ", ".join(texts[:60])
            more = "" if len(texts) <= 60 else f" ... 共 {len(texts)} 个"
            self._write_log("OCR关键词：" + shown + more, "info")
        else:
            self._write_log("OCR关键词：无有效文本。", "warning")

    def _write_net_summary(self, net_names: dict, net_aliases: dict | None = None) -> None:
        if not net_names:
            self._write_log("网络命名：无。", "warning")
            return

        net_aliases = net_aliases or {}
        important = []
        key_names = {"SOUT", "OUT", "SW", "LX", "VIN", "VOUT", "GND", "PGND", "VCC", "VDD", "3V3", "5V", "+5V", "+3.3V", "ACL", "ACN"}
        for key, value in net_names.items():
            aliases = net_aliases.get(key) or net_aliases.get(str(key)) or []
            names = {str(value).upper()} | {str(x).upper() for x in aliases}
            if names & key_names:
                alias_text = "/".join(sorted(names))
                important.append(f"{key}:{alias_text}")

        if important:
            self._write_log("关键网络命名/别名：" + ", ".join(important), "ok")
        else:
            self._write_log("关键网络命名：未发现 OUT/SOUT/SW/GND/VIN/+5V/ACL 等关键网络。", "warning")

    def _write_diode_scene_reports(self, diode_reports: list) -> None:
        if not diode_reports:
            self._write_log("二极管场景判断：无。", "warning")
            return

        self._write_log("二极管局部拓扑场景判断：", "info")
        for r in diode_reports:
            status = str(r.get("final_status") or r.get("scene_status") or "UNKNOWN")
            tag = "ok" if status == "PASS" else ("danger" if status == "ERROR" else "warning")
            target = r.get("target") or r.get("ref") or r.get("class")
            scene = r.get("scene")
            conf = r.get("confidence", 0)
            a = r.get("anode_net")
            k = r.get("cathode_net")
            msg = r.get("message") or r.get("reason") or ""
            expected = r.get("expected") or ""
            try:
                conf_text = f"{float(conf):.2f}"
            except Exception:
                conf_text = str(conf)
            self._write_log(
                f"[{status}] {target} | scene={scene} | conf={conf_text} | A={a} | K={k} | {msg} {expected}",
                tag,
            )

    def _show_error(self, err_msg: str) -> None:
        self._finish_running()
        self.status_var.set("检测失败。")
        self._write_log(err_msg, "danger")
        messagebox.showerror("错误", err_msg)

    def _finish_running(self) -> None:
        self.is_running = False
        self.progress.stop()
        self.detect_btn.state(["!disabled"])
        self.upload_btn.state(["!disabled"])

    def _reset_metrics(self) -> None:
        self.metric_components.set("0")
        self.metric_ocr.set("0")
        self.metric_nets.set("0")
        self.metric_faults.set("0 / 0")

    def _clear_log(self) -> None:
        self.text.configure(state="normal")
        self.text.delete("1.0", tk.END)

    def _write_log(self, text: str, tag: str = "info") -> None:
        self.text.configure(state="normal")
        self.text.insert(tk.END, text + "\n", tag)
        self.text.see(tk.END)
        self.text.configure(state="disabled")


if __name__ == "__main__":
    app = CircuitPolarityApp()
    app.mainloop()
