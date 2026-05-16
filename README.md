# Schematic-defect-detection
Schematic defect detection
# 电路原理图极性错误检测系统

本项目用于在 PyCharm 中搭建一个“YOLOv8-Pose + OCR + 导线网络提取 + 拓扑规则判断”的原理图元器件极性错误检测系统。

## 1. 系统功能

系统输入一张电路原理图图片，输出带故障框的检测结果图和文字判断结果。

核心流程：

1. YOLOv8-Pose 识别极性元件：普通二极管、稳压/齐纳二极管、肖特基二极管、LED、电源符号、GND、整流桥。
2. 对二极管族元件识别 2 个关键点：anode 阳极 A、cathode 阴极 K。
3. OCR 识别 VCC、GND、3V3、5V、VIN、VOUT、R1、D1、L1 等文字。
4. OpenCV 提取导线，连通域编号形成网络 net。
5. 将 A/K 引脚、OCR 网络名、导线 net 关联起来。
6. 通过规则判断二极管、LED、稳压管、续流/钳位二极管是否极性异常。
7. GUI 页面提供“上传图片”“开始检测”“关闭”按钮。

## 2. 已包含的数据集

项目已经放入你提供的两个数据集：

- `data/polarity_component_aug_dataset/`：元器件识别与 A/K 关键点训练数据。
- `data/circuit_polarity_fault_testset/`：正常/故障原理图系统测试集。

其中 `polarity_component_aug_dataset/yolo_pose_all/data.yaml` 可以直接用于 YOLOv8-Pose 训练。关键点顺序是：

```text
0: anode 阳极 A
1: cathode 阴极 K
```

## 3. PyCharm 运行步骤

### 3.1 打开项目

在 PyCharm 中打开本文件夹：

```text
CircuitPolarityChecker
```

### 3.2 创建虚拟环境并安装依赖

在 PyCharm Terminal 或 PowerShell 中执行：

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3.3 训练 YOLOv8-Pose 模型

先训练所有类别的元器件 + 二极管 A/K 关键点模型：

```powershell
python train_pose.py
```

训练完成后，权重默认在：

```text
runs/pose/circuit_polarity_pose/weights/best.pt
```

如果电脑有 NVIDIA GPU，可以把 `train_pose.py` 里的：

```python
device="cpu"
```

改成：

```python
device=0
```

如果你只想训练二极管族 A/K 关键点模型，可运行：

```powershell
python train_pose_diode_only.py
```

然后把 `config.yaml` 中的 `model.pose_weights` 改成：

```yaml
model:
  pose_weights: runs/pose/diode_polarity_pose/weights/best.pt
```

## 4. 运行 GUI 页面

训练完成后运行：

```powershell
python app.py
```

页面按钮：

- 上传图片：选择一张原理图。
- 开始检测：执行 YOLOv8-Pose、OCR、导线提取、拓扑网络分析和规则判断。
- 关闭：退出程序。

检测结果图会保存到：

```text
results/gui/
```

## 5. 单张图片命令行检测

例如检测测试集中的一张故障图：

```powershell
python run_single.py "data/circuit_polarity_fault_testset/images/fault/case_001_fault.png"
```

输出结果图会保存到：

```text
results/case_001_fault_result.png
```

## 6. 批量测试故障数据集

```powershell
python run_batch_test.py
```

输出：

```text
results/batch_fault_test/evaluation.csv
```

该脚本会把系统输出故障框与测试集标注的 `fault_bbox` 做 IoU 对比，方便后续写实验结果。

## 7. OCR 使用说明

默认 `config.yaml` 中：

```yaml
ocr:
  backend: none
```

这表示先不启用 OCR，系统仍然可以跑 YOLO、导线提取和规则流程，但网络命名会少一些。

如需启用 PaddleOCR：

1. 安装 PaddlePaddle CPU 版或 GPU 版；
2. 安装 PaddleOCR；
3. 修改 `config.yaml`：

```yaml
ocr:
  backend: paddle
```

Windows CPU 常用安装方式可参考 PaddleOCR 官方文档。若安装后仍报错，先保持 `backend: none`，等 YOLO 主流程跑通后再开启 OCR。

## 8. 规则判断逻辑

### 8.1 LED

常规指示 LED：

```text
A 阳极 → 电源/电阻侧
K 阴极 → GND/低电位侧
```

如果识别为：

```text
A → GND
K → VCC/3V3/5V
```

则判为 LED 极性反接。

### 8.2 稳压/齐纳二极管

常见钳位/稳压连接：

```text
K 阴极 → 信号/正电源侧
A 阳极 → GND
```

如果接反，则判为稳压方向错误。

### 8.3 普通二极管/肖特基二极管

需要结合拓扑判断用途：

- 串联导通/防反接二极管：一般看作电流从 A 流向 K。
- 续流/钳位二极管：通常 K 接高电位/电源侧，A 接低电位/开关侧。

当前代码通过 OCR 文字、网络名、GND/VCC、L1/coil/motor/relay 等邻近信息做启发式判断。如果缺少上下文，会输出 warning，让人工复核。

## 9. 后续可改进点

1. 增加完整原理图级别的元件框与引脚标注数据，而不仅是单个符号截图。
2. 增加电感、继电器、MOS 管、芯片引脚等类别，以便更准确识别续流二极管场景。
3. 对导线交叉但没有连接点的情况加入“交叉不连通”规则，避免错误合并网络。
4. 将正常样例图作为模板，采用图匹配/网络差分进一步判断同一 D1 在正常图和故障图中的方向变化。
5. 将规则判断结果输出为 JSON/CSV，方便写论文实验表格。
