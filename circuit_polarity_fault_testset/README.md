# Circuit Polarity Fault Test Set

本数据集由用户上传的 `images.rar` 整理而成，包含正常电路图与故障电路图配对样本。

## 目录结构

```
circuit_polarity_fault_testset/
├── images/
│   ├── normal/        # 正常电路图，10张
│   └── fault/         # 故障电路图，10张
├── annotations/
│   ├── normal/        # 正常图 JSON，faults 为空
│   ├── fault/         # 故障图 JSON，含 fault_bbox
│   └── all_annotations.json
├── yolo_fault_detection/
│   ├── images/test/   # YOLO 故障位置测试图片
│   ├── labels/test/   # 正常图为空标签，故障图为 polarity_fault 框
│   └── data.yaml
├── previews/fault_bbox/ # 红框故障位置预览图
├── dataset_index.csv
└── dataset_summary.json
```

## 标注说明

- 正常图：`label = normal`，`faults = []`。
- 故障图：`label = fault`，`fault_category = polarity_fault`。
- `fault_bbox` 格式为 `[x1, y1, x2, y2]`，单位是像素。
- YOLO 标签中类别 `0` 表示 `polarity_fault`。

## 使用建议

该数据集适合作为系统级测试集，而不是元器件识别训练集。推荐流程：

1. 用 YOLO-Pose 识别二极管/LED/极性电容及 A/K、+/- 关键点；
2. 提取电路连接关系；
3. 用规则判断极性错误；
4. 将系统输出的故障位置与本数据集的 `fault_bbox` 进行对比。

## 注意

部分故障框是根据图片差异和人工查看生成的初始标注，建议正式实验前打开 `previews/fault_bbox/` 逐张快速复核，尤其是复杂电路中的第 7～10 组样本。