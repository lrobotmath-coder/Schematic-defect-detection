from __future__ import annotations

from pathlib import Path
from ultralytics import YOLO


def main() -> None:
    root = Path(__file__).resolve().parent

    data_yaml = root / "data" / "polarity_component_aug_dataset" / "yolo_pose_all" / "data.yaml"

    if not data_yaml.exists():
        raise FileNotFoundError(f"未找到数")

    # 类别数已经变化，不建议从旧 best.pt 继续训，直接用官方 pose 预训练模型重新迁移训练
    model = YOLO("yolov8n-pose.pt")

    model.train(
        data=str(data_yaml),
        epochs=200,
        imgsz=960,
        batch=4,
        device=0,          # CUDA 可用就用 0；如果报错改成 "cpu"
        workers=0,         # Windows 下建议先设为 0，减少多进程错误
        project=str(root / "runs" / "pose"),
        name="circuit_polarity_pose_aug_v2",
        patience=50,
        exist_ok=True,
        pretrained=True,
        cache=False,
        # 极性任务不建议随机翻转，否则 A/K 极性容易乱
        fliplr=0.0,
        flipud=0.0,
        degrees=0.0,

        # 降低 mosaic，避免破坏完整原理图结构
        mosaic=0.3,
    )

    print("训练完成")
    print("新权重位置：")
    print(root / "runs" / "pose" / "circuit_polarity_pose_aug_v2" / "weights" / "best.pt")


if __name__ == "__main__":
    main()