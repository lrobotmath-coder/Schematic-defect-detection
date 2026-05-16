from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    data_yaml = Path("data/polarity_component_aug_dataset/yolo_pose_diode_only/data.yaml")
    if not data_yaml.exists():
        raise FileNotFoundError(f"未找到数据集配置：{data_yaml}")

    model = YOLO("yolov8n-pose.pt")
    model.train(
        data=str(data_yaml),
        epochs=120,
        imgsz=640,
        batch=8,
        project="runs/pose",
        name="diode_polarity_pose",
        device="cpu",  # 有 NVIDIA GPU 时可改为 device=0
        patience=30,
    )
    print("训练完成。权重位置：runs/pose/diode_polarity_pose/weights/best.pt")


if __name__ == "__main__":
    main()
