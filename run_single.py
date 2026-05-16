from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.pipeline import CircuitPolarityPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="单张原理图极性检测")
    parser.add_argument("image", help="输入原理图图片路径")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    parser.add_argument("--out", default="results", help="输出目录")
    args = parser.parse_args()

    pipe = CircuitPolarityPipeline(args.config)
    result = pipe.run(args.image, args.out)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    print(f"\n标注结果图：{result['result_image']}")


if __name__ == "__main__":
    main()
