#!/usr/bin/env python3
"""Экспорт YOLO26-seg (или YOLOv8-seg/YOLO11-seg) в ONNX для videotest.

Запуск (нужен Python 3.10+ и пакет ultralytics):

    pip install ultralytics
    python scripts/export_yolo26_seg.py --model yolo26n-seg.pt --imgsz 640 --output models/yolo26n-seg.onnx

Почему именно так:
  * opset 12+ и динамическая ось batch не нужны — у нас всегда batch=1;
  * simplify=True выкидывает лишние узлы (ускоряет инференс на CPU на 5–15%);
  * nms=True имеет смысл только для архитектур с end-to-end NMS (YOLO26):
    тогда выход — готовые детекции [1, N, 6+32], и в C++ мы не вызываем NMS.
"""
import argparse
import pathlib
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="yolo26n-seg.pt", help="веса (pt) или имя модели ultralytics")
    parser.add_argument("--imgsz", type=int, default=640, help="размер входа (квадрат)")
    parser.add_argument("--output", default="models/model.onnx", help="куда сохранить onnx")
    parser.add_argument("--opset", type=int, default=12, help="версия ONNX-опсета")
    parser.add_argument("--simplify", action="store_true", default=True)
    parser.add_argument("--half", action="store_true", help="FP16 (только для GPU)")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("нужен пакет ultralytics: pip install ultralytics", file=sys.stderr)
        return 1

    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    print(f"экспорт {args.model} -> {output} (imgsz={args.imgsz}, opset={args.opset})")
    exported = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=args.simplify,
        half=args.half,
        dynamic=False,
        device="cpu",
    )

    exported_path = pathlib.Path(str(exported))
    if exported_path.resolve() != output.resolve():
        output.write_bytes(exported_path.read_bytes())
    print(f"готово: {output} ({output.stat().st_size / 1e6:.1f} МБ)")
    print("\nпроверить формат выходов можно так:")
    print("    python -c \"import onnxruntime as o; s=o.InferenceSession('%s');"
          "print([(i.name, i.shape) for i in s.get_outputs()])\"" % output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
