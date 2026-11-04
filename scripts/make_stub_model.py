#!/usr/bin/env python3
"""Генератор заглушки ONNX-модели instance-сегментации для проверки конвейера.

Настоящие веса YOLO не нужны, чтобы проверить всё, кроме качества модели:
формы выходов, декодирование, сборку масок, letterbox и отрисовку. Заглушка
воспроизводит ровно те же формы тензоров, что и настоящая модель:

    input:  images   [1, 3, 640, 640]        float32
    output: output0  [1, 4 + nc + 32, 8400]  (вариант "сырой", YOLOv8/11-seg)
    output: output1  [1, 32, 160, 160]       (прототипы масок)

Значения выходов — линейные комбинации средних по изображению цветов. То есть
результат зависит от входа (детерминированно), но никаких весов не требует.

Использование:
    python3 scripts/make_stub_model.py --classes 2 --output models/stub-seg.onnx
"""
import argparse
import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

IMG_SIZE = 640


def build_model(num_classes: int, anchors: int = 8400, protos: int = 32,
                proto_size: int = 160, img_size: int = IMG_SIZE,
                end2end: bool = False, detections: int = 300) -> onnx.ModelProto:
    # "сырой" выход: [1, 4+nc+32, anchors]; end-to-end: [1, detections, 6+32]
    channels = (6 + protos) if end2end else (4 + num_classes + protos)
    pred_axis_len = detections if end2end else anchors
    rng = np.random.default_rng(1234)

    # weights[3, channels]: MatMul([1,3] x [3,channels]) -> [1,channels]
    w_pred = rng.normal(0.0, 0.08, size=(3, channels)).astype(np.float32)
    # Смещения подобраны так, чтобы сырой выход давал несколько кандидатов выше
    # порога: центры в середине кадра, размеры ~1/4 кадра, уверенность ~0.3-0.9.
    b_pred = np.zeros((channels,), dtype=np.float32)
    b_pred[0] = img_size * 0.5   # cx
    b_pred[1] = img_size * 0.55  # cy
    b_pred[2] = img_size * 0.25  # w
    b_pred[3] = img_size * 0.30  # h
    b_pred[4] = 0.75             # уверенность класса 0
    if num_classes > 1:
        b_pred[4 + 1] = 0.45     # уверенность класса 1
    w_proto = rng.normal(0.0, 0.08, size=(3, protos)).astype(np.float32)
    b_proto = rng.normal(0.0, 0.4, size=(protos,)).astype(np.float32)

    # Пространственный «базис» прототипов: без него все каналы прототипов были бы
    # константой по изображению, и маска после порога оказалась бы либо пустой,
    # либо залитой целиком. Радиальные пятна разного размера/смещения дают
    # реалистичную картину для проверки отрисовки.
    proto_base = np.zeros((1, protos, proto_size, proto_size), dtype=np.float32)
    yy, xx = np.mgrid[0:proto_size, 0:proto_size].astype(np.float32)
    for k in range(protos):
        cx = proto_size * (0.5 + 0.35 * rng.uniform(-1, 1))
        cy = proto_size * (0.5 + 0.35 * rng.uniform(-1, 1))
        radius = proto_size * rng.uniform(0.08, 0.35)
        proto_base[0, k] = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * radius ** 2))

    initializers = [
        numpy_helper.from_array(w_pred, "w_pred"),
        numpy_helper.from_array(b_pred, "b_pred"),
        numpy_helper.from_array(w_proto, "w_proto"),
        numpy_helper.from_array(b_proto, "b_proto"),
        numpy_helper.from_array(np.array([1, channels, pred_axis_len], dtype=np.int64),
                                "pred_shape"),
        numpy_helper.from_array(np.array([1, protos, proto_size, proto_size], dtype=np.int64),
                                "proto_shape"),
        numpy_helper.from_array(np.array([2, 3], dtype=np.int64), "squeeze_axes"),
        numpy_helper.from_array(np.array([2], dtype=np.int64), "unsqueeze_axes"),
        numpy_helper.from_array(np.array([2, 3], dtype=np.int64), "unsqueeze_axes4"),
        numpy_helper.from_array(np.array([1, channels, 1], dtype=np.int64), "pred_unsqueeze"),
        numpy_helper.from_array(np.array([1, protos, 1], dtype=np.int64), "proto_unsqueeze"),
        numpy_helper.from_array(np.array([1, protos, 1, 1], dtype=np.int64),
                                "proto_unsqueeze4"),
        numpy_helper.from_array(np.array([1, 1, proto_size, proto_size], dtype=np.int64),
                                "proto_spatial"),
        numpy_helper.from_array(proto_base, "proto_base"),
    ]

    nodes = [
        # [1,3,640,640] -> [1,3,1,1]
        helper.make_node("GlobalAveragePool", ["images"], ["pooled"], name="pool"),
        # [1,3,1,1] -> [1,3]
        # в opset 13+ оси Squeeze/Unsqueeze передаются входом, а не атрибутом
        helper.make_node("Squeeze", ["pooled", "squeeze_axes"], ["flat"], name="squeeze_flat"),
        # [1,3] x [3,channels] -> [1,channels]
        helper.make_node("MatMul", ["flat", "w_pred"], ["pred_matmul"], name="pred_matmul"),
        helper.make_node("Add", ["pred_matmul", "b_pred"], ["pred_vec"], name="pred_add"),
        # [1,channels] -> [1,channels,1] -> [1,channels,anchors]
        helper.make_node("Unsqueeze", ["pred_vec", "unsqueeze_axes"], ["pred_vec3"],
                         name="pred_unsqueeze"),
        helper.make_node("Expand", ["pred_vec3", "pred_shape"], ["pred_expand0"],
                         name="pred_expand"),
    ]
    if end2end:
        # end-to-end выход имеет вид [1, N, 6+32] (как у YOLO26), поэтому
        # транспонируем развёрнутый тензор: [1, ch, N] -> [1, N, ch].
        nodes.append(helper.make_node("Transpose", ["pred_expand0"], ["output0"],
                                      name="pred_transpose", perm=[0, 2, 1]))
    else:
        nodes.append(helper.make_node("Identity", ["pred_expand0"], ["output0"],
                                      name="pred_identity"))

    nodes = nodes[:8] + [
        # прототипы: [1,32] -> [1,32,1,1], масштабируем пространственный базис
        helper.make_node("MatMul", ["flat", "w_proto"], ["proto_matmul"], name="proto_matmul"),
        # то же для прототипов
        helper.make_node("Add", ["proto_matmul", "b_proto"], ["proto_vec"], name="proto_add"),
        helper.make_node("Unsqueeze", ["proto_vec", "unsqueeze_axes4"], ["proto_vec4"],
                         name="proto_unsqueeze"),
        helper.make_node("Mul", ["proto_base", "proto_vec4"], ["proto_scaled"], name="proto_mul"),
        helper.make_node("Add", ["proto_base", "proto_scaled"], ["output1"], name="proto_out"),
    ]

    graph = helper.make_graph(
        nodes,
        "stub_seg",
        inputs=[helper.make_tensor_value_info("images", TensorProto.FLOAT,
                                              [1, 3, img_size, img_size])],
        outputs=[
            helper.make_tensor_value_info(
                "output0", TensorProto.FLOAT,
                [1, pred_axis_len, channels] if end2end else [1, channels, pred_axis_len]),
            helper.make_tensor_value_info("output1", TensorProto.FLOAT,
                                          [1, protos, proto_size, proto_size]),
        ],
        initializer=initializers,
    )

    model = helper.make_model(graph, producer_name="videotest-stub",
                              opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    entry = model.metadata_props.add()
    entry.key = "names"
    entry.value = str({i: f"class_{i}" for i in range(num_classes)})
    onnx.checker.check_model(model)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classes", type=int, default=2)
    parser.add_argument("--output", default="models/stub-seg.onnx")
    parser.add_argument("--anchors", type=int, default=8400)
    parser.add_argument("--proto-size", type=int, default=160)
    parser.add_argument("--end2end", action="store_true",
                        help="формат YOLO26: [1,300,6+32], NMS уже в графе")
    args = parser.parse_args()

    model = build_model(args.classes, anchors=args.anchors, proto_size=args.proto_size,
                        end2end=args.end2end)
    directory = os.path.dirname(args.output)
    if directory:
        os.makedirs(directory, exist_ok=True)
    onnx.save(model, args.output)
    print(f"готово: {args.output}")
    print("вход:  images  [1,3,640,640]")
    print("использование в проекте:  --model <файл> --classes %d --out-layout %s"
          % (args.classes, "nms" if args.end2end else "raw"))
    if args.end2end:
        print(f"выход: output0 [1,300,{6 + 32}] (end-to-end), "
              f"output1 [1,32,{args.proto_size},{args.proto_size}]")
    else:
        print(f"выход: output0 [1,{4 + args.classes + 32},{args.anchors}], "
              f"output1 [1,32,{args.proto_size},{args.proto_size}]")


if __name__ == "__main__":
    main()
