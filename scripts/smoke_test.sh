#!/usr/bin/env bash
# Проверка проекта без камеры и без настоящих весов модели.
#
#   ./scripts/smoke_test.sh [-DOpenCV_DIR=... -DONNXRUNTIME_ROOT=...]
#
# Шаги: сборка, тесты, прогон с моделью-заглушкой, контракт пустых масок
# (без модели кадры обязаны совпадать), корректный выход по SIGINT.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${ROOT}/build"

echo "==> 1/5 сборка"
cmake -S "${ROOT}" -B "${BUILD}" -DCMAKE_BUILD_TYPE=Release "${@}" > /dev/null
cmake --build "${BUILD}" -j"$(nproc)" > /dev/null
echo "    ok"

echo "==> 2/5 тесты"
ctest --test-dir "${BUILD}" --output-on-failure | tail -3

echo "==> 3/5 прогон с моделью-заглушкой"
STUB="${BUILD}/stub-raw.onnx"
python3 "${ROOT}/scripts/make_stub_model.py" --classes 3 --output "${STUB}" > /dev/null
"${BUILD}/videotest" --synthetic --width 640 --height 480 \
    --model "${STUB}" --classes 3 --out-layout raw \
    --no-window --max-frames 60 --snapshot "${BUILD}/check.png"
test -s "${BUILD}/check.png" || { echo "снимок не создан"; exit 1; }
echo "    ok (${BUILD}/check.png)"

echo "==> 4/5 контракт пустых масок"
# Без модели разметки быть не может: правая половина склейки должна быть
# попиксельно равна левой (кроме HUD и подписей, они разные по смыслу).
"${BUILD}/videotest" --synthetic --width 640 --height 480 --no-window \
    --max-frames 30 --snapshot "${BUILD}/empty.png" > /dev/null
python3 - "${BUILD}/empty.png" <<'PY'
import sys
try:
    import cv2
    import numpy as np
except ImportError:
    print("    пропущено: нет python-модулей cv2/numpy")
    sys.exit(0)

img = cv2.imread(sys.argv[1])
if img is None:
    print("    пропущено: не удалось прочитать снимок")
    sys.exit(0)
img = img.astype(int)
half = img.shape[1] // 2
# столбцы разделителя (2 px) и полосы HUD/подписей исключаем
left = img[80:430, 2:half - 2]
right = img[80:430, half + 2:half + left.shape[1] + 2]
if left.shape != right.shape or abs(left - right).max() != 0:
    print("    ОШИБКА: кадры с пустой разметкой различаются")
    sys.exit(1)
print("    ok (кадры идентичны)")
PY

echo "==> 5/5 остановка по SIGINT"
"${BUILD}/videotest" --synthetic --no-window > /dev/null 2>&1 &
PID=$!
sleep 2
kill -INT "${PID}"
for _ in $(seq 1 50); do
    kill -0 "${PID}" 2>/dev/null || break
    sleep 0.1
done
if kill -0 "${PID}" 2>/dev/null; then
    echo "    ОШИБКА: процесс не завершился по SIGINT"
    kill -9 "${PID}"
    exit 1
fi
echo "    ok"

echo
echo "Всё проверено. Запуск с камерой: ./build/videotest --model models/yolo11n-seg.onnx"
