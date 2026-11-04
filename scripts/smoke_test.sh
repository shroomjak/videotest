#!/usr/bin/env bash
# Проверка проекта без камеры и без настоящих весов модели.
#
#   ./scripts/smoke_test.sh [-DOpenCV_DIR=... -DONNXRUNTIME_ROOT=...]
#
# Шаги: сборка -> тесты -> генерация модели-заглушки -> прогон конвейера
# на синтетическом источнике -> проверка корректного выхода по SIGINT.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${ROOT}/build"

echo "==> 1/4 сборка"
cmake -S "${ROOT}" -B "${BUILD}" -DCMAKE_BUILD_TYPE=Release "${@}" > /dev/null
cmake --build "${BUILD}" -j"$(nproc)" > /dev/null
echo "    ok"

echo "==> 2/4 тесты"
ctest --test-dir "${BUILD}" --output-on-failure | tail -3

echo "==> 3/4 прогон с моделью-заглушкой"
STUB="${BUILD}/stub-raw.onnx"
python3 "${ROOT}/scripts/make_stub_model.py" --classes 3 --output "${STUB}" > /dev/null
"${BUILD}/videotest" --synthetic --width 640 --height 480 \
    --model "${STUB}" --classes 3 --out-layout raw \
    --no-window --max-frames 60 --snapshot "${BUILD}/check.png"
test -s "${BUILD}/check.png" || { echo "снимок не создан"; exit 1; }
echo "    ok (${BUILD}/check.png)"

echo "==> 4/4 остановка по SIGINT"
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
