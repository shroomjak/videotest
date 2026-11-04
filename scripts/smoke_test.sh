#!/usr/bin/env bash
# Сквозная проверка проекта без камеры и без настоящих весов модели.
#
# Что проверяется:
#   1. сборка;
#   2. юнит-тесты (очереди, NMS/IoU, геометрия letterbox);
#   3. запуск конвейера на синтетическом источнике с моделью-заглушкой;
#   4. корректный выход по Ctrl+C (SIGINT) без зависания join().
#
# Требуется: cmake, OpenCV, ONNX Runtime SDK в third_party/onnxruntime
# (или -DONNXRUNTIME_ROOT=...), python3 с пакетом onnx — только для заглушки.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${ROOT}/build"

echo "==> 1/4 конфигурация и сборка"
cmake -S "${ROOT}" -B "${BUILD}" -DCMAKE_BUILD_TYPE=Release "${@}" > /dev/null
cmake --build "${BUILD}" -j"$(nproc)" > /dev/null
echo "    ok"

echo "==> 2/4 юнит-тесты"
ctest --test-dir "${BUILD}" --output-on-failure | tail -4

echo "==> 3/4 модель-заглушка и запуск конвейера"
STUB="${BUILD}/stub-seg.onnx"
python3 "${ROOT}/scripts/make_stub_model.py" --classes 3 --output "${STUB}" > /dev/null
"${BUILD}/videotest" \
    --synthetic --width 640 --height 480 --fps 30 \
    --model "${STUB}" \
    --headless --max-frames 30 \
    --snapshot "${BUILD}/check.png" \
    --record "${BUILD}/check.mp4" \
    --log-level info 2>&1 | tail -4
test -s "${BUILD}/check.png" || { echo "снимок не создан"; exit 1; }
echo "    ok (снимок: ${BUILD}/check.png)"

echo "==> 4/4 остановка по SIGINT"
"${BUILD}/videotest" --synthetic --model "${STUB}" --log-level warn &
PID=$!
sleep 2
kill -INT "${PID}"
# ждём не дольше 10 секунд: если процесс завис на join(), это баг
for _ in $(seq 1 100); do
    if ! kill -0 "${PID}" 2>/dev/null; then break; fi
    sleep 0.1
done
if kill -0 "${PID}" 2>/dev/null; then
    echo "    ОШИБКА: процесс не завершился по SIGINT"
    kill -9 "${PID}"
    exit 1
fi
wait "${PID}" 2>/dev/null || true
echo "    ok (завершился по SIGINT)"

echo
echo "Всё проверено. Для работы с камерой и настоящей моделью смотрите README."
