#!/usr/bin/env bash
# Скачивает SDK ONNX Runtime (include/ + lib/) в third_party/onnxruntime.
#   ./scripts/fetch_onnxruntime.sh [версия] [linux-x64|linux-aarch64]
set -euo pipefail

VERSION="${1:-1.20.1}"
ARCH="${2:-linux-x64}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${ROOT}/third_party/onnxruntime"
ARCHIVE="onnxruntime-${ARCH}-${VERSION}.tgz"
URL="https://github.com/microsoft/onnxruntime/releases/download/v${VERSION}/${ARCHIVE}"

mkdir -p "${ROOT}/third_party"
echo "==> скачивание ${URL}"
curl -fL --retry 3 -o "/tmp/${ARCHIVE}" "${URL}"

TMP="$(mktemp -d)"
tar -xzf "/tmp/${ARCHIVE}" -C "${TMP}"
rm -rf "${DEST}"
mkdir -p "${DEST}"
cp -r "${TMP}/onnxruntime-${ARCH}-${VERSION}/include" "${DEST}/include"
cp -r "${TMP}/onnxruntime-${ARCH}-${VERSION}/lib" "${DEST}/lib"
rm -rf "${TMP}" "/tmp/${ARCHIVE}"

echo "==> готово: ${DEST}"
echo "    собирать так: cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j"
