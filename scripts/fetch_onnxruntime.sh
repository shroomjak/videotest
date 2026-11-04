#!/usr/bin/env bash
# Скачивает официальный ONNX Runtime C/C++ SDK в third_party/onnxruntime.
#
#   ./scripts/fetch_onnxruntime.sh [версия] [linux-x64|linux-aarch64]
#
# SDK — это tar.gz с include/ и lib/, распакованный в third_party/onnxruntime.
# Каталог третьих сторон не коммитится (см. .gitignore).
set -euo pipefail

VERSION="${1:-1.20.1}"
ARCH="${2:-linux-x64}"           # linux-x64 | linux-aarch64
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${ROOT}/third_party/onnxruntime"
ARCHIVE="onnxruntime-${ARCH}-${VERSION}.tgz"
URL="https://github.com/microsoft/onnxruntime/releases/download/v${VERSION}/${ARCHIVE}"

mkdir -p "${ROOT}/third_party"
echo "==> скачивание ${URL}"
curl -fL --retry 3 -o "/tmp/${ARCHIVE}" "${URL}"

TMPDIR="$(mktemp -d)"
tar -xzf "/tmp/${ARCHIVE}" -C "${TMPDIR}"
SRC="${TMPDIR}/onnxruntime-${ARCH}-${VERSION}"

rm -rf "${DEST}"
mkdir -p "${DEST}"
cp -r "${SRC}/include" "${DEST}/include"
cp -r "${SRC}/lib" "${DEST}/lib"
rm -rf "${TMPDIR}" "/tmp/${ARCHIVE}"

echo "==> готово: ${DEST}"
echo "    заголовки: $(ls "${DEST}/include" | wc -l) файлов"
echo "    библиотеки: $(ls "${DEST}/lib" | tr '\n' ' ')"
echo
echo "Теперь можно собрать:"
echo "    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j"
