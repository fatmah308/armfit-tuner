#!/usr/bin/env bash
#
# Builds TWO Android arm64-v8a binaries of llama.cpp:
#   1. build-baseline/   — generic build, no Arm-specific matmul kernels
#   2. build-kleidiai/   — GGML_CPU_KLEIDIAI=ON, targets DotProd
#
# This is the core "Arm-specific optimization" comparison: the same
# model, same quant level, same device, two different builds of the
# same framework, with Arm's own accelerated kernels toggled on or off.
#
# Requirements:
#   - Android NDK r27 or later (set NDK_PATH below or via env var)
#   - cmake, ninja, git
#
# Usage:
#   NDK_PATH=/path/to/android-ndk-r27 ./setup_llama_cpp.sh /path/to/llama.cpp

set -euo pipefail

LLAMA_DIR="${1:?Usage: setup_llama_cpp.sh /path/to/llama.cpp}"
NDK_PATH="${NDK_PATH:?Set NDK_PATH to your Android NDK r27+ install directory}"
TOOLCHAIN="${NDK_PATH}/build/cmake/android.toolchain.cmake"
ARCH_FLAGS="armv8.2-a+dotprod"

if [ ! -f "${TOOLCHAIN}" ]; then
    echo "Could not find NDK toolchain file at ${TOOLCHAIN}"
    echo "Check NDK_PATH points at the NDK root (the folder containing build/cmake/...)"
    exit 1
fi

cd "${LLAMA_DIR}"

echo "==> Building BASELINE (no Arm-specific matmul kernels) ..."
cmake -B build-baseline -G Ninja \
    -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN}" \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-24 \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CPU_KLEIDIAI=OFF \
    -DLLAMA_CURL=OFF \
    -DLLAMA_BUILD_UI=OFF \
    -DLLAMA_BUILD_WEBUI=OFF
cmake --build build-baseline --config Release --target llama-bench -j"$(nproc)"

echo ""
echo "==> Building KLEIDIAI (arch flags: ${ARCH_FLAGS}) ..."
echo "    Note: if your device's CPU lacks a feature in these flags, the"
echo "    binary can crash at runtime. Check actual support with:"
echo "        adb shell cat /proc/cpuinfo"
cmake -B build-kleidiai -G Ninja \
    -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN}" \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-24 \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CPU_KLEIDIAI=ON \
    -DGGML_CPU_ARM_ARCH="${ARCH_FLAGS}" \
    -DLLAMA_CURL=OFF \
    -DLLAMA_BUILD_UI=OFF \
    -DLLAMA_BUILD_WEBUI=OFF
cmake --build build-kleidiai --config Release --target llama-bench -j"$(nproc)"

echo ""
echo "==> Locating libomp.so (OpenMP runtime) in the NDK ..."
LIBOMP=$(find "${NDK_PATH}" -name "libomp.so" -path "*aarch64*" | head -n 1)
if [ -n "${LIBOMP}" ]; then
    echo "    Found: ${LIBOMP}"
    cp "${LIBOMP}" "${LLAMA_DIR}/build-baseline/bin/"
    cp "${LIBOMP}" "${LLAMA_DIR}/build-kleidiai/bin/"
    echo "    Copied into both build-baseline/bin and build-kleidiai/bin"
else
    echo "    WARNING: could not find libomp.so (aarch64) under the NDK."
    echo "    If the on-device run fails with a missing libomp.so error,"
    echo "    locate it manually and copy it into build-baseline/bin and build-kleidiai/bin."
fi

echo ""
echo "==> Done."
echo "Baseline binary:  ${LLAMA_DIR}/build-baseline/bin/llama-bench"
echo "KleidiAI binary:  ${LLAMA_DIR}/build-kleidiai/bin/llama-bench"
echo ""
echo "Pass these two paths to armfit via --baseline-bin and --kleidiai-bin"