#!/bin/bash
#
# Cross-compile aria-ai-core for Android.
#
# Getting the inference backends to cross-compile from a developer Mac took
# four separate fixes, none of which fail in an obvious place. They are all
# applied below, each with the symptom it cures, so the next person does not
# have to rediscover them:
#
#   1. Homebrew's cargo has no Android std. Symptom: "can't find crate for
#      `core`". Fix: use the rustup cargo.
#   2. The NDK toolchain file defaults ANDROID_ABI to armeabi-v7a, so the C++
#      compiles as 32-bit and only fails at link. Symptom: "ggml.c.o is
#      incompatible with aarch64linux". Fix: cmake/android-arm64.toolchain.cmake.
#   3. bindgen does not know the NDK sysroot. Symptom: "'stdio.h' file not
#      found". Fix: BINDGEN_EXTRA_CLANG_ARGS.
#   4. whisper-rs-sys 0.15.0 asks for -lggml-blas on any macOS *host*, because
#      build.rs:316 uses `cfg!(target_os = "macos")`, which in a build script
#      describes the host and not the target. CMake correctly never builds
#      ggml-blas for Android. Symptom: "could not find native static library
#      `ggml-blas`". Fix: the empty shim archive below. Delete it once upstream
#      switches to CARGO_CFG_TARGET_OS — this does not silently rot, it will
#      just become dead weight.
#
# Building on Linux avoids (4) entirely, since the host is then not macOS.
#
# Usage: ./build-android.sh [debug|release] [feature,feature]

set -euo pipefail
cd "$(dirname "$0")"

MODE="${1:-release}"
FEATURES="${2:-stt-whisper,llm-llama,insights,net,disk-space}"

# ── 1. rustup's cargo, not Homebrew's ───────────────────────────────────────
if [[ "$(command -v cargo)" == "/opt/homebrew/bin/cargo" && -x "$HOME/.cargo/bin/cargo" ]]; then
    export PATH="$HOME/.cargo/bin:$PATH"
    echo "Using rustup cargo: $(command -v cargo)"
fi

# ── NDK discovery ───────────────────────────────────────────────────────────
if [ -z "${ANDROID_NDK_HOME:-}" ]; then
    SDK_NDK_DIR="$HOME/Library/Android/sdk/ndk"
    if [ -d "$SDK_NDK_DIR" ]; then
        ANDROID_NDK_HOME="$SDK_NDK_DIR/$(ls "$SDK_NDK_DIR" | sort -V | tail -1)"
        echo "Auto-detected NDK: $ANDROID_NDK_HOME"
    else
        echo "error: set ANDROID_NDK_HOME (no NDK under $SDK_NDK_DIR)" >&2
        exit 1
    fi
fi
export ANDROID_NDK_HOME
# llama-cpp-sys-2 looks for these three names, not ANDROID_NDK_HOME.
export ANDROID_NDK="$ANDROID_NDK_HOME"
export NDK_ROOT="$ANDROID_NDK_HOME"
export ANDROID_NDK_ROOT="$ANDROID_NDK_HOME"

HOST_TAG="darwin-x86_64"
[ "$(uname -s)" = "Linux" ] && HOST_TAG="linux-x86_64"
TOOLCHAIN="$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/$HOST_TAG"
TC_BIN="$TOOLCHAIN/bin"
API=24

if [ ! -d "$TC_BIN" ]; then
    echo "error: NDK toolchain not found at $TC_BIN" >&2
    exit 1
fi

export CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER="$TC_BIN/aarch64-linux-android$API-clang"
export CC_aarch64_linux_android="$TC_BIN/aarch64-linux-android$API-clang"
export CXX_aarch64_linux_android="$TC_BIN/aarch64-linux-android$API-clang++"
export AR_aarch64_linux_android="$TC_BIN/llvm-ar"

# ── 2. Pin the ABI via a wrapper toolchain file ─────────────────────────────
export CMAKE_TOOLCHAIN_FILE="$PWD/cmake/android-arm64.toolchain.cmake"
export CMAKE_GENERATOR="Unix Makefiles"
export CMAKE_MAKE_PROGRAM="$(command -v make)"

# CMake 4 removed compatibility with cmake_minimum_required(VERSION < 3.5), and
# vendored C sources in this dependency tree still declare one. aria-android's
# build-rust.sh sets the same escape hatch for audiopus_sys.
export CMAKE_POLICY_VERSION_MINIMUM=3.5

# ── 3. Give bindgen the sysroot ─────────────────────────────────────────────
export BINDGEN_EXTRA_CLANG_ARGS="--sysroot=$TOOLCHAIN/sysroot --target=aarch64-linux-android$API"

# ── 4. Shim for the spurious -lggml-blas (macOS hosts only) ─────────────────
SHIM_DIR="target/android-shims"
if [ "$(uname -s)" = "Darwin" ]; then
    mkdir -p "$SHIM_DIR"
    if [ ! -f "$SHIM_DIR/libggml-blas.a" ]; then
        : > "$SHIM_DIR/empty.c"
        "$CC_aarch64_linux_android" -c "$SHIM_DIR/empty.c" -o "$SHIM_DIR/empty.o"
        "$AR_aarch64_linux_android" rcs "$SHIM_DIR/libggml-blas.a" "$SHIM_DIR/empty.o"
        echo "Built ggml-blas shim (works around whisper-rs-sys build.rs:316)"
    fi
    export RUSTFLAGS="${RUSTFLAGS:-} -L native=$PWD/$SHIM_DIR"
fi

CARGO_FLAGS=""
[ "$MODE" = "release" ] && CARGO_FLAGS="--release"

echo "=== aria-ai-core → aarch64-linux-android ($MODE) ==="
echo "    features: $FEATURES"
cargo build --target aarch64-linux-android $CARGO_FLAGS \
    --no-default-features --features "$FEATURES"

SO="target/aarch64-linux-android/$MODE/libaria_ai_core.so"
if [ -f "$SO" ]; then
    echo
    file "$SO"
    ls -lh "$SO" | awk '{print "    size:", $5}'
else
    echo "error: expected $SO" >&2
    exit 1
fi
