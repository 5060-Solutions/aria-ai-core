# CMake toolchain wrapper for Android arm64, used when cross-compiling the
# inference backends (whisper.cpp, llama.cpp) from a developer machine.
#
# Why this file exists rather than just setting environment variables:
#
# The `cmake` crate forwards CMAKE_TOOLCHAIN_FILE from the environment, but it
# has no mechanism to pass an arbitrary -D flag through to a dependency's build
# script. The NDK's own android.toolchain.cmake defaults ANDROID_ABI to
# armeabi-v7a when nothing sets it, so the C++ sources compile happily as 32-bit
# ARM and only fail much later at link time with
#
#     ld.lld: error: ...(ggml.c.o) is incompatible with aarch64linux
#
# which points at the linker rather than at the real cause. Pinning the ABI here,
# before including the NDK file, is the documented way to control it.
#
# ANDROID_NDK_HOME must be exported by the caller; see build-android.sh.

set(ANDROID_ABI arm64-v8a CACHE STRING "" FORCE)
set(ANDROID_PLATFORM android-24 CACHE STRING "" FORCE)

if(NOT DEFINED ENV{ANDROID_NDK_HOME})
  message(FATAL_ERROR "ANDROID_NDK_HOME is not set")
endif()

include("$ENV{ANDROID_NDK_HOME}/build/cmake/android.toolchain.cmake")
