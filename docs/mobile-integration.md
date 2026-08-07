# Mobile integration — audit and build notes

Findings from cross-compiling `aria-ai-core` for Android and iOS, 2026-07-29.
Everything below was verified by building, not by reading documentation.

## Build matrix

| | Android arm64 | iOS arm64 (device) |
|---|---|---|
| Core (SQLite, rustls, UniFFI) | ✅ | ✅ |
| `stt-whisper` (whisper.cpp) | ✅ after four fixes | ✅ |
| `llm-llama` (llama.cpp) | ✅ | ✅ |
| **both together** | ✅ via shared ggml | ✅ via shared ggml |
| `stt-parakeet` (sherpa-onnx) | plausible, untested | ❌ unsupported |

## Parakeet cannot target iOS devices

`sherpa-rs-sys` 0.6.8 maps exactly two mobile targets (`build.rs:26-29`):
`aarch64-linux-android` and `aarch64-apple-ios-**sim**`. There is no
`aarch64-apple-ios`, so the device build has no prebuilt onnxruntime and fails
at link with missing search paths. Simulator only.

Since the model research names Parakeet the primary STT, this is a product
constraint, not a build detail: whisper.cpp is the only STT that runs on both
platforms today.

## The ggml collision

whisper.cpp and llama.cpp each vendor their own copy of ggml. Linking both into
one binary fails:

```
ld.lld: error: duplicate symbol: ggml_backend_buffer_set_usage
```

iOS is more dangerous, because it looks like it works. A Rust `staticlib` is
just an archive, so the build succeeds — but the archive carries **two**
definitions:

```console
$ nm -g target/aarch64-apple-ios/debug/libaria_ai_core.a | grep -c "T _ggml_backend_buffer_set_usage"
2
```

The conflict is deferred to Xcode's final link, where one copy is silently
discarded. The two copies are **different versions** — whisper's `ggml.h` is
2724 lines / 359 `GGML_API` declarations, llama's is 2845 / 363 — so
`--allow-multiple-definition` is not a workaround: it would let one library call
the other's ggml across mismatched struct layouts.

Both upstreams anticipate this. `whisper.cpp/CMakeLists.txt:71` offers
`WHISPER_USE_SYSTEM_GGML` and `llama.cpp/CMakeLists.txt:34` offers
`LLAMA_USE_SYSTEM_GGML`. Neither Rust `-sys` wrapper exposes the option, which
is the gap to close.

### Sharing one ggml is feasible — verified

The obvious risk in that plan is API drift: whisper.cpp and llama.cpp pin
different ggml revisions, so whisper might simply not compile against llama's.
It does. Built directly with CMake, outside cargo:

1. llama.cpp's ggml (version 0.13.1) built and installed to a prefix. It must be
   added via `add_subdirectory` from a wrapper `CMakeLists.txt` rather than
   configured as the top-level project — ggml forces `GGML_STANDALONE=ON` when it
   *is* the top-level project, and standalone mode wants `tests/`, `examples/`
   and `ggml.pc.in`, all of which the crates.io package strips out. Its install
   rules and `ggml-config.cmake` live outside that guard, so a subproject build
   still installs a usable package.
2. whisper.cpp built with `-DWHISPER_USE_SYSTEM_GGML=ON` and `-Dggml_DIR=<prefix>`
   → **compiles and links**, producing `libwhisper.a` with no ggml of its own.

### Resolved — the `stt-whisper-with-llm` feature

Patched crates live in the sibling `vendor/` repo and are wired in via
`[patch.crates-io]`. It turned out no second ggml build is needed: llama.cpp
already installs `lib/cmake/ggml` next to its own libraries, so whisper.cpp can
simply be pointed at it.

`llama-cpp-sys-2` publishes that path, and because it declares `links = "llama"`
cargo exposes it to dependents as `DEP_LLAMA_GGML_DIR`:

```rust
println!("cargo:ggml_dir={}", build_dir.join("lib/cmake/ggml").display());
```

`whisper-rs-sys` gains an optional `system-ggml` feature which takes a **normal**
dependency on `llama-cpp-sys-2` — normal rather than build-, because cargo only
exports a `links` crate's metadata to crates that depend on it normally, and it
also forces the build order. When `DEP_LLAMA_GGML_DIR` is set it builds with
`WHISPER_USE_SYSTEM_GGML=ON` and stops emitting its own `-lggml*`.

Verified: `--features stt-whisper-with-llm` links on the host, Android arm64 and
iOS arm64, and the combined iOS static library contains **one** definition of
`ggml_init` and `ggml_backend_buffer_set_usage`, where it previously had two.

The maintenance cost is real: on any whisper.cpp or llama.cpp bump the two ggml
revisions must still be compatible, so re-run that symbol count after upgrading.

## Android cross-compilation: four fixes

All applied by `build-android.sh`; each is commented there with its symptom.
None of them fails anywhere near its cause.

1. **Homebrew's cargo has no Android std.** → `can't find crate for 'core'`.
   Use the rustup cargo.
2. **The NDK toolchain file defaults `ANDROID_ABI` to `armeabi-v7a`.** The C++
   compiles cleanly as 32-bit and only fails at link with
   `ggml.c.o is incompatible with aarch64linux`. The `cmake` crate forwards
   `CMAKE_TOOLCHAIN_FILE` from the environment but cannot pass a `-D` flag to a
   dependency's build script, so the ABI is pinned by
   `cmake/android-arm64.toolchain.cmake`, which sets it and then includes the
   NDK file.
3. **bindgen does not know the NDK sysroot.** → `'stdio.h' file not found`.
   Fixed with `BINDGEN_EXTRA_CLANG_ARGS`.
4. **Upstream bug in `whisper-rs-sys` 0.15.0.** `build.rs:316` reads
   `cfg!(target_os = "macos")`, but inside a build script `cfg!` describes the
   **host**, not the target. Cross-compiling from a Mac therefore emits
   `-lggml-blas` for a library CMake correctly never builds on Android →
   `could not find native static library 'ggml-blas'`. The correct form is
   `env::var("CARGO_CFG_TARGET_OS")`. A build on Linux never sees this.

`llama-cpp-sys-2` looks for `ANDROID_NDK`, `NDK_ROOT` or `ANDROID_NDK_ROOT` —
not `ANDROID_NDK_HOME` — and panics with a clear message if none is set.

Two more surfaced when building `aria-mobile-core` (rather than `aria-ai-core`
alone) for each platform:

5. **CMake 4 dropped `cmake_minimum_required(VERSION < 3.5)`.** `audiopus_sys`
   vendors an Opus that still declares one, so the Android build of
   `aria-mobile-core` dies in Opus, nowhere near the AI code.
   `CMAKE_POLICY_VERSION_MINIMUM=3.5` restores it — `aria-android/build-rust.sh`
   already sets this, so a standalone script has to as well.
6. **Rust defaults `aarch64-apple-ios` to a 10.0 deployment target.**
   whisper.cpp compiles against the current SDK and emits `__chkstk_darwin`,
   which does not exist that far back, so the link fails with
   `Undefined symbols for architecture arm64: ___chkstk_darwin`. The app targets
   iOS 16 (`project.yml`), so export `IPHONEOS_DEPLOYMENT_TARGET=16.0`.

## Verified: audio reaches the AI

`aria-mobile-core` builds with `--features ai` for **both** targets
(`libaria_mobile_core.so`, ARM aarch64; iOS static lib, 257 MB debug), and six
tests in `media.rs` drive the real path — a live `MobileMediaSession`, genuine
RTP at its socket — rather than calling the tap directly:

- far-end RTP reaches the AI *and* still reaches the speaker
- near-end microphone audio reaches the AI (the positive control)
- a muted microphone is silence to the AI, so the transcript cannot contain
  words the far end never heard
- detaching the tap stops the flow without disturbing the call

Note that `push_local`/`push_remote` only fill a ring buffer; `captured_ms`
counts what `pump()` has drained and resampled. A test that asserts on stats
without pumping passes whether or not the tap is wired at all.

## Why the AI core is a dependency of aria-mobile-core

`aria-mobile-core` already decodes both call legs: `media.rs:206` hands remote
PCM to `on_playback_audio`, and `media.rs:261` receives local mic PCM from
`on_capture_audio`. Both are on the Rust side of the FFI boundary.

Having the apps tap those callbacks in Kotlin/Swift and push samples back into
`AriaAi` would move every 20 ms frame across FFI twice — roughly a hundred
crossings per second per call — which is what this crate's own design notes warn
against. Instead `aria-mobile-core` gains an optional `ai` feature, depends on
`aria-ai-core`, and taps the PCM where it already sits. The apps only ever see
control calls and finished transcripts.

The feature is off by default: it pulls in whisper.cpp and llama.cpp, a slow C++
build that a host not shipping AI should not pay for.

## Android packaging notes

- `build-rust.sh` currently produces three ABIs (arm64, armv7, x86_64). The
  inference backends should be **arm64-only**: a 32-bit armv7 device cannot
  usefully run a 600 MB model, and x86_64 is emulator-only.
- `minSdk` is 26; the NDK API level used here is 24, so that is consistent.
