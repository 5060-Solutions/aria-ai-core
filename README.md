# aria-ai-core

On-device call transcription and summarization for the Aria softphone family
(desktop `softphone`, `aria-ios`, `aria-android`).

Everything runs locally. The only network capability in the entire crate is the
model downloader, and it is restricted to a compile-time host allowlist
(`models::ALLOWED_HOSTS`). No call audio, transcript or summary ever leaves the
device through this crate.

Implements Phase 1 of `docs/superpowers/specs/2026-07-28-on-device-ai-design.md`.

## What it does

```text
push_local / push_remote  ──►  CallAudioTap        two-leg ring buffers, realtime-safe
        │                      SincResampler       band-limited 8/16/32/48 kHz → 16 kHz
        ▼
      pump()               ──►  segment_call        energy VAD + cross-leg echo gate
        │                                           → speaker-attributed, timed segments
        ▼
   transcribe()            ──►  stt::Transcriber    per-segment STT, audio freed on return
        │
        ▼
   summarize()             ──►  llm::Summarizer     map-reduce for long calls, GBNF-constrained
        │                                           JSON, validated and clamped in Rust
        ▼
    insight()              ──►  CallInsight         what the Call Insights screen renders
```

The two call legs are never summed, so **speaker attribution is free** — `Local`
is the microphone, `Remote` is the decoded far end.

Two properties the code enforces rather than documents:

- **The captured audio is dropped the moment transcription finishes**, on every
  exit path including failure. A `CallSession` never holds call audio after it
  has produced a transcript.
- **Nothing panics.** This crate is called across an FFI boundary where a Rust
  unwind is undefined behaviour. Every fallible public function returns
  `AiError`; `unwrap`/`expect`/`panic!` are `deny`-linted outside `#[cfg(test)]`
  code; mutex poisoning is recovered from rather than propagated. That extends
  to the native backends, which have their own `expect`s on inputs we control:
  `LlamaSummarizer::load` checks the file exists before `llama-cpp-2`'s
  `debug_assert!` can fire, and `WhisperTranscriber` rejects a language tag with
  an interior NUL before `whisper-rs` turns it into a `CString`. Both have
  regression tests.

## Feature flags

Default features pull in nothing larger than `reqwest` and a bundled SQLite.
Verify with `cargo tree` — a default build contains no `*-sys` inference crate.

| Feature | Default | Pulls in | What it gives you |
|---|:-:|---|---|
| `net` | ✅ | `reqwest` (rustls) | HTTPS model downloads. Off ⇒ supply your own `models::ModelSource`. |
| `disk-space` | ✅ | `fs4` | Real free-space preflight. Off ⇒ supply your own `models::DiskSpaceProbe`. |
| `insights` | ✅ | `rusqlite` (bundled), `aes-gcm`, `rand` | Encrypted `InsightStore`. Off ⇒ this crate persists nothing. |
| `uniffi` | ❌ | `uniffi` | Swift/Kotlin scaffolding and the `ffi` module. Desktop never needs it. |
| `stt-parakeet` | ❌ | `sherpa-rs` → sherpa-onnx, onnxruntime | Parakeet-TDT-0.6B-v2. **Recommended primary STT.** |
| `stt-whisper` | ❌ | `whisper-rs` → whisper.cpp | Whisper GGML. Secondary / "back pocket". |
| `llm-llama` | ❌ | `llama-cpp-2` → llama.cpp | Qwen3-Instruct GGUF with GBNF-constrained JSON. |
| `inference` | ❌ | — | `stt-parakeet` + `llm-llama`. |

```sh
cargo test                          # no models, no network, no C++
cargo build --features stt-parakeet # sherpa-onnx / Parakeet-TDT
cargo build --features stt-whisper  # whisper.cpp
cargo build --features llm-llama    # llama.cpp (Metal on Apple silicon)
cargo build --features uniffi       # Swift/Kotlin scaffolding
```

`llama-cpp-2` explicitly does not follow semver and is pinned exactly.

Without any inference feature the backends are still *reachable* — the engine
returns `AiError::FeatureNotEnabled("stt-parakeet")` rather than silently doing
nothing — and `stt::MockTranscriber` / `llm::MockSummarizer` let the whole
pipeline be driven deterministically in tests.

## How each platform consumes it

### Desktop (`softphone`, Tauri v2) — plain Rust dependency

Tauri is Rust; there is no FFI in the way, so the desktop depends on this crate
directly and uses the native API. Do **not** enable `uniffi`: it costs a binding
layer the desktop has no use for.

```toml
[dependencies]
aria-ai-core = { path = "../aria-ai-core", features = ["inference"] }
```

```rust
let engine = AiEngine::new(AiConfig::new(app_data_dir.join("ai"), capability))?;
let session = engine.begin_call(SessionConfig::new(call_id))?;
let _drain = session.spawn_drain_thread(Duration::from_millis(100));
// rtp-engine's PcmSink feeds session.push_local / push_remote
session.stop()?;
let transcript = session.transcribe(engine.transcriber(stt_id)?.as_ref())?;
let summary = session.summarize(
    engine.summarizer(llm_id)?.as_ref(), &ctx, &SummaryConfig::default())?;
```

### iOS / Android — through UniFFI

Built with `--features uniffi`; `src/ffi.rs` exposes two objects, `AriaAi` and
`AriaCall`, plus every result type as a UniFFI record/enum. Generate bindings
with the crate's own bindgen binary:

```sh
cargo build --features uniffi --release
cargo run --features uniffi --bin uniffi-bindgen -- generate \
    --library target/release/libaria_ai_core.dylib \
    --language swift --out-dir generated/
# --language kotlin for Android
```

Two shapes do not cross a language boundary, so the facade re-casts them:

- **Trait objects.** A host cannot pass `&dyn Transcriber`, so the FFI verbs are
  `AriaAi::transcribe_call(call_id, stt_model_id)` and
  `summarize_call(call_id, llm_model_id, context)` — the engine loads the
  backend itself.
- **Tuples.** `CallSession::stats()` returns a pair; the facade returns a
  `CallStats` record.

**Audio should not cross this boundary.** `AriaCall::push_local` / `push_remote`
exist so a host can drive a call with no Rust of its own, but copying every
20 ms frame through UniFFI costs an allocation per frame per leg. The production
mobile path keeps the realtime side in Rust: `aria-mobile-core` holds the
`CallAudioTap` and feeds it directly from `media.rs`, and only the control plane
(model management, session lifecycle, results) crosses into Swift/Kotlin.

### Device capability is supplied by the host

The crate never asks the OS anything — the host fills in `DeviceCapability` and
the crate does all tier gating from it. One iOS-specific trap: the number that
belongs in `available_ram_mb` is `os_proc_available_memory()`, **not**
`ProcessInfo.physicalMemory`. iOS jetsams well below physical RAM, and a VoIP app
killed mid-call is a catastrophic failure.

## Model integrity

The catalogue is compiled into the binary and never fetched, so a compromised or
unreachable CDN cannot inject a model: the id, size, URL and SHA-256 all come
from the signed app binary. Downloads are resumable (the on-disk prefix is
re-hashed on resume), cancellable, and verified before install via an atomic
rename.

Every `sha256` in `catalog.json` is a real digest, taken from the Hugging Face
tree API — for an LFS object the `oid` *is* the SHA-256 — and spot-checked by
downloading a file and hashing it. Entries point at the publisher's own
repository (`ggerganov/whisper.cpp`, `unsloth/Qwen3-*-GGUF`,
`csukuangfj/sherpa-onnx-nemo-parakeet-*`, `onnx-community/silero-vad`).

`ModelDescriptor::has_real_checksum()` still reports the all-zero placeholder and
the downloader **fails closed** on one, so a half-filled catalogue cannot install
an unverified file. A unit test asserts no bundled model carries a placeholder.

Beyond the unit tests, which run against a fake transport, `tests/live_download.rs`
exercises the real thing — download, redirect, resume and digest check against the
live CDN:

```bash
cargo test --test live_download -- --ignored --nocapture
```

### Hosts and redirects

`ALLOWED_HOSTS` is `huggingface.co`, plus the `.hf.co` / `.huggingface.co`
suffixes for its content CDN, which is where a weights request gets redirected
and whose exact hostname varies by region. Redirects are followed manually,
never by the HTTP library, and **every hop is allowlist-checked before it is
dialled** — the guarantee is "this crate only ever connects to an allowlisted
host", and that is preserved hop by hop rather than by refusing redirects and
making downloads impossible.

### Multi-file models

sherpa-onnx models are a directory, not a blob: Parakeet is an encoder, decoder,
joiner and token list. Those entries carry `ModelDescriptor::files`, each member
with its own URL, size and digest, and install into `models/<file_name>/` — which
is exactly what `ParakeetTranscriber::load` wants. Members are fetched and
verified individually, so each is range-resumable on its own, a corrupt member is
re-fetched without disturbing its 650 MB sibling, and nothing needs twice the
disk to unpack. This replaced the original plan of shipping a `.tar.bz2`.

## Stored insights

Transcripts and summaries are the most sensitive thing this crate produces: the
verbatim contents of someone's phone calls. `InsightStore` (feature `insights`,
on by default) keeps them in SQLite with the payload sealed under AES-256-GCM,
so a copy of the database file — a device backup, a pulled disk, a
misconfigured sync — is not a copy of the conversations.

The 32-byte key comes from the host and is never generated, derived, logged or
persisted here: the Keychain on iOS (`…AfterFirstUnlockThisDeviceOnly`), the
Keystore on Android, the OS credential store on desktop. `InsightStore`'s
`Debug` is hand-written so the key cannot reach a log line.

Left in the clear, deliberately, so the list screen can page and sort without
decrypting every row: call id, created-at, duration, and the status
*discriminant*. Everything with words in it — every transcript segment, every
summary, every action item, the model ids, and any failure reason — is inside
the sealed blob. Each row gets its own random 96-bit nonce, and the call id and
schema version are bound in as associated data, so a row lifted from one call
cannot be replayed as another's. All of that is tested, including a check that
the transcript text does not appear anywhere in the database files on disk.

## Command-line harness

`aria-ai-cli` drives the pipeline from a desktop shell, so it can be exercised
without a phone. It is a development and benchmarking tool, not a product
surface.

```bash
cargo run --bin aria-ai-cli -- models                 # catalogue + what this box can run
cargo run --bin aria-ai-cli -- download whisper-base-q5_1
cargo run --bin aria-ai-cli -- verify whisper-base-q5_1

# Transcription and summarisation need the inference backends:
cargo run --features inference --bin aria-ai-cli -- \
    summarize call.wav --stt whisper-base-q5_1 --llm qwen3-1.7b-instruct-q4_k_m

export ARIA_AI_KEY_HEX=$(openssl rand -hex 32)   # what a device reads from its Keychain
cargo run --bin aria-ai-cli -- insights list
```

It reads 16-bit PCM WAV directly (mono, or stereo averaged down) rather than
adding an audio-file dependency to the library's tree.

## Deferred to later phases

Not oversights — scoped out, with the layer they belong to:

| Deferred | Phase | Note |
|---|---|---|
| Measured RAM/speed figures per model | 1 (benchmarking) | Checksums and sizes are real; `runtime_ram_mb` and `quality_tier` are still estimates. |
| Encrypted on-disk spill for long captures | 2 | Capture is RAM-bound by `TapConfig::max_capture_secs` (default 120 min). |
| Serial post-call job queue, thermal/battery policy | 2–3 | Sits *above* `AiEngine`. |
| Consent state machine and capture-mode policy | 3 | Product/legal decision, then UI. |
| Silero VAD | 6 | Current VAD is energy-based with an adaptive noise floor — honestly scoped, not a stub. |
| Live/streaming transcription | 6 | Post-call batch only, by design. |
| Callback/event progress instead of polling | 6 | Downloads and job state are polled. |

## Layout

| Module | Owns |
|---|---|
| `audio` | ring buffers, band-limited resampling to 16 kHz, VAD, speaker-attributed segmentation |
| `models` | catalogue, device gating, resumable/verified download, on-disk store |
| `stt` | the `Transcriber` trait, a deterministic mock, feature-gated Parakeet/Whisper |
| `llm` | the `Summarizer` trait, prompts + GBNF grammar, chunking, the validating map-reduce pipeline |
| `session` | `CallSession` — start/stop, feed frames, transcript, summary |
| `engine` | `AiEngine` — the one object each platform binds to |
| `ffi` | UniFFI facade (feature-gated) |

## Development

The lint bar matches the sibling Rust repos and is enforced by
`.cargo/config.toml` (`-Dwarnings` + clippy `all`/`pedantic`/`nursery`), with
`missing_errors_doc` and `missing_panics_doc` left on because the error contract
of every public function is part of the FFI API.

```sh
cargo test
cargo clippy --all-targets
cargo check --no-default-features   # the "no network, no model files" build
```
