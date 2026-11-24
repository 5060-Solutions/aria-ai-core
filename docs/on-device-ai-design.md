# On-Device AI for Aria Softphones — Design Spec

**Status:** Draft for owner review — DESIGN ONLY, no implementation
**Date:** 2026-07-28
**Scope:** `aria-mobile-core`, `aria-ios`, `aria-android`, `softphone` (desktop), `rtp-engine`
**Author:** design pass over the four repos as they exist on 2026-07-28

---

## 0. The product ask

Two features, verbatim from the owner:

> **AI Engine — On-Device AI Setup**: Choose transcription and LLM models that run entirely on your iPhone. Select recognition language, pick a model size that fits your device, and enable AI-powered summaries — no API key, no cloud.
>
> **Call Insights — Smart Call Analysis**: After every call, read the full AI transcript and an LLM-generated summary. Catch what you missed, review key points, and build sharper communication habits — all completely offline.

Constraints the owner added: build it in the **shared core first**, then iOS, Android, and desktop. Optional feature, optional model download, needs UI.

---

## 1. Ground truth: what actually exists today

Everything below was read directly. This section matters because two assumptions in the ask turn out to be wrong, and the architecture has to route around them.

### 1.1 There is no single shared core across all three platforms

| Repo | SIP stack | Media stack | Consumes `aria-mobile-core`? |
|---|---|---|---|
| `aria-ios` | `aria-mobile-core` via UniFFI xcframework | `aria-mobile-core::media` | **Yes** |
| `aria-android` | `aria-mobile-core` via UniFFI + JNA | `aria-mobile-core::media` | **Yes** |
| `softphone` (desktop) | its own stack in `src-tauri/src/sip/` on top of `sip-5060` | `rtp-engine` with the `device` (cpal) feature | **No** |

`/Users/josh/dev/5060/softphone/src-tauri/Cargo.toml` has no `aria-mobile-core` dependency and no `aria-sip-core` dependency. The only Rust code genuinely shared by all three products today is `rtp-engine` and `sip-5060`.

**Consequence:** "build it in the shared core" cannot mean "put it in `aria-mobile-core`" — desktop would not get it. It has to mean a **new crate that all three depend on**. See §2.

### 1.2 `PlatformAudioBridge` — the tap point the owner pointed at

`/Users/josh/dev/5060/aria-mobile-core/src/media.rs:22-31`

```rust
pub trait PlatformAudioBridge: Send + Sync + 'static {
    fn on_playback_audio(&self, samples: Vec<i16>, sample_rate: u32);
    fn on_capture_audio(&self, sample_rate: u32, frame_size: u32) -> Vec<i16>;
}
```

The two call sites inside `MobileMediaSession::start_with_bridge` (`media.rs:108`) are where both legs of call audio already exist as decoded mono `i16` PCM:

- **Far end (remote):** `media.rs:196-207` — RTP payload → `decoder.decode(payload, &mut pcm)` → RMS level → `bridge_rx.on_playback_audio(pcm, sample_rate)`.
- **Near end (local mic):** `media.rs:258-271` — `bridge.on_capture_audio(sample_rate, frame_samples)` → RMS level → encode → send.

This is the right place, but the tap belongs **inside Rust at these two sites**, not inside the Swift/Kotlin implementations of the trait. Reasons in §3.1.

### 1.3 Desktop already has the exact equivalent, and better

`rtp-engine` already tees both legs for call recording:

- `/Users/josh/dev/5060/rtp-engine/src/recorder.rs` — `CallRecorder::new(path, 8000)`, `tx_handle()` (`:58`), `rx_handle()` (`:66`), `RecorderHandle::add_samples(&[i16])` (`:173`).
- Fed from `/Users/josh/dev/5060/rtp-engine/src/session.rs:784` (TX) and `:1231` (RX).
- Output is **stereo 16-bit WAV, TX = left channel, RX = right channel** — free speaker separation.
- `/Users/josh/dev/5060/rtp-engine/src/resample.rs` already has `StreamResampler` (`:154`), `resample_linear_i16` (`:119`), `f32_to_i16`, `i16_to_f32`.

The desktop's internal PCM pipeline is 8 kHz mono i16 (`codec_rate = 8000`, `samples_per_frame = 160`) regardless of whether Opus was negotiated in SDP.

### 1.4 Bug found while reading: incoming calls never start media on mobile

`AriaMobileEngine::make_call` calls `media_session.start_with_bridge(bridge)` at `/Users/josh/dev/5060/aria-mobile-core/src/lib.rs:510-515`.

`AriaMobileEngine::accept_incoming_call` (`lib.rs:383-449`) creates the session via `media::create_answer_session(...)` at `:405-406`, inserts it into `self.calls`, starts the status poll — and **never calls `start_with_bridge`**. Inbound calls therefore have no RX/TX threads at all.

This is out of scope for this spec but it is a hard blocker for the AI feature: half of all calls would produce no audio to transcribe. **Fix this before Phase 2.**

### 1.5 Per-platform state of the art

**iOS** (`/Users/josh/dev/5060/aria-ios`)
- Deployment target **16.0** (`project.yml`, `AriaSoftphone.xcodeproj/project.pbxproj`). Rules out SwiftData, the `Observable` macro, and iOS 26's `SpeechAnalyzer`/`SpeechTranscriber`.
- **No SPM at all.** No `Package.swift`. Rust is vendored as a static xcframework built by `/Users/josh/dev/5060/aria-ios/build-rust.sh`; UniFFI Swift bindings land in `/Users/josh/dev/5060/aria-ios/AriaSoftphone/Bridge/aria_mobile.swift`. `project.yml` is the source of truth; the `.xcodeproj` is generated by XcodeGen.
- Audio: `/Users/josh/dev/5060/aria-ios/AriaSoftphone/Services/AudioBridge.swift` — `class iOSAudioBridge: PlatformAudioBridge`, AVAudioEngine, `installTap` on the input node, **nearest-neighbour decimation with no anti-alias filter**, a plain `[Int16]` array under `NSLock` as the capture buffer.
- Settings: `/Users/josh/dev/5060/aria-ios/AriaSoftphone/Views/Settings/SettingsView.swift` — SwiftUI, hand-rolled cards in a `ScrollView`, private helpers `sectionHeader(_:)` / `settingsRow(_:_:)`. `settingsRow` is **read-only label+value only** — no toggle/picker/progress row exists yet. Theme color `Color.ariaIndigo` (#6366F1).
- Secure storage: `/Users/josh/dev/5060/aria-ios/AriaSoftphone/Services/KeychainStore.swift` — `enum KeychainStore`, static `set/get/remove`, **String-only, no `Data` variant**.
- **Zero persistence.** No CoreData, no SQLite, no `FileManager` usage anywhere in app code. `CallHistoryEntry` (`Views/Call/HistoryView.swift:51`) is in-memory and never written to.
- Navigation: `MainTabView` in `App/ContentView.swift` — `TabView` with dialer/contacts/history/settings; `CallView` is a full-screen `ZStack` overlay driven by `callManager.activeCall != nil`. **No call-ended screen** — `CallManager.updateCallState` nils `activeCall` immediately on `.ended` (`CallKit/CallManager.swift:163-167`).
- No download infrastructure of any kind: no `URLSessionDownloadTask`, no background session, no `BGTaskScheduler`, no checksum verification.
- Entitlements: `aps-environment`, `pushkit.unrestricted-voip`. `UIBackgroundModes = [voip, audio]`. No `NSSpeechRecognitionUsageDescription`.

**Android** (`/Users/josh/dev/5060/aria-android`)
- minSdk 26, compileSdk/targetSdk 36, AGP 8.7.0, Kotlin 2.1.0, Java 17. 100% Compose + Navigation Compose.
- Rust: `/Users/josh/dev/5060/aria-android/build-rust.sh` builds **three ABIs** — `arm64-v8a`, `armeabi-v7a`, `x86_64` — and the `.so`s are **checked in** under `app/src/main/jniLibs/`. There is no Gradle↔Cargo integration and no `abiFilters`. FFI is **JNA**, not direct JNI.
- Audio: `/Users/josh/dev/5060/aria-android/app/src/main/kotlin/com/solutions5060/aria/service/AudioBridge.kt` — plain `AudioTrack`/`AudioRecord` (not Oboe), `VOICE_COMMUNICATION` source. Samples cross FFI as **boxed `List<Short>`** — already an expensive hot path.
- Settings: `.../ui/settings/SettingsScreen.kt` — `SectionHeader` + `Card` + `SettingsRow` + `SettingsDivider` helpers. Store is plain **SharedPreferences** (`aria_prefs`); no DataStore.
- Secure storage: `.../security/SecurePrefs.kt` — `object SecurePrefs` over `EncryptedSharedPreferences` (`aria_secure_prefs`), `getString/putString/remove/clear/migrateFrom`.
- Persistence: no Room, no SQLite. Call history is a JSON blob in SharedPreferences file `aria_history`, key `call_history`, capped at 100 (`.../ui/history/HistoryScreen.kt`, `loadHistory`/`saveHistory`).
- Navigation: `.../ui/AriaApp.kt`, `NavHost` with 4 tab routes; the call UI is a `showCallScreen` boolean overlay, not a route. **No call-ended screen.** `setEventHandler` is never called from Kotlin — the app polls `getActiveCall()` every 300–500 ms.
- Existing foreground service: `.../service/IncomingCallService.kt` (`phoneCall|microphone`). **No `FOREGROUND_SERVICE_DATA_SYNC`, no `WAKE_LOCK`, no WorkManager, no DownloadManager.** HTTP is OkHttp 4.12 with synchronous `.execute()`.
- De-facto global singleton: `object SipEngineHolder { var engine; var audioBridge }` at `.../service/AriaConnectionService.kt:227`.

**Desktop** (`/Users/josh/dev/5060/softphone`)
- Tauri v2, React 19 + TypeScript + MUI v9 + zustand + framer-motion + i18next (en/de/es/fr). Vite 8, pnpm.
- Rust: 50 `#[tauri::command]`s in `src-tauri/src/commands.rs`, registered in one `generate_handler!` in `src-tauri/src/lib.rs`. Events Rust→JS go through a single `mpsc` forwarding task in `lib.rs .setup()` emitting `sip-registration`, `sip-call`, `sip-diagnostic`, etc.
- Capabilities (`src-tauri/capabilities/default.json`) grant only `core:default`, `opener:default`, `dialog:default`. **The webview has no fs and no http permission** — all disk and network work must go through a Rust command. This is a good default and should not change.
- Settings UI: `src/components/settings/Settings.tsx` (1497 lines) — a `List` of `ListItemButton` + `Collapse` sections. Adding an "AI" section is a drop-in.
- Persistence: **localStorage only**, via zustand setters in `src/stores/appStore.ts`. `CallHistoryEntry` type at `src/types/sip.ts:126` already has `recordingPath?` and `sipCallId?`. History rows are appended in `src/hooks/useSip.ts:149-165` on `sip-call` with `state === "ended"`.
- Navigation: no router. `src/components/layout/AppShell.tsx` swaps a `views` map keyed by `currentView` (`dialer | history | contacts | settings | diagnostics`). **No call-ended screen** — `setActiveCall(null)` on ended. Nearest analogue is the details `Dialog` in `src/components/history/CallHistory.tsx`.
- Secure storage: `keyring` v3 with native backends, service `"aria-softphone"` (`commands.rs:783-915`).
- No download-with-progress pattern exists anywhere. `reqwest` is already a dependency (needs `stream` added).

---

## 2. Architecture

### 2.1 Recommendation

**Create one new Rust crate, `aria-ai-core`, that owns 100% of the AI logic, and have all three products depend on it. Do the inference in Rust using `whisper.cpp` (via `whisper-rs`) for STT and `llama.cpp` (via `llama-cpp-2`) for summarization, both on GGUF/GGML weights, with the ggml hardware backend selected per-platform by Cargo feature.**

```
                     ┌─────────────────────────────┐
                     │        aria-ai-core         │   ← NEW crate, all logic
                     │  ────────────────────────   │
                     │  audio::CallAudioTap        │  two-leg ring buffers
                     │  audio::LegResampler        │  → 16 kHz mono f32
                     │  models::Catalog            │  catalog + device gating
                     │  models::Downloader         │  resume/verify/cancel
                     │  models::Store              │  paths, eviction, integrity
                     │  stt::Transcriber           │  whisper-rs
                     │  llm::Summarizer            │  llama-cpp-2
                     │  store::InsightStore        │  SQLite + AES-GCM blobs
                     │  AiEngine (facade)          │  one object, one state machine
                     └──────────┬──────────────────┘
              ┌─────────────────┼──────────────────┐
              │                 │                  │
   ┌──────────▼────────┐  ┌─────▼──────────┐  ┌────▼────────────────┐
   │ aria-mobile-core  │  │ aria-mobile-   │  │ softphone/src-tauri │
   │ (UniFFI → Swift)  │  │ core (→Kotlin) │  │ (#[tauri::command]) │
   └──────────┬────────┘  └─────┬──────────┘  └────┬────────────────┘
              │                 │                  │
        ┌─────▼─────┐     ┌─────▼─────┐      ┌─────▼─────┐
        │  aria-ios │     │aria-android│      │ React UI  │
        └───────────┘     └───────────┘      └───────────┘
```

`aria-ai-core` is a plain Rust library with **no UniFFI in it**. `aria-mobile-core` owns the UniFFI surface for mobile (adding to `src/aria_mobile.udl`); `softphone/src-tauri` owns the Tauri command surface for desktop. Both are thin adapters over the same `AiEngine` facade.

Repo layout decision (owner's call): a sibling repo `/Users/josh/dev/5060/aria-ai-core` published to the same GitHub org and consumed via `git` dependency, matching how `sip-5060` and `rtp-engine` are already consumed. That keeps the existing dependency style. Path dependencies for local dev.

### 2.2 Why inference belongs in the shared core, not per-platform

The question "portable one implementation vs. per-platform hardware acceleration" is a false dichotomy here, because **ggml already is the per-platform hardware acceleration layer**:

| Platform | ggml backend | Enabled by |
|---|---|---|
| iOS (A-series/M-series) | **Metal** | `whisper-rs`/`llama-cpp-2` `metal` feature; ships a `.metallib` |
| macOS | Metal | same |
| Windows | Vulkan, or CPU AVX2 | `vulkan` feature, runtime fallback to CPU |
| Linux | Vulkan / CPU | same |
| Android arm64 | CPU with NEON + i8mm/dotprod; optionally OpenCL (Adreno) | default CPU; OpenCL as a later opt-in |

So we get one implementation *and* GPU acceleration on the platforms where it matters most (iOS, macOS, Windows). Android is CPU-bound either way — the alternative (Gemini Nano via AICore) is unavailable on the vast majority of devices.

The decisive arguments:

1. **Only Rust can serve all three products.** iOS has strong native options; Android and desktop do not. Any platform-native choice means writing and maintaining the summarizer twice or three times regardless.
2. **One transcript format, one prompt, one output schema.** "Call Insights" must look identical on all three or the product is incoherent. Divergent STT engines produce visibly different transcripts for the same audio.
3. **The tap point is already in Rust.** §1.2 and §1.3 — both mobile and desktop already have the PCM in Rust. Sending it back out over UniFFI (as boxed `List<Short>` on Android, no less) to a platform-native recognizer is strictly worse.
4. **One model catalog, one downloader, one integrity story, one device-gating table.** These are ~60% of the actual work and are entirely platform-independent.
5. **The owner can develop and benchmark the whole thing on a laptop** before touching Xcode or Gradle. That is the single biggest schedule risk reducer here.

Trade-offs accepted:
- Binary size: `libwhisper` + `libllama` adds roughly 3–8 MB per architecture (Metal/Vulkan shaders add more). See §7.4.
- We do not get Apple's Neural Engine. ggml Metal uses the GPU, not ANE. Core ML whisper encoders exist for whisper.cpp and can be added later as an iOS-only optimization *without changing the architecture* — the ggml decoder stays.
- We carry the C++ build for three mobile targets. `build-rust.sh` on both mobile repos needs an NDK/iOS-SDK-aware cmake invocation.

### 2.3 Alternatives rejected

**Alternative A — Platform-native everywhere.** iOS: `SFSpeechRecognizer(requiresOnDeviceRecognition: true)` + an MLX or Core ML summarizer. Android: ML Kit / `SpeechRecognizer` + Gemini Nano via AICore. Desktop: ...nothing.

Rejected because:
- **Three implementations, three model catalogs, three quality profiles, three sets of bugs.** The maintenance cost is 3× forever, on a product with one owner.
- **Android has no on-device LLM story we can rely on.** Gemini Nano/AICore is Pixel-and-a-few-flagships only, and its availability is a runtime lottery. We would need llama.cpp on Android anyway — at which point we have already paid for the cross-platform path.
- **Desktop has no native option at all** on Windows/Linux. We would need whisper.cpp/llama.cpp there regardless.
- iOS 16 deployment target rules out `SpeechAnalyzer`/`SpeechTranscriber` (iOS 26). `SFSpeechRecognizer` on iOS 16 has real problems for this use case: on-device availability is per-locale and not guaranteed, historically a ~1 minute utterance ceiling requiring restart-and-stitch, no speaker attribution, no timestamps you can trust for merging two legs, and no control over the audio format path. It is also a *dictation* recognizer tuned for a single near-field speaker, not for 8 kHz telephony.
- The owner's copy says "Choose transcription and LLM models" and "pick a model size that fits your device." Apple's recognizer has no models to choose and no size to pick. The native path can't deliver the stated product.

**Alternative B — ONNX Runtime (or Candle) with ONNX/safetensors weights.** Run Whisper-ONNX and a small instruct model through `ort` or pure-Rust `candle`.

Rejected because:
- `ort` means a second large native dependency with its own per-platform execution providers (CoreML EP, NNAPI EP, DirectML) — the same per-platform matrix as ggml, but less mature for quantized LLM decode, and NNAPI is deprecated.
- Candle is pure Rust and genuinely attractive for build simplicity (no cmake, no C++), but its quantized-LLM decode throughput and its Metal kernels lag llama.cpp by a wide margin, and `candle-whisper` is not tuned the way `whisper.cpp` is for long-form with VAD and beam search. For a battery-sensitive mobile feature, throughput *is* the product.
- The GGUF ecosystem is where every small instruct model ships first. Picking ONNX narrows the model catalog we can offer, which directly attacks the "choose your model" feature.

**Alternative C — Hybrid: Apple Speech on iOS, whisper.cpp elsewhere.** Rejected: still needs whisper.cpp for two of three platforms, still needs llama.cpp on all three for summaries, and produces *different transcripts on different devices for the same call*. All of the cost of A, none of the savings of the recommendation.

**Alternative D — Ship the model in the app bundle instead of downloading it.** Rejected explicitly by the ask ("optional model download"), and by the numbers: the smallest useful pair is ~90 MB (whisper base q5_1) + ~400 MB (0.5B Q4_K_M). App Store cellular download limits and Play Store's 200 MB base AAB cap make bundling impossible for anything but the tiniest STT model. Downloading is not just nicer, it is mandatory.

### 2.4 `AiEngine` facade — the one object each platform binds to

Sketch (Rust, not final):

```rust
pub struct AiEngine { /* … */ }

impl AiEngine {
    pub fn new(cfg: AiConfig) -> Self;              // paths, device tier, keys

    // ── capability & config ──
    pub fn device_capability(&self) -> DeviceCapability;
    pub fn catalog(&self) -> Vec<ModelDescriptor>;   // filtered by capability
    pub fn installed_models(&self) -> Vec<InstalledModel>;
    pub fn settings(&self) -> AiSettings;
    pub fn set_settings(&self, s: AiSettings);

    // ── model lifecycle ──
    pub fn download_model(&self, id: &str) -> DownloadHandle;   // resume-aware
    pub fn cancel_download(&self, id: &str);
    pub fn delete_model(&self, id: &str) -> Result<(), AiError>;
    pub fn download_progress(&self, id: &str) -> Option<DownloadProgress>;

    // ── per-call ──
    pub fn begin_call_capture(&self, call_id: &str, consent: ConsentDecision)
        -> Option<Arc<CallAudioTap>>;                 // None if disabled/denied
    pub fn end_call_capture(&self, call_id: &str);    // seals the buffer, queues job
    pub fn job_status(&self, call_id: &str) -> InsightJobStatus;

    // ── results ──
    pub fn insight(&self, call_id: &str) -> Option<CallInsight>;
    pub fn delete_insight(&self, call_id: &str);
    pub fn purge_all(&self);
}
```

`DownloadHandle` and `InsightJobStatus` are polled, not callback-driven, in the first cut. Rationale: Android never wired up `setEventHandler` (§1.5) and both mobile apps already poll for call state; desktop's only Rust→JS path is the single `sip-*` event bus. Polling is the path of least resistance and least new plumbing. A callback/event upgrade is a Phase 6 nicety.

`CallInsight` is the shared result type:

```rust
pub struct CallInsight {
    pub call_id: String,
    pub created_at: i64,
    pub language: String,          // detected or forced
    pub duration_secs: u32,
    pub segments: Vec<TranscriptSegment>,   // merged, speaker-attributed
    pub summary: Option<Summary>,
    pub stt_model_id: String,
    pub llm_model_id: Option<String>,
    pub status: InsightStatus,     // Pending | Transcribing | Summarizing | Complete | Failed(reason)
}

pub struct TranscriptSegment {
    pub speaker: Speaker,          // Local | Remote
    pub start_ms: u32,
    pub end_ms: u32,
    pub text: String,
    pub confidence: f32,
}

pub struct Summary {
    pub headline: String,          // one line
    pub abstract_: String,         // 2–4 sentences
    pub key_points: Vec<String>,   // 3–7
    pub action_items: Vec<ActionItem>,   // { owner: Local|Remote|Unassigned, text }
    pub topics: Vec<String>,
}
```

---

## 3. Audio pipeline

### 3.1 Where the tap goes

**Mobile — tap inside `MobileMediaSession`, not inside the platform bridge implementations.**

Add to `/Users/josh/dev/5060/aria-mobile-core/src/media.rs`:

```rust
pub struct MobileMediaSession {
    // … existing fields …
    /// Optional AI capture tap. None when the feature is off — zero cost.
    ai_tap: Arc<Mutex<Option<Arc<aria_ai_core::CallAudioTap>>>>,
}
```

and two one-line fan-outs at the sites identified in §1.2:

- RX thread, immediately after the RMS calculation at `media.rs:200-206`, before `bridge_rx.on_playback_audio(pcm, sample_rate)`:
  `if let Some(t) = tap.as_ref() { t.push_remote(&pcm, sample_rate); }`
- TX thread, at `media.rs:261-270`, on the `captured` samples:
  `if let Some(t) = tap.as_ref() { t.push_local(&captured, sample_rate); }`

Why here and not in `iOSAudioBridge` / `AudioBridge.kt`:

1. **Written once, both platforms get it.** The alternative is duplicating ring-buffer + resampling logic in Swift and Kotlin.
2. **Android's FFI hot path stays untouched.** Samples already cross as boxed `List<Short>`; tapping in Kotlin means a second traversal of a boxed list per 20 ms frame per leg.
3. **iOS's capture path is lossy before the bridge returns.** `AudioBridge.swift` does nearest-neighbour decimation with no anti-alias filter from the hardware rate down to the codec rate. Tapping at the Rust site gets us the same samples the encoder sees, which is what the far end hears, which is what the transcript should reflect. (Improving that decimation is a separate, worthwhile fix — see §9.)
4. **`on_capture_audio` returns `[]` on underrun** (`AudioBridge.swift`) and the Rust side substitutes silence. Tapping in Rust after that substitution keeps the two legs time-aligned on the same 20 ms grid, which is what makes merge-by-timestamp work.

Note the TX site currently only calls `on_capture_audio` when not muted/held; when muted it substitutes a zero frame. Push the zero frame to the tap too — timeline continuity matters more than saving bytes, and "you were muted here" is legitimately useful.

**Desktop — reuse the recorder tap shape in `rtp-engine`.**

`rtp-engine` already tees exactly the right two points (`session.rs:784` TX, `session.rs:1231` RX) into `RecorderHandle`. Two options:

- **(Recommended)** Generalise: give `MediaSession` a `Vec<Arc<dyn PcmSink>>` per leg instead of a single `Option<RecorderHandle>`, where `RecorderHandle` becomes one implementation and `CallAudioTap` another. Small, clean change to a crate we own.
- (Fallback if `rtp-engine` changes are undesirable) Have the AI feature turn on call recording to a temp WAV and transcribe the file after hangup. Works today with zero `rtp-engine` changes, but couples the AI feature to the user-visible recording feature and doubles disk writes. Acceptable for a Phase-3 spike, not for ship.

### 3.2 Two legs, never mixed

The single most important pipeline decision: **the two legs are never summed.** Speaker attribution is free if and only if `local` and `remote` stay in separate buffers all the way through STT. Whisper does no diarization; running it twice — once per leg — and merging by timestamp gives perfect two-speaker attribution at the cost of ~2× inference time.

This is not free of edge cases:
- **Echo/leakage.** The far end's voice can bleed into the local mic. Android's `VOICE_COMMUNICATION` source and iOS's `.voiceChat` mode both apply platform AEC, so leakage is typically low, but it is not zero. Mitigation: cross-leg energy gating — when the remote leg RMS is high and the local leg RMS is low-but-nonzero over the same window, suppress the local segment. Reuse the RMS values `media.rs` already computes (`rx_level`/`tx_level`).
- **Overlapping speech** produces genuinely overlapping segments. Render them as overlapping in the UI rather than forcing a linear order.

### 3.3 Rates, resampling, and the narrowband problem

| Source | Rate | Path to 16 kHz mono |
|---|---|---|
| PCMU / PCMA (mobile + desktop, `AudioCodec::Pcmu`/`Pcma`, clock 8000) | 8 kHz | ×2 upsample (polyphase, low-pass) |
| Opus on mobile (`AudioCodec::Opus`, clock 48000, 960-sample frames) | 48 kHz | ÷3 decimate (anti-aliased) |
| Opus on desktop | negotiated 48 kHz in SDP but the `rtp-engine` internal pipeline is `codec_rate = 8000` | effectively 8 kHz — see below |
| G729 (`AudioCodec::G729`) | 8 kHz | ×2 upsample |

**Call this out to the owner plainly: G.711 at 8 kHz is narrowband — everything above 4 kHz is simply absent.** Upsampling to 16 kHz gives Whisper the sample rate it wants but not the information it wants. Expect materially higher word error rate on PCMU/PCMA calls than on wideband ones — telephony-band WER for whisper-small is typically several points worse than for studio audio, and worse still for accented speech, names, numbers, and spelling. Two implications:

1. **Prefer Opus.** `AriaMobileEngine::new` already defaults `codec_prefs` to `[Opus, Pcmu, Pcma]` (`lib.rs:123-127`). Good. But the PBX/gateway leg has to preserve it — if the far end is PSTN, the call is narrowband no matter what. Surface the actual negotiated codec in the Call Insights UI ("Narrowband call — transcript accuracy may be reduced").
2. **The desktop's 8 kHz internal pipeline caps desktop quality even on Opus calls.** Widening `rtp-engine`'s internal rate is a real project and out of scope here; note it as a known ceiling.

Resampling: `rtp_engine::resample::StreamResampler` (`resample.rs:154`) is **linear interpolation**, fine for playback, not ideal ahead of an ASR frontend. Use a proper band-limited resampler in `aria-ai-core` (`rubato`, or a hand-rolled polyphase FIR — the ratios are the fixed integers 2 and 3, so a fixed-coefficient polyphase filter is cheap and exact). Whisper's frontend wants **16 kHz mono f32 in [-1.0, 1.0]**.

### 3.4 Buffering during the call

`CallAudioTap` per call, holding two independent lanes:

```
push_local(&[i16], rate) ──► SPSC ring (lock-free, preallocated)
                                │  drop-oldest on overflow, never block
                                ▼
                          drain thread (own OS thread, low priority)
                                │  resample → 16 kHz mono i16
                                ▼
                          encrypted spill file  <app>/ai/tmp/<call_id>.local.pcm
```

Hard requirement: **the RTP TX/TX threads must never block on the tap.** They are already running blocking work inside `tokio::runtime::Builder::new_current_thread()` blocks (`media.rs:127-133`, `:228-234`); adding disk I/O or a mutex-under-contention there produces audible glitches. `push_local`/`push_remote` must be a bounded lock-free write that drops on overflow and increments a counter.

Spill sizing at 16 kHz mono i16 = **32 KB/s per leg, 64 KB/s both = ~230 MB/hour**. That is too much to hold in RAM and enough to matter on disk.

Mitigations, in order:
1. **Store at the source rate, resample at inference time.** For an 8 kHz call that halves it to ~115 MB/hour. Recommended default.
2. **VAD-gate the spill.** Skip writing runs of silence longer than ~1 s, recording only a gap marker. Typical two-party calls are 40–60% silence per leg. Combined with (1), a 30-minute PCMU call lands around 25–35 MB.
3. Hard cap: `max_capture_minutes` (default 120). Past the cap, stop capturing and mark the insight `Truncated`.
4. Free-space precondition: refuse to start capture below a floor (e.g. 500 MB free) and surface it in the UI.

**File format:** interleave the two legs as a **stereo 16-bit WAV, local = left, remote = right**, at the codec rate. This is byte-for-byte the format `rtp_engine::recorder::CallRecorder` already writes (`recorder.rs`), so desktop can reuse the existing recorder plumbing verbatim, and the format is trivially inspectable during development. Encrypt at rest (§5.2).

### 3.5 Streaming vs. post-call batch

**Recommendation: post-call batch for v1. Live streaming is Phase 6, opt-in, and gated to high-tier devices only.**

Reasons:
- The product copy says "After every call, read the full AI transcript" — batch satisfies the stated feature completely.
- Whisper is an offline model with a 30-second receptive field. Streaming means sliding windows with overlap-and-stitch, unstable partial hypotheses that rewrite themselves on screen, and roughly 2–3× the total compute for the same call. It is a meaningfully harder product, not just a harder implementation.
- **Thermal and audio-glitch risk during a live call is the real objection.** Running a GPU/CPU-saturating inference loop concurrently with a hard-realtime 20 ms RTP loop, on a phone, on battery, with an active radio, is exactly the scenario that produces dropped audio and a hot device. The call is the product; the transcript is not. Never risk the call.
- Batch lets us run inference **after** the call ends, when we can request a foreground/background execution window explicitly and even defer to charging.
- Batch gives whisper full-file context, VAD-based segmentation, and the option of beam search, all of which improve accuracy over greedy streaming.

What batch costs: the user waits. A 10-minute call with whisper-base on an A17 is a few tens of seconds; whisper-small on a mid-range Android could be a couple of minutes. Mitigate with an explicit, honest progress UI (§6) and a notification when it's ready — do not pretend it's instant.

### 3.6 End-to-end job

```
call ends
  └─ AiEngine::end_call_capture(call_id)
       ├─ flush + seal both lanes, fsync, close
       ├─ enqueue InsightJob{call_id}
       └─ InsightStore: upsert row status=Pending

worker (single-threaded queue, one job at a time, never concurrent with a live call)
  1. status=Transcribing
  2. load STT model (mmap; keep hot for N minutes then unload)
  3. per leg: decode WAV → resample→16 kHz f32 → VAD-segment → whisper_full()
  4. merge legs by start_ms; apply cross-leg echo gate; assign Speaker
  5. persist segments; status=Summarizing (skip if summaries disabled or no LLM installed)
  6. load LLM model; chunk (§4.3); map-reduce; parse JSON
  7. persist Summary; status=Complete
  8. shred + delete the spill WAV   ← unconditional, even on failure
  9. notify platform (poll flag flips; optional local notification)
```

The queue is strictly serial and **refuses to run while a call is active**. Job survives app restart (status is in SQLite; the spill file is on disk); on launch, resume any `Pending`/`Transcribing` job. Cap retries at 1, then `Failed`, and still delete the audio.

---

## 4. Summarization

### 4.1 Model class

Small instruct models, GGUF, Q4_K_M quantization. Candidates for the catalog (exact picks are a benchmarking task, not a design decision):

| Tier | Example | Approx. Q4_K_M size | Notes |
|---|---|---|---|
| Compact | Qwen2.5-0.5B-Instruct | ~400 MB | Weak at structured output; needs heavy prompt scaffolding |
| Standard | Llama-3.2-1B-Instruct | ~800 MB | Good size/quality knee for phones |
| Better | Qwen2.5-1.5B-Instruct | ~1.0 GB | Noticeably better key-point extraction |
| Desktop / high-end only | Llama-3.2-3B / Phi-3.5-mini | ~2.0–2.4 GB | Best quality; 6 GB+ RAM devices only |

Licensing check is required per model before shipping (Llama Community License has attribution/naming clauses; Qwen is Apache-2.0). **Owner decision — see §9.**

### 4.2 Prompt design

Constraints: small models are bad at long instructions, bad at unprompted JSON, and prone to inventing action items. So: short system prompt, explicit schema, low temperature, grammar-constrained decoding.

Use **GBNF grammar constraint** (llama.cpp supports it natively via `llama-cpp-2`) to force syntactically valid JSON. This eliminates the single largest failure mode of 0.5–1.5B models — malformed output — and is the main reason to prefer llama.cpp over a generic runtime for this job.

System prompt (v1 draft):

```
You summarize telephone calls. You are given a transcript with two speakers:
"Me" (the phone's owner) and "Them" (the other party).

Rules:
- Use only information stated in the transcript. Never infer or invent.
- If something is unclear or the transcript is garbled, say so rather than guessing.
- An action item must be an explicit commitment or request made by a speaker.
  If there are none, return an empty list. Do not manufacture action items.
- Write in the same language as the transcript.
- Output JSON only, matching the given schema.
```

User message:

```
Call with {remote_name_or_number} on {date}, duration {mm:ss}.

TRANSCRIPT
[00:04] Them: …
[00:11] Me: …
…
```

Sampling: `temperature 0.2`, `top_p 0.9`, `repeat_penalty 1.05`, `n_ctx 4096` (8192 on desktop/high-end), `n_predict` capped at 512 for the reduce step.

### 4.3 Output schema

```json
{
  "headline": "string, <= 80 chars",
  "abstract": "string, 2-4 sentences",
  "key_points": ["string", "..."],
  "action_items": [{ "owner": "me" | "them" | "unassigned", "text": "string" }],
  "topics": ["string", "..."]
}
```

Enforced by GBNF grammar plus a Rust-side validator: clamp `key_points` to 7, `action_items` to 10, strip empties, reject items longer than 200 chars. If parsing still fails after one retry, persist `abstract` only from a plain-text fallback prompt, and mark the summary `degraded` in the UI rather than showing nothing.

### 4.4 Long calls — chunking

Token budget arithmetic: conversational speech runs ~140–160 wpm per active speaker; a two-party call is roughly 130–170 words/minute of transcript total. At ~1.35 tokens/word plus timestamp/speaker overhead, budget **~250 tokens per minute of call**. With `n_ctx = 4096` and ~600 tokens reserved for system + schema + output, a single pass tops out around **13–14 minutes of call**.

Strategy — **map-reduce with overlap**:

- **Map.** Split the transcript into windows of ~1,800 tokens (≈7 minutes) with ~150 tokens of overlap, split preferentially at speaker-turn boundaries and at silence gaps > 3 s. For each window emit a compact intermediate: `{ abstract (≤3 sentences), key_points (≤4), action_items }`.
- **Reduce.** Feed the concatenated intermediates (plus the first and last 60 s of raw transcript, which carry disproportionate signal — greeting/identification and close/commitments) into the final schema prompt.
- **Recurse** only if intermediates themselves exceed the window, i.e. calls beyond roughly 90 minutes.
- **Short-circuit:** calls under the single-pass ceiling skip map entirely — one call to the model.
- **Floor:** calls under 30 seconds of speech get no summary at all. Show the transcript and a "too short to summarize" note. Small models hallucinate freely on thin input.

Cost note: map-reduce multiplies inference time by roughly (n_windows + 1). A 45-minute call is 7 windows + 1 reduce = 8 generations. On a phone that is minutes. Show it honestly in the progress UI and consider defaulting long-call summarization to "when charging" on mobile (§7).

---

## 5. Storage & privacy

### 5.1 Where things live

| | iOS | Android | Desktop |
|---|---|---|---|
| Models | `Application Support/ai/models/` | `context.filesDir/ai/models/` | `app_data_dir()/ai/models/` |
| Spill audio (transient) | `Application Support/ai/tmp/` | `context.filesDir/ai/tmp/` | `app_data_dir()/ai/tmp/` |
| Insight DB | `Application Support/ai/insights.sqlite` | same relative | same relative |
| Encryption key | `KeychainStore` | `SecurePrefs` | `keyring` service `"aria-softphone"` |

Notes:
- iOS: set `URLResourceValues.isExcludedFromBackup = true` on the models directory (Apple rejects apps that back up re-downloadable content) and `.completeUntilFirstUserAuthentication` file protection on the insight DB and spill files. Models can be `.completeUntilFirstUserAuthentication` too so a background job can mmap them.
- Android: **internal storage only** — no `WRITE_EXTERNAL_STORAGE`, no scoped-storage complexity. `allowBackup="false"` is already set in the manifest, so nothing leaks into cloud backup.
- Desktop: models can be large; offer a "change model location" path picker on desktop only.
- `KeychainStore` is **String-only today** — either add a `Data` variant or base64 the 32-byte key. Flagged as a small required change.

### 5.2 Encryption at rest

- **Key:** 256-bit, generated once per install by the core, stored in the platform secure store (above). iOS uses `kSecAttrAccessibleWhenUnlockedThisDeviceOnly` today; the AI key needs `AfterFirstUnlockThisDeviceOnly` so post-call jobs can run with the screen locked. Non-exportable, never synced.
- **Transcripts and summaries:** stored as AES-256-GCM blobs. Recommendation: keep SQLite plain (schema, timestamps, `call_id`, model ids, status) and encrypt only the text payload columns. Rationale: avoids taking on SQLCipher as a native dependency across five build targets, keeps the DB inspectable for debugging, and the sensitive content is the text. If the owner wants full-DB encryption, SQLCipher is the answer and it is a Phase-6 change, not a v1 blocker.
- **Spill audio:** AES-256-GCM in 1 MB chunks with per-chunk nonces so it can be streamed without buffering the whole file. Deleted immediately after transcription, on every exit path including failure and app kill (delete orphaned `ai/tmp/*` on launch).
- **Models are not encrypted.** They are public artifacts; encrypting them buys nothing and costs decrypt-on-every-load. Integrity is the concern there, not confidentiality (§6.3).

### 5.3 Retention

- Default retention: **30 days**, then automatic deletion. Options: 7 / 30 / 90 / Forever.
- Deleting a call-history entry deletes its insight. This requires that both are keyed by the same id — today they are not (Android generates a fresh `UUID.randomUUID()` at save time in `AriaApp.kt`, not the SIP call id). **The key needs to be unified as part of Phase 2.**
- Settings actions: "Delete all transcripts and summaries", "Delete all downloaded models", "Turn off AI Engine" (which offers to do both).
- Uninstall removes everything on all three platforms (internal storage / app container / app data dir).

### 5.4 The privacy guarantee

Stated plainly in the UI, and true by construction:

> Transcription and summarization run entirely on this device. Your call audio, transcripts, and summaries are never uploaded, never sent to a server, and never used to train anything. The only network request this feature makes is downloading the model files you choose.

Enforcement, so the claim survives audit:
- `aria-ai-core` has exactly one network capability: the model downloader, restricted to a **compile-time host allowlist**. No other HTTP client in the crate. (`aria-ios` already has host-validation prior art to mirror: `isValidProvisioningDestination` / `isValidHost` in `Views/Settings/QRScannerView.swift:78,98`.)
- No telemetry, no crash-report attachment of transcript content, no logging of transcript text at any log level. Add a lint/CI check that transcript-bearing types don't implement `Display`/`Debug` in a way that reaches the log.
- Desktop: the webview still has no `fs`/`http` capability (`src-tauri/capabilities/default.json`) — do not add any.

### 5.5 Consent and the law — the real risk

This is the highest-risk part of the feature and it is mostly a legal/product problem, not a technical one.

**Facts that matter:**
- The pipeline **records call audio to disk**, even if only briefly. Any description that implies otherwise ("we only make notes") is not accurate and should not be used.
- In the US, federal law and most states are one-party consent, but a meaningful set are **all-party consent** — California, Florida, Illinois, Pennsylvania, Washington, Maryland, Massachusetts, Montana, New Hampshire, Connecticut, Delaware, Oregon (with nuances), Nevada (per state courts). Interstate calls raise which-law-applies questions with no clean answer.
- Outside the US: recording without consent is broadly restricted under GDPR (recording is processing personal data and needs a lawful basis), and several jurisdictions (Germany §201 StGB, France, Canada's PIPEDA, Australia's state Surveillance Devices Acts) are stricter still.
- The app is a **business PBX client**, so calls will routinely cross state and national lines and involve customers, not just the user's own contacts.
- App Store Review 5.1.1 and Play's policies both require clear disclosure and, in practice, an in-app consent surface for call recording.

**Recommended UI handling:**

1. **Default OFF.** The AI Engine ships disabled. No models are downloaded until the user explicitly enables it.
2. **One-time legal acknowledgement** on first enable — a blocking modal, not a checkbox in a settings row. Plain language, and it must state that audio is recorded to the device during the call. Requires an affirmative tap ("I understand and I'm responsible for obtaining consent"). Record the acknowledgement (version + timestamp) in secure storage so it can be re-shown when the text changes.
3. **Capture mode** setting, three values:
   - `Off` (default)
   - `Ask each call` — a prompt at call connect: "Transcribe this call?" with Yes / No / Always for this contact
   - `Always on` — with an extra confirmation and an explicit jurisdiction warning at the moment it's selected
   **Recommendation: default `Ask each call` when the feature is first enabled.** It is the only mode that is defensible in a two-party jurisdiction without an audible announcement.
4. **Persistent, unmissable in-call indicator.** A recording chip in `CallView` / `CallScreen` / desktop `CallScreen`, always visible while capturing, with a one-tap stop that discards the capture immediately.
5. **Optional audible announcement**, default ON when capture mode is `Always on`: inject a short pre-recorded phrase into the TX leg at call connect ("This call may be transcribed"). Technically easy — mix a WAV into the TX path at the `media.rs:261` site for the first ~2 seconds. **Whether to make this mandatory is a product/legal decision (§9).**
6. **Per-contact opt-out** honoured before anything else.
7. **Never capture** when the codec/session indicates an emergency call, and never capture calls to numbers matching an emergency pattern.
8. **Admin/tenant policy hook.** This is a business PBX. A tenant admin may need to disable the feature org-wide for compliance. Design the setting so a server-pushed policy flag can force it off — the plumbing already exists (the gateway/PBX provisioning path). Not required in v1 but the flag should be in the data model from day one.

**This spec cannot substitute for legal review. Get counsel to sign off on the acknowledgement copy and the default capture mode before ship.**

---

## 6. Model management

### 6.1 Catalog

A signed JSON catalog, **bundled in the app binary** as the source of truth (so a compromised or unreachable host can't inject a model), with an optional signed remote refresh later.

```json
{
  "catalog_version": 3,
  "models": [
    {
      "id": "whisper-base-q5_1",
      "kind": "stt",
      "family": "whisper",
      "display_name": "Standard",
      "quality_tier": 2,
      "size_bytes": 59_000_000,
      "runtime_ram_mb": 210,
      "languages": ["multilingual"],
      "min_tier": "low",
      "sha256": "…",
      "url": "https://models.5060solutions.com/whisper/ggml-base-q5_1.bin"
    }
  ]
}
```

Indicative catalog (final sizes/picks require benchmarking):

**STT (whisper.cpp GGML, quantized)**

| Tier | Model | Approx. size | Approx. peak RAM | Min device tier |
|---|---|---|---|---|
| Fast | `tiny-q5_1` | ~32 MB | ~130 MB | Low |
| Standard | `base-q5_1` | ~59 MB | ~210 MB | Low |
| Accurate | `small-q5_1` | ~190 MB | ~600 MB | Medium |
| Best | `large-v3-turbo-q5_0` | ~575 MB | ~1.5 GB | High / desktop |

English-only `.en` variants are ~same size and measurably better for English — offer them when the user picks English as the recognition language rather than "Auto".

**LLM (GGUF Q4_K_M)** — see §4.1.

### 6.2 Device capability gating

Compute a `DeviceCapability { tier, ram_budget_mb, cpu_cores, has_gpu_backend, arch }` at startup in the core, fed by a small platform-provided struct:

- **iOS:** the number that matters is **`os_proc_available_memory()`**, not `ProcessInfo.physicalMemory`. iOS jetsams apps well below physical RAM, and a VoIP app that gets killed mid-call is a catastrophic failure. Budget conservatively: never allocate a model whose `runtime_ram_mb` exceeds ~50% of `os_proc_available_memory()` at the moment of load. Use mmap so the pages are file-backed and evictable rather than dirty.
- **Android:** `ActivityManager.MemoryInfo.totalMem` and `getLargeMemoryClass()`, plus `Build.SUPPORTED_ABIS`. **`armeabi-v7a` (32-bit) is excluded from LLM entirely** — a 4 GB address space with a 2 GB practical per-process ceiling cannot host a 1B model plus whisper plus a live call. Offer STT only (tiny/base) on 32-bit, and say so in the UI.
- **Desktop:** physical RAM and core count; be generous.

Proposed tiers:

| Tier | Criteria | Offered |
|---|---|---|
| Unsupported | 32-bit ABI, or < 2 GB RAM | Feature hidden with an explanation |
| Low | ≥ 3 GB RAM, arm64 | STT tiny/base. Summaries off or 0.5B with a warning |
| Medium | ≥ 6 GB RAM | STT up to small; LLM up to 1B |
| High | ≥ 8 GB RAM (A17/M-series, recent Snapdragon/Tensor) | STT up to turbo; LLM up to 1.5B |
| Desktop | ≥ 8 GB RAM | Everything, incl. 3B |

Models above the device tier are shown **greyed with the reason** ("Needs 6 GB RAM — your device has 4 GB"), not hidden. Users understand a locked door; they don't understand a missing option. Allow an "advanced: install anyway" escape hatch behind a warning on desktop only.

### 6.3 Download

Requirements, all in `aria-ai-core::models::Downloader`:

- **HTTPS only**, host allowlist, TLS via `rustls` (already the pattern in `aria-mobile-core`'s reqwest config).
- **Resumable:** write to `<id>.part`, track bytes, resume with `Range: bytes=N-`, verify the server honours it (206 vs 200) and restart cleanly if not.
- **Progress:** bytes done / total, instantaneous rate, ETA, polled via `download_progress(id)`.
- **Cancel:** cooperative flag checked per chunk; keeps the `.part` file so the user can resume, with a "discard partial download" action.
- **Integrity:** SHA-256 streamed during download and compared to the catalog value. Mismatch → delete, do not retry automatically, surface an error. This is the security boundary; the catalog hash is bundled in the signed binary, so a compromised CDN cannot ship us a malicious GGUF. (GGUF parsers have had CVEs; treat model files as untrusted input.)
- **Atomic install:** verify, then `rename()` into place. Never leave a half-file where the loader can find it.
- **Preconditions:** free space ≥ `size × 1.2`, and refuse to start on metered connections without explicit confirmation (iOS `NWPathMonitor.isExpensive`, Android `ConnectivityManager.isActiveNetworkMetered` — note `NetworkMonitor.kt` already exists on Android).
- **No download while a call is active.** Pause and resume after.

Platform execution:
- **iOS:** the download loop lives in Rust; the app must hold a `beginBackgroundTask` window and pause/resume across suspension. A background `URLSession` would survive suspension better but would split the implementation — accept foreground-only downloads in v1 and tell the user "keep Aria open."
- **Android:** needs a new foreground service with `foregroundServiceType="dataSync"` plus `FOREGROUND_SERVICE_DATA_SYNC` and `WAKE_LOCK` in the manifest (neither exists today). Model it on the existing `IncomingCallService.kt`.
- **Desktop:** straightforward; `reqwest` needs the `stream` feature added to `src-tauri/Cargo.toml`. Progress via a new Tauri event on the existing forwarding bus in `lib.rs`, or a polled `get_model_download_status` command matching the existing `get_audio_levels` polling precedent.

### 6.4 Eviction and deletion

- Explicit per-model delete, with a confirmation showing reclaimed space.
- Never auto-evict a model the user chose. Do auto-clean: orphaned `.part` files older than 7 days, `ai/tmp/*` on every launch, superseded catalog entries.
- If a model file fails its hash check at load time (corruption, partial restore), mark it invalid, delete it, and prompt to re-download. Do not feed a corrupt file to the GGUF parser.

---

## 7. Performance and battery

### 7.1 During the call

Only the tap runs: two lock-free ring writes per 20 ms frame, a fixed-ratio resample, a VAD gate, and a buffered encrypted write. Target **< 2% CPU**, no allocation in the hot path, no GPU. Effectively free.

The one real risk is disk I/O stalling the drain thread on a slow device. Buffer to 256 KB before writing and keep the drain thread off the RTP threads entirely.

### 7.2 After the call

Rough expectations (to be replaced by real measurements in Phase 1):

| Device class | whisper-base, 10 min call, both legs | 1B LLM summary, single pass |
|---|---|---|
| M-series Mac / desktop GPU | seconds | seconds |
| A17/A18 iPhone (Metal) | ~20–40 s | ~10–20 s |
| Recent flagship Android (CPU) | ~1.5–3 min | ~30–60 s |
| Mid-range Android (CPU) | ~4–8 min | ~2–4 min |

Real-time factor for whisper-base on mobile CPU is roughly 0.15–0.4× per leg, so two legs of a 10-minute call is 3–8 minutes of wall time on the slower end. **This is why batch, not streaming, and why the progress UI has to be honest.**

Battery: a sustained 2-minute all-core inference is on the order of 1–3% of a phone battery. Acceptable occasionally; not acceptable if it happens after every 2-minute call all day.

### 7.3 Policy

- **Serial queue, one job at a time, never during a call.** Non-negotiable.
- **Thermal awareness.** iOS: `ProcessInfo.thermalState` — pause at `.serious`, abort at `.critical`, resume when it drops. Android: `PowerManager.getCurrentThermalStatus()` / `addThermalStatusListener` (API 29+; on 26–28 just use conservative thread counts). Desktop: no throttling needed.
- **Battery/charging policy** setting: `Always` / `Only above 20% battery` (default on mobile) / `Only while charging`.
- **Thread budget:** `n_threads = min(perf_cores, 4)` on mobile. Never saturate all cores — that is what makes the device hot and the UI stutter. Desktop can use `cores - 2`.
- **Model residency:** mmap and keep the STT model resident for 5 minutes after the last job (back-to-back calls are common), then unload. Never keep both STT and LLM resident simultaneously on Low/Medium tiers — load, run, unload, load the next.
- **Low-end devices:** hide summaries entirely below Medium tier and offer STT-only. A 0.5B model producing bad summaries slowly is worse than no summaries. Say "Summaries need 6 GB of RAM" and stop there.
- **Deferral:** on mobile, calls longer than ~20 minutes default to deferring the LLM step until the device is charging and idle, with the transcript available immediately. Transcript-now / summary-later is a much better experience than a 6-minute spinner.

### 7.4 Binary size

- **iOS:** `libwhisper` + `libllama` + Metal shaders into the existing `AriaMobileCore.xcframework`, arm64 only. Expect **+8–15 MB** to the IPA. Acceptable.
- **Android:** three ABIs are currently built and checked in with no `abiFilters`. Adding the AI libs to all three multiplies the cost. **Recommendation: ship the AI native code for `arm64-v8a` only**, and use ABI splits or app bundles so the other ABIs don't carry it. `armeabi-v7a` gets the feature hidden (§6.2) — consistent with excluding 32-bit from LLM anyway. Expect **+6–10 MB** on the arm64 split.
- **Desktop:** immaterial.
- Model weights are never bundled. Downloads only.

---

## 8. UI/UX

Two surfaces on each platform: **Settings → AI Engine**, and **Call Insights**.

### 8.1 Settings → AI Engine (shared information architecture)

```
AI Engine                                          [ ● ]     ← master toggle, default OFF
  Transcribe and summarize calls entirely on this device.

── when enabled ────────────────────────────────────────────

TRANSCRIPTION
  Recognition language        English ▾          ← Auto-detect + ~15 named
  Model                       Standard (59 MB) ▾
     ○ Fast        32 MB   ~130 MB RAM   Lower accuracy
     ● Standard    59 MB   ~210 MB RAM   Recommended
     ○ Accurate   190 MB   ~600 MB RAM   Slower
     ⊘ Best       575 MB   Needs 8 GB RAM — your device has 6 GB
     [ Download ]  ▓▓▓▓▓▓░░░░ 61%  ·  38 MB / 59 MB  ·  2.1 MB/s  ·  [Cancel]

SUMMARIES
  AI summaries                [ ● ]
  Model                       Llama 3.2 1B (808 MB) ▾
  Run summaries               Only while charging ▾

CAPTURE
  Transcribe calls            Ask each call ▾     ← Off / Ask each call / Always on
  Announce to the other party [ ● ]               ← shown when "Always on"

STORAGE
  Models                      867 MB   [ Manage ]
  Transcripts & summaries      12 MB   ·  34 calls
  Keep transcripts for        30 days ▾
  [ Delete all transcripts ]   [ Delete all models ]

  🔒 Nothing leaves this device. Model downloads are the only network use.
```

Every model row shows **size on disk, RAM needed, and a plain-language quality hint**. Never a bare model name.

### 8.2 Call Insights

Reached from: the post-call sheet (immediately after hangup) and from a call-history row (any time later). Both render the same view.

```
  Sarah Chen · +1 415 555 0142
  Today 2:14 PM · 8:32 · Narrowband call ⓘ

  [ Summary ]  [ Transcript ]                        ← segmented control

  ── Summary ──────────────────────────────────
  Rescheduled the Q3 review and agreed on scope.

  Sarah asked to move the Q3 review from Thursday to the
  following Tuesday because of a customer escalation…

  KEY POINTS
   • Q3 review moved to Tuesday the 12th
   • Customer escalation at Northwind is the blocker
   • Budget sign-off still pending from finance

  ACTION ITEMS
   ☐ You — send the revised deck by Friday
   ☐ Sarah — confirm the room booking

  ── Transcript ───────────────────────────────
  00:04  Them   Hi, it's Sarah — do you have a minute?
  00:07  You    Sure, what's up?
  …

  [ Copy ]  [ Share ]  [ Delete ]
```

Design notes that apply everywhere:
- **Progress state is a first-class screen**, not a spinner: "Transcribing… 3 of 8 minutes" then "Writing summary…". Users tolerate a wait they can see.
- **Speaker labels are "You" and the contact's name**, not "Local"/"Remote".
- Timestamps are tappable in a later phase (jump to audio) — but v1 deletes the audio, so no playback. Do not build a UI that implies playback exists.
- **Degraded states are shown, not hidden**: "Narrowband call — accuracy may be reduced", "Summary unavailable — call too short", "Summary failed — [Retry]".
- Search across transcripts is deliberately deferred to Phase 6.

### 8.3 Per-platform

**iOS** (`aria-ios`)
- New: `Views/Settings/AIEngineView.swift`, pushed from a new nav row in `Views/Settings/SettingsView.swift`. This requires **new settings row components** — the existing `settingsRow(_:_:)` is read-only label+value. Add `settingsToggleRow`, `settingsPickerRow`, `settingsNavRow`, `modelDownloadRow`, keeping the existing card/`Color.ariaIndigo` styling.
- New: `Views/Call/CallInsightsView.swift`, presented two ways: as a `.sheet` from a new `@Published var lastEndedCall: EndedCall?` on `CallManager` (set at the three teardown sites — `updateCallState` `.ended` at `CallKit/CallManager.swift:163-167`, `handleRemoteHangup()`, and the `CXEndCallAction` handler), and pushed from a history row.
- **History has to become real.** `CallHistoryEntry` (`Views/Call/HistoryView.swift:51`) is in-memory and never populated. Persisting it is a prerequisite, not a nice-to-have.
- In-call indicator: a chip in `CallView`.
- `project.yml` is the source of truth for new files — regenerate with XcodeGen.
- No new usage-description strings are needed (mic is already declared) *unless* the design changes to use `SFSpeechRecognizer`, which it does not.

**Android** (`aria-android`)
- New: `ui/settings/AiEngineScreen.kt`, reachable from a new section in `ui/settings/SettingsScreen.kt`, reusing `SectionHeader` / `Card` / `SettingsRow` / `SettingsDivider`.
- New: `ui/insights/CallInsightsScreen.kt`. Wire it as a real `NavHost` route (`"insights/{callId}"`) in `ui/AriaApp.kt` rather than another boolean overlay — the existing `showCallScreen` pattern doesn't scale to a screen reachable from two places.
- The post-call trigger goes in the `CallScreen(onDismiss = …)` lambda in `AriaApp.kt`, where history is already written.
- Needs a new download foreground service + manifest permissions (§6.3).
- **Consider Room** for insights instead of extending the SharedPreferences JSON blob — a 100-entry JSON string is already the wrong shape for call history, and transcripts are far larger. At minimum, transcripts go to files keyed by call id, with only metadata in the DB.
- The insight/history join needs a stable id (§5.3).

**Desktop** (`softphone`)
- New collapsible "AI Engine" section in `src/components/settings/Settings.tsx`, matching the existing `ListItemButton` + `Collapse` pattern (the Audio devices section around line 871 is the template).
- New view `insights` added to the `View` union in `src/stores/appStore.ts:19` and to the `views` map in `src/components/layout/AppShell.tsx`; plus a transcript/summary panel inside the existing call-details `Dialog` in `src/components/history/CallHistory.tsx` (~lines 380–505), which already shows the recording row and is the natural home.
- Post-call: add `lastEndedCall` to the store and show the insights view when `sip-call` reports `ended` in `src/hooks/useSip.ts:149-165`.
- New Tauri commands, registered in the single `generate_handler!` in `src-tauri/src/lib.rs`: `ai_get_capability`, `ai_get_catalog`, `ai_get_settings`, `ai_set_settings`, `ai_download_model`, `ai_cancel_download`, `ai_download_progress`, `ai_delete_model`, `ai_get_insight`, `ai_get_job_status`, `ai_delete_insight`, `ai_purge_all`.
- **All strings go through i18next** (`src/locales/{en,de,es,fr}.json`) — desktop is the only localized surface today and the AI section must not regress that.
- **Desktop-only affordances:** a model storage location picker, "install anyway" override for over-tier models, and larger models in the catalog. Desktop has no thermal or memory ceiling worth defending.
- Do **not** add `fs` or `http` capabilities to the webview.

### 8.4 Where the platforms genuinely differ

| | iOS | Android | Desktop |
|---|---|---|---|
| Largest offered model | 1.5B | 1B (arm64 only) | 3B |
| Download while backgrounded | No (foreground task window) | Yes (dataSync FGS) | Yes |
| Inference while backgrounded | Limited; defer to next foreground | Yes via FGS | Yes |
| Battery/charging policy | Shown | Shown | Hidden (irrelevant) |
| Model storage location picker | No | No | Yes |
| Localization | English only today | English only today | 4 languages |
| Feature hidden entirely | < 3 GB RAM | 32-bit ABI or < 3 GB | Never |

---

## 9. Phasing

Each phase is independently shippable and independently valuable. Ordering is chosen so the riskiest, most reusable work happens where iteration is cheapest.

### Phase 0 — Decisions and prerequisites (owner + small fixes)
- Legal review of the consent model and acknowledgement copy (§5.5). **Blocks everything user-facing.**
- Model licensing decisions (§4.1).
- **Fix `accept_incoming_call` not calling `start_with_bridge`** (`aria-mobile-core/src/lib.rs:383-449`, §1.4). Hard blocker.
- Unify the call-history id with the SIP call id on all three platforms (§5.3).
- Decide the `aria-ai-core` repo location and dependency style.

*Exit:* decisions written down; inbound calls have audio; history rows have stable ids.

### Phase 1 — `aria-ai-core`, headless (biggest phase, no app changes)
- New crate: catalog, downloader (resume/verify/cancel), model store, device-capability struct, `stt::Transcriber` on `whisper-rs`, `llm::Summarizer` on `llama-cpp-2` with GBNF-constrained JSON, chunking/map-reduce, `store::InsightStore` (SQLite + AES-GCM), the `AiEngine` facade.
- A `cargo run --bin aria-ai-cli -- transcribe call.wav` harness for development.
- Benchmark the real model matrix on a Mac and fill in §4.1/§6.1/§7.2 with measured numbers.
- Unit tests on fixture WAVs; golden tests on summary structure.

*Ships:* nothing user-visible. *Value:* every downstream phase becomes UI work.
*Risk retired:* model quality, model sizes, inference speed, JSON reliability, chunking — all answered on a laptop.

### Phase 2 — Audio capture plumbing
- `CallAudioTap` (two-leg rings, resample, VAD gate, encrypted stereo WAV spill).
- Wire into `aria-mobile-core/src/media.rs` at the two identified sites, behind an `Option` so it costs nothing when off.
- Generalise `rtp-engine`'s recorder tap to a `PcmSink` list (§3.1) and wire the desktop.
- Verify: a real call on each platform produces a correct stereo WAV with legs aligned.

*Ships:* nothing user-visible. *Value:* the hard realtime-safety work is done and validated.

### Phase 3 — Desktop vertical slice ← **build here first**
Full feature on `softphone`: AI Engine settings section, model download with progress, post-call Insights view, history integration, i18n.

Why desktop before iOS, despite the ask being iPhone-worded:
- No app-store review loop, no code signing, no xcframework rebuild — the iteration cycle is seconds, not minutes.
- No memory ceiling, no jetsam, no thermal throttling — the feature can be made *correct* before it has to be made *small*.
- `rtp-engine` already has the tap points and a stereo-WAV recorder; the desktop has the shortest path from Phase 2 to a working screen.
- The React/MUI settings pattern makes the richest UI the cheapest to build, so UX can be settled once and then ported down.

*Ships:* a complete, sellable feature on desktop.

### Phase 4 — iOS vertical slice
- UniFFI surface in `aria_mobile.udl` + `AriaMobileEngine` passthroughs.
- `build-rust.sh` extended to build whisper/llama with Metal for `aarch64-apple-ios` and `-sim`.
- `AIEngineView.swift`, new settings row components, `CallInsightsView.swift`, `lastEndedCall` on `CallManager`.
- **Persist call history on iOS** (currently entirely in-memory — prerequisite).
- Memory gating on `os_proc_available_memory()`, thermal handling, backup exclusion, file protection.

*Ships:* the flagship platform, with the UX already proven on desktop.

### Phase 5 — Android vertical slice
- Same UniFFI surface (free — same UDL).
- `build-rust.sh` extended for `aarch64-linux-android`; **arm64-only** for the AI libs, ABI splits.
- `AiEngineScreen.kt`, `CallInsightsScreen.kt`, `insights/{callId}` route.
- New `dataSync` foreground service + `FOREGROUND_SERVICE_DATA_SYNC` / `WAKE_LOCK` permissions.
- Room (or files + metadata) for insights.
- 32-bit exclusion messaging.

*Ships:* feature parity.

### Phase 6 — Enhancements (each independent)
- Live/streaming transcription, opt-in, High tier only.
- Full-text search across transcripts.
- Core ML whisper encoder on iOS (ANE) as a drop-in optimization.
- Event/callback progress instead of polling; fix Android's unused `setEventHandler`.
- Widen `rtp-engine`'s internal pipeline beyond 8 kHz for wideband desktop calls.
- Tenant-admin policy enforcement from the PBX.
- Export to Markdown / calendar / task apps.
- SQLCipher full-DB encryption.

**Recommended build order: 0 → 1 → 2 → 3 → 4 → 5 → 6.**

---

## 10. Risks and open questions

### Needs the owner's decision (product/legal, not technical)

1. **Default capture mode.** `Ask each call` (recommended, defensible) versus `Always on` (better product, legal exposure in all-party-consent jurisdictions). This choice shapes the whole consent UI.
2. **Mandatory audible announcement?** Making "This call may be transcribed" non-optional in `Always on` mode is the strongest legal position and the worst user experience. Counsel should weigh in.
3. **Tenant admin override.** Should a PBX admin be able to force the feature off org-wide? Recommendation: yes, and put the flag in the data model now even if enforcement ships later.
4. **Model licensing.** Llama Community License has attribution and naming requirements that touch the UI and the About screen; Qwen is Apache-2.0 and simpler. Do we take on Llama at all?
5. **Model hosting.** Self-host on a 5060 CDN (control, cost, integrity) versus pull from Hugging Face (free, but a third-party availability and rate-limit dependency, and the URL implies a relationship). Recommendation: self-host, mirror from HF.
6. **Is this free or paid?** An "AI Engine" is the kind of thing that gates a tier. It changes nothing technically but changes the settings UI (upsell state) and whether the tenant policy flag is a licence check.
7. **Which languages ship in the picker?** Whisper supports ~99; realistically ship Auto + ~15 and expand on demand. The LLM's multilingual summary quality at 1B is uneven — a summary in a language the model handles poorly is worse than no summary. Recommendation: STT in many languages, summaries restricted to a validated shortlist in v1.
8. **Desktop-first build order.** Contradicts the iPhone-worded ask. Justified in Phase 3, but it is the owner's call.

### Technical risks

9. **Narrowband accuracy may disappoint.** 8 kHz G.711 is the dominant case for PSTN-terminated calls and Whisper's WER there is materially worse than the demos people have seen. **This is the single biggest risk to the feature's perceived quality.** Mitigate by benchmarking on real narrowband call recordings in Phase 1 *before* committing to the rest of the plan, and by setting expectations in the UI. If Phase 1 measurements are bad, that is a legitimate reason to stop.
10. **Small-model summary quality.** A 1B model on a garbled narrowband transcript can produce confident nonsense, especially fabricated action items. Grammar constraints fix the format, not the content. Phase 1 must include a human review of summaries from ~20 real calls before Phase 3 commits to shipping summaries at all.
11. **iOS memory pressure.** A VoIP app that gets jetsammed is a broken phone. Conservative budgets, mmap, never load two models at once, and hard-abort inference on memory warnings. Prove it with a soak test.
12. **Build complexity.** Neither mobile repo has Cargo↔build-system integration; `.so`s and generated bindings are checked in by hand on Android. Adding two cmake-built C++ dependencies to that will hurt. Consider fixing the build integration in Phase 4/5 rather than layering onto the manual scripts.
13. **App review.** Both stores scrutinize call recording. Expect at least one rejection round; the consent modal and the privacy copy are what get reviewed. Budget time.
14. **Disk usage.** Models plus transcripts plus spill audio can reach 2–3 GB. Users will blame the app for a full phone. Make the storage screen prominent and the numbers honest.
15. **Existing audio-path quality.** `iOSAudioBridge`'s unfiltered nearest-neighbour decimation adds aliasing to the mic leg — it degrades the *call* today and would degrade the transcript. Worth fixing on its own merits; it is a small change with a real payoff for this feature.

---

## Appendix A — Files this design touches

**New**
- `/Users/josh/dev/5060/aria-ai-core/` (whole crate)
- `aria-ios`: `AriaSoftphone/Views/Settings/AIEngineView.swift`, `AriaSoftphone/Views/Call/CallInsightsView.swift`, a persistence layer
- `aria-android`: `.../ui/settings/AiEngineScreen.kt`, `.../ui/insights/CallInsightsScreen.kt`, `.../service/ModelDownloadService.kt`, a persistence layer
- `softphone`: `src/components/settings/AiEngineSettings.tsx`, `src/components/insights/CallInsights.tsx`, `src-tauri/src/ai.rs`

**Modified**
- `/Users/josh/dev/5060/aria-mobile-core/src/media.rs` — tap at `:196-207` and `:258-271`
- `/Users/josh/dev/5060/aria-mobile-core/src/lib.rs` — `AiEngine` wiring; **fix `accept_incoming_call` at `:383-449`**
- `/Users/josh/dev/5060/aria-mobile-core/src/aria_mobile.udl` — AI interface
- `/Users/josh/dev/5060/aria-mobile-core/Cargo.toml`, both `build-rust.sh` scripts
- `/Users/josh/dev/5060/rtp-engine/src/session.rs` (`:784`, `:1231`), `src/recorder.rs` — `PcmSink` generalization
- `/Users/josh/dev/5060/aria-ios/project.yml`, `.../Views/Settings/SettingsView.swift`, `.../CallKit/CallManager.swift`, `.../Views/Call/HistoryView.swift`, `.../Services/KeychainStore.swift` (add `Data` API)
- `/Users/josh/dev/5060/aria-android/app/src/main/AndroidManifest.xml`, `app/build.gradle.kts`, `.../ui/settings/SettingsScreen.kt`, `.../ui/AriaApp.kt`, `.../ui/history/HistoryScreen.kt`
- `/Users/josh/dev/5060/softphone/src-tauri/Cargo.toml` (reqwest `stream`), `src-tauri/src/lib.rs`, `src/stores/appStore.ts`, `src/components/settings/Settings.tsx`, `src/components/layout/AppShell.tsx`, `src/hooks/useSip.ts`, `src/components/history/CallHistory.tsx`, `src/locales/*.json`

**Explicitly not modified**
- `/Users/josh/dev/5060/softphone/src-tauri/capabilities/default.json` — the webview must not gain `fs` or `http`.
